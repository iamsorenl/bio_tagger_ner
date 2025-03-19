import random
import time
from collections import defaultdict
from conlleval import evaluate as conllevaluate

random.seed(1234)
START = '<START>'
STOP = '<STOP>'
EPOCHS = 3



def backtrack(viterbi_matrix, tagset, max_tag):
    tags = []
    for k in reversed(range(len(viterbi_matrix))):
        last_max_tag_idx = tagset.index(max_tag)
        viterbi_list = viterbi_matrix[k]
        max_tag, _ = viterbi_list[last_max_tag_idx]
        tags = [max_tag] + tags
    return tags

def decode(input_length, tagset, score):
    tags = []
    viterbi_matrix = []

    # Initial step
    initial_list = []
    for tag in tagset:
        tag_score = score(tag, START, 1)  # Use different name to avoid conflict
        initial_list.append((START, tag_score))
    viterbi_matrix.append(initial_list)

    # Recursion step
    for t in range(2, input_length - 1):
        viterbi_list = []
        for tag in tagset:
            max_tag = None
            max_score = float("-inf")
            for prev_tag in tagset:
                last_viterbi_list = viterbi_matrix[t - 2]
                prev_tag_idx = tagset.index(prev_tag)
                last_score = last_viterbi_list[prev_tag_idx][1]
                tag_score = score(tag, prev_tag, t) + last_score  # Use 'tag_score' instead of 'score'
                if tag_score > max_score:
                    max_score = tag_score
                    max_tag = prev_tag
            viterbi_list.append((max_tag, max_score))  # Use 'max_score' instead of 'score'
        viterbi_matrix.append(viterbi_list)

    # Termination step
    tags = [STOP] + tags

    # Calculate the max tag
    last_viterbi_list = []
    for tag in tagset:
        stop_score = score(STOP, tag, input_length - 1)
        prev_score = viterbi_matrix[-1][tagset.index(tag)][1]
        final_score = stop_score + prev_score  # Use 'final_score' instead of 'score'
        last_viterbi_list.append((tag, final_score))
    max_tag, _ = max(last_viterbi_list, key=lambda x: x[1])

    tags = backtrack(viterbi_matrix, tagset, max_tag) + [max_tag] + tags
    return tags

def compute_score(tag_seq, input_length, score):
    """
    Computes the total score of a tag sequence
    :param tag_seq: Array of String of length input_length. The tag sequence including <START> and <STOP>
    :param input_length: Int. input length including the padding <START> and <STOP>
    :param score: function from current_tag (string), previous_tag (string), i (int) to the score.  i=0 points to
        <START> and i=1 points to the first token. i=input_length-1 points to <STOP>
    :return:
    """
    total_score = 0
    for i in range(1, input_length):
        total_score += score(tag_seq[i], tag_seq[i - 1], i)
    return total_score


def compute_features(tag_seq, input_length, features):
    """
    Compute f(xi, yi)
    :param tag_seq: [tags] already padded with <START> and <STOP>
    :param input_length: input length including the padding <START> and <STOP>
    :param features: func from token index to FeatureVector
    :return:
    """
    feats = FeatureVector({})
    for i in range(1, input_length):
        feats.times_plus_equal(1, features.compute_features(tag_seq[i], tag_seq[i - 1], i))
    return feats

    # Examples from class (from slides Jan 15, slide 18):
    # x = will to fight
    # y = NN TO VB
    # features(x,y) =
    #  {"wi=will^yi=NN": 1, // "wi="+current_word+"^yi="+current_tag
    # "yi-1=START^yi=NN": 1,
    # "ti=to+^yi=TO": 1,
    # "yi-1=NN+yi=TO": 1,
    # "xi=fight^yi=VB": 1,
    # "yi-1=TO^yi=VB": 1}

    # x = will to fight
    # y = NN TO VBD
    # features(x,y)=
    # {"wi=will^yi=NN": 1,
    # "yi-1=START^yi=NN": 1,
    # "ti=to+^yi=TO": 1,
    # "yi-1=NN+yi=TO": 1,
    # "xi=fight^yi=VBD": 1,
    # "yi-1=TO^yi=VBD": 1}

def sgd(training_size, epochs, gradient, parameters, training_observer):
    """
    Stochastic gradient descent
    :param training_size: int. Number of examples in the training set
    :param epochs: int. Number of epochs to run SGD for
    :param gradient: func from index (int) in range(training_size) to a FeatureVector of the gradient
    :param parameters: FeatureVector.  Initial parameters.  Should be updated while training
    :param training_observer: func that takes epoch and parameters.  You can call this function at the end of each
           epoch to evaluate on a dev set and write out the model parameters for early stopping.
    :return: final parameters
    """
    # Look at the FeatureVector object.  You'll want to use the function times_plus_equal to update the
    # parameters.
    # To implement early stopping you can call the function training_observer at the end of each epoch.
    return

def optimizer(update_func="ssgd", l2_lambda=0.01):

    # only for adagrad
    # a feature vector for accumulated gradient square sum
    accum_sum = FeatureVector({})

    def adagrad(
        i,
        gradient,
        parameters,
        step_size,
    ):
        """
        AdaGrad update
        :param i: index of current instance
        :param gradient: func from index (int) in range(training_size) to a FeatureVector of the gradient
        :param parameters: FeatureVector.  Initial parameters.  Should be updated while training
        :param step_size: int. Learning rate, step size
        :return: updated parameters
        """
        # step_size / sqrt(accum_sum) * grad
        # accum_sum = sum_t(grad_t**2)
        grad = gradient(i)
        accum_sum.times_plus_equal(1, grad.square())
        parameters.times_plus_equal(
            -step_size, grad.divide(accum_sum.square_root())
        )
        return parameters

    def ssgd(i, gradient, parameters, step_size):
        """
        Stochastic sub-gradient descent update
        :param i: index of current instance
        :param gradient: func from index (int) in range(training_size) to a FeatureVector of the gradient
        :param parameters: FeatureVector.  Initial parameters.  Should be updated while training
        :param step_size: int. Learning rate, step size
        :return: updated parameters
        """
        # Look at the FeatureVector object.  You'll want to use the function times_plus_equal to update the parameters.
        # gradient in feature vector class
        grad = gradient(i)
        # w − α g(x, y)
        parameters.times_plus_equal(-step_size, grad)
        return parameters

    def l2_regularizer(i, gradient, parameters, step_size):
        """
        Stochastic sub-gradient descent update with L2 regularizer
        :param i: index of current instance
        :param gradient: func from index (int) in range(training_size) to a FeatureVector of the gradient
        :param parameters: FeatureVector.  Initial parameters.  Should be updated while training
        :param step_size: int. Learning rate, step size
        :return: updated parameters
        """
        # Look at the FeatureVector object.  You'll want to use the function times_plus_equal to update the parameters.
        # gradient in feature vector class
        grad = gradient(i)
        #  learning for l2 regularizer: w − α g(x, y) − αλw
        # λw
        regularizer = FeatureVector({})
        regularizer.times_plus_equal(l2_lambda, parameters)
        # w − α g(x, y) − αλw
        parameters.times_plus_equal(-step_size, grad)
        parameters.times_plus_equal(-step_size, regularizer)
        return parameters

    update = ssgd
    if update_func == "adagrad":
        update = adagrad
    elif update_func == "l2_regularizer":
        update = l2_regularizer

    def optimizer_func(
        training_size,
        epochs,
        gradient,
        parameters,
        training_observer,
        step_size=1,
    ):
        """
        Optimization Function (Based on Gradient Descent)
        :param training_size: int. Number of examples in the training set
        :param epochs: int. Number of epochs to run SGD for
        :param gradient: func from index (int) in range(training_size) to a FeatureVector of the gradient
        :param parameters: FeatureVector.  Initial parameters.  Should be updated while training
        :param training_observer: func that takes epoch and parameters.  You can call this function at the end of each
            epoch to evaluate on a dev set and write out the model parameters for early stopping.
        :param step_size: int. Learning rate, step size
        :return: final parameters
        """

        no_improve_count = 0

        best_params = parameters
        max_score = float("-inf")

        # go through every epochs
        for epoch in range(1, epochs + 1):
            # go through every training data
            for i in tqdm(range(training_size), desc="Training"):
                parameters = update(i, gradient, parameters, step_size)

            # dev score
            cur_score = training_observer(epoch, parameters)
            print(f"F-1 score at epoch {epoch}: {cur_score}")

            # updating best parameters
            if cur_score >= max_score:
                best_params = FeatureVector({})
                best_params.times_plus_equal(1, parameters)
                max_score = cur_score
                no_improve_count = 0
            else:
                # if no improvement
                no_improve_count += 1

            # if larger than tolerable no improvement times
            if no_improve_count > EARLY_STOP_NO_IMPROVE_LIMIT:
                # early stopping
                return best_params

        return best_params

    return optimizer_func


def hamming_loss(loss_val=10, penalty=0):
    """
    Modify the cost function to penalize mistakes three times more (penalty of 30) if the gold standard has a tag
    that is not O but the candidate tag is O.

    Args:
        penalty (Int)
    """

    def loss(gold, pred):
        result = loss_val
        if penalty > 0:
            if gold != "O" and pred == "O":
                result = penalty * result
        return result if gold != pred else 0

    return loss


def svm_with_cost_func(cost_func):
    def score(gold_labels, parameters, features):
        return svm_score(
            gold_labels, parameters, features, cost_func=cost_func
        )

    return score


def perceptron_score(gold_labels, parameters, features):
    # score function given current tag and previous tag with the parameter
    def score(cur_tag, pre_tag, i):
        # w dot f(x, y')
        return parameters.dot_product(
            features.compute_features(cur_tag, pre_tag, i)
        )

    return score


def svm_score(gold_labels, parameters, features, cost_func=hamming_loss()):
    # score function given current tag and previous tag with the parameter
    def score(cur_tag, pre_tag, i):
        # w dot f(x, y')
        cost_val = cost_func(gold_labels[i], cur_tag)
        cur_score = parameters.dot_product(
            features.compute_features(cur_tag, pre_tag, i)
        )
        return cur_score + cost_val

    return score


def get_gradient(data, feature_names, tagset, parameters, score_func):
    data = sample(data, sample_num)

    def subgradient(i):
        """
        Computes the subgradient of the Perceptron loss for example i
        :param i: Int
        :return: FeatureVector
        """
        # data point at i
        inputs = data[i]
        # get the token length
        input_len = len(inputs["tokens"])
        # get the gold labels
        gold_labels = inputs["gold_tags"]
        # get the features given feature names
        features = Features(inputs, feature_names)
        score = score_func(gold_labels, parameters, features)
        # use viterbi algorithm for decoding the tags
        tags = decode(input_len, tagset, score)
        # print(tags, gold_labels)
        # Add the predicted features
        fvector = compute_features(tags, input_len, features)

        # print("Input:", inputs)  # helpful for debugging
        # print("Predicted Feature Vector:", fvector.fdict)
        # print(
        #     "Predicted Score:", parameters.dot_product(fvector)
        # )  # compute_score(tags, input_len, score)

        # Subtract the features for the gold labels
        fvector.times_plus_equal(
            -1, compute_features(gold_labels, input_len, features)
        )
        # print(
        #     "Gold Labels Feature Vector: ",
        #     compute_features(gold_labels, input_len, features).fdict,
        # )
        # print(
        #     "Gold Labels Score:",
        #     parameters.dot_product(
        #         compute_features(gold_labels, input_len, features)
        #     ),
        # )
        # return the difference between features: which will be the update step
        return fvector

    return subgradient

def train(data, feature_names, tagset, epochs):
    """
    Trains the model on the data and returns the parameters
    :param data: Array of dictionaries representing the data.  One dictionary for each data point (as created by the
        make_data_point function).
    :param feature_names: Array of Strings.  The list of feature names.
    :param tagset: Array of Strings.  The list of tags.
    :param epochs: Int. The number of epochs to train
    :return: FeatureVector. The learned parameters.
    """
    parameters = FeatureVector({})   # creates a zero vector

    def perceptron_gradient(i):
        """
        Computes the gradient of the Perceptron loss for example i
        :param i: Int
        :return: FeatureVector
        """
        inputs = data[i]
        input_len = len(inputs['tokens'])
        gold_labels = inputs['gold_tags']
        features = Features(inputs, feature_names)

        def score(cur_tag, pre_tag, i):
            return parameters.dot_product(features.compute_features(cur_tag, pre_tag, i))

        tags = decode(input_len, tagset, score)
        fvector = compute_features(tags, input_len, features)           # Add the predicted features
        #print('Input:', inputs)        # helpful for debugging
        #print("Predicted Feature Vector:", fvector.fdict)
        #print("Predicted Score:", parameters.dot_product(fvector)) # compute_score(tags, input_len, score)
        fvector.times_plus_equal(-1, compute_features(gold_labels, input_len, features))    # Subtract the features for the gold labels
        #print("Gold Labels Feature Vector: ", compute_features(gold_labels, input_len, features).fdict)
        #print("Gold Labels Score:", parameters.dot_product(compute_features(gold_labels, input_len, features)))
        return fvector

    def training_observer(epoch, parameters):
        """
        Evaluates the parameters on the development data, and writes out the parameters to a 'model.iter'+epoch and
        the predictions to 'ner.dev.out'+epoch.
        :param epoch: int.  The epoch
        :param parameters: Feature Vector.  The current parameters
        :return: Double. F1 on the development data
        """
        dev_data = read_data('ner.dev')
        (_, _, f1) = evaluate(dev_data, parameters, feature_names, tagset)
        write_predictions('ner.dev.out'+str(epoch), dev_data, parameters, feature_names, tagset)
        parameters.write_to_file('model.iter'+str(epoch))
        return f1

    return sgd(len(data), epochs, perceptron_gradient, parameters, training_observer)


def predict(inputs, input_len, parameters, feature_names, tagset):
    """

    :param inputs:
    :param input_len:
    :param parameters:
    :param feature_names:
    :param tagset:
    :return:
    """
    features = Features(inputs, feature_names)

    def score(cur_tag, pre_tag, i):
        return parameters.dot_product(features.compute_features(cur_tag, pre_tag, i))

    return decode(input_len, tagset, score)


def make_data_point(sent):
    """
        Creates a dictionary from String to an Array of Strings representing the data.  The dictionary items are:
        dic['tokens'] = Tokens padded with <START> and <STOP>
        dic['pos'] = POS tags padded with <START> and <STOP>
        dic['NP_chunk'] = Tags indicating noun phrase chunks, padded with <START> and <STOP>
        dic['gold_tags'] = The gold tags padded with <START> and <STOP>
    :param sent: String.  The input CoNLL format string
    :return: Dict from String to Array of Strings.
    """
    dic = {}
    sent = [s.strip().split() for s in sent]
    dic['tokens'] = ['<START>'] + [s[0] for s in sent] + ['<STOP>']
    dic['pos'] = ['<START>'] + [s[1] for s in sent] + ['<STOP>']
    dic['NP_chunk'] = ['<START>'] + [s[2] for s in sent] + ['<STOP>']
    dic['gold_tags'] = ['<START>'] + [s[3] for s in sent] + ['<STOP>']
    return dic

def read_data(filename):
    """
    Reads the CoNLL 2003 data into an array of dictionaries (a dictionary for each data point).
    :param filename: String
    :return: Array of dictionaries.  Each dictionary has the format returned by the make_data_point function.
    """
    data = []
    filepath = "data/" + filename
    with open(filepath, 'r') as f:
        sent = []
        for line in f.readlines():
            if line.strip():
                sent.append(line)
            else:
                data.append(make_data_point(sent))
                sent = []
        data.append(make_data_point(sent))

    return data

def read_gazetteer():
    """
    Reads the gazetteer file and returns a list of all the words in the gazetteer
    :return: List of Strings
    """
    data = list()
    with open("data/gazetteer.txt", "r") as f:
        for line in f.readlines():
            data += line.split()[1:]
    return data

def write_predictions(out_filename, all_inputs, parameters, feature_names, tagset):
    """
    Writes the predictions on all_inputs to out_filename, in CoNLL 2003 evaluation format.
    Each line is token, pos, NP_chuck_tag, gold_tag, predicted_tag (separated by spaces)
    Sentences are separated by a newline
    The file can be evaluated using the command: python conlleval.py < out_file
    :param out_filename: filename of the output
    :param all_inputs:
    :param parameters:
    :param feature_names:
    :param tagset:
    :return:
    """
    with open(out_filename, 'w', encoding='utf-8') as f:
        for inputs in all_inputs:
            input_len = len(inputs['tokens'])
            tag_seq = predict(inputs, input_len, parameters, feature_names, tagset)
            for i, tag in enumerate(tag_seq[1:-1]):  # deletes <START> and <STOP>
                f.write(' '.join([inputs['tokens'][i+1], inputs['pos'][i+1], inputs['NP_chunk'][i+1], inputs['gold_tags'][i+1], tag])+'\n') # i + 1 because of <START>
            f.write('\n')

def evaluate(data, parameters, feature_names, tagset):
    """
    Evaluates precision, recall, and F1 of the tagger compared to the gold standard in the data
    :param data: Array of dictionaries representing the data.  One dictionary for each data point (as created by the
        make_data_point function)
    :param parameters: FeatureVector.  The model parameters
    :param feature_names: Array of Strings.  The list of features.
    :param tagset: Array of Strings.  The list of tags.
    :return: Tuple of (prec, rec, f1)
    """
    all_gold_tags = [ ]
    all_predicted_tags = [ ]
    for inputs in data:
        all_gold_tags.extend(inputs['gold_tags'][1:-1])  # deletes <START> and <STOP>
        input_len = len(inputs['tokens'])
        all_predicted_tags.extend(predict(inputs, input_len, parameters, feature_names, tagset)[1:-1]) # deletes <START> and <STOP>
    return conllevaluate(all_gold_tags, all_predicted_tags)

def test_decoder():
    # See https://classes.soe.ucsc.edu/nlp202/Winter21/assignments/A1_Debug_Example.pdf

    tagset = ['NN', 'VB']     # make up our own tagset

    def score_wrap(cur_tag, pre_tag, i):
        retval = score(cur_tag, pre_tag, i)
        print('Score('+cur_tag+','+pre_tag+','+str(i)+') returning '+str(retval))
        return retval

    def score(cur_tag, pre_tag, i):
        if i == 0:
            print("ERROR: Don't call score for i = 0 (that points to <START>, with nothing before it)")
        if i == 1:
            if pre_tag != '<START>':
                print("ERROR: Previous tag should be <START> for i = 1. Previous tag = "+pre_tag)
            if cur_tag == 'NN':
                return 6
            if cur_tag == 'VB':
                return 4
        if i == 2:
            if cur_tag == 'NN' and pre_tag == 'NN':
                return 4
            if cur_tag == 'NN' and pre_tag == 'VB':
                return 9
            if cur_tag == 'VB' and pre_tag == 'NN':
                return 5
            if cur_tag == 'VB' and pre_tag == 'VB':
                return 0
        if i == 3:
            if cur_tag != '<STOP>':
                print('ERROR: Current tag at i = 3 should be <STOP>. Current tag = '+cur_tag)
            if pre_tag == 'NN':
                return 1
            if pre_tag == 'VB':
                return 1

    predicted_tag_seq = decode(4, tagset, score_wrap)
    print('Predicted tag sequence should be = <START> VB NN <STOP>')
    print('Predicted tag sequence = '+' '.join(predicted_tag_seq))
    print("Score of ['<START>','VB','NN','<STOP>'] = "+str(compute_score(['<START>','VB','NN','<STOP>'], 4, score)))
    print('Max score should be = 14')
    print('Max score = '+str(compute_score(predicted_tag_seq, 4, score)))


def main_predict(data_filename, model_filename):
    """
    Main function to make predictions.
    Loads the model file and runs the NER tagger on the data, writing the output in CoNLL 2003 evaluation format to data_filename.out
    :param data_filename: String
    :param model_filename: String
    :return: None
    """
    data = read_data(data_filename)
    parameters = FeatureVector({})
    parameters.read_from_file(model_filename)

    tagset = ['B-PER', 'B-LOC', 'B-ORG', 'B-MISC', 'I-PER', 'I-LOC', 'I-ORG', 'I-MISC', 'O']
    feature_names = ['tag', 'prev_tag', 'current_word']

    write_predictions(data_filename+'.out', data, parameters, feature_names, tagset)
    evaluate(data, parameters, feature_names, tagset)

    return


def main_train():
    """
    Main function to train the model
    :return: None
    """
    print('Reading training data')
    train_data = read_data('ner.train')
    #train_data = read_data('ner.train')[1:1] # if you want to train on just one example

    tagset = ['B-PER', 'B-LOC', 'B-ORG', 'B-MISC', 'I-PER', 'I-LOC', 'I-ORG', 'I-MISC', 'O']
    feature_names = ['tag', 'prev_tag', 'current_word']

    print('Training...')
    parameters = train(train_data, feature_names, tagset, epochs=10)
    print('Training done')
    dev_data = read_data('ner.dev')
    evaluate(dev_data, parameters, feature_names, tagset)
    test_data = read_data('ner.test')
    evaluate(test_data, parameters, feature_names, tagset)
    parameters.write_to_file('model')

    return

def add_features(feats, key):
    feats.times_plus_equal(1, FeatureVector({key: 1}))

def get_char_shape(char):
    encoding = ord(char)
    if encoding >= ord("a") and encoding <= ord("z"):
        return "a"
    if encoding >= ord("A") and encoding <= ord("Z"):
        return "A"
    if encoding >= ord("0") and encoding <= ord("9"):
        return "d"
    return char

def get_word_shape(word):
    shape = ""
    for c in word:
        shape += get_char_shape(c)
    return shape

def is_gazetteer(word):
    gazetteer = read_gazetteer()
    if word in gazetteer:
        return "True"
    return "False"

def is_capital(word):
    if len(word) == 0:
        return "False"
    c = ord(word[0])
    if c >= ord("A") and c <= ord("Z"):
        return "True"
    return "False"

class Features(object):
    def __init__(self, inputs, feature_names):
        """
        Creates a Features object
        :param inputs: Dictionary from String to an Array of Strings.
            Created in the make_data_point function.
            inputs['tokens'] = Tokens padded with <START> and <STOP>
            inputs['pos'] = POS tags padded with <START> and <STOP>
            inputs['NP_chunk'] = Tags indicating noun phrase chunks, padded with <START> and <STOP>
            inputs['gold_tags'] = DON'T USE! The gold tags padded with <START> and <STOP>
        :param feature_names: Array of Strings.  The list of features to compute.
        """
        self.feature_names = feature_names
        self.inputs = inputs

    def compute_features(self, cur_tag, pre_tag, i):
        """
        Computes the local features for the current tag, the previous tag, and position i
        :param cur_tag: String.  The current tag.
        :param pre_tag: String.  The previous tag.
        :param i: Int. The position
        :return: FeatureVector
        """
        feats = FeatureVector({})
        cur_word = self.inputs["tokens"][i]
        pos_tag = self.inputs["pos"][i]
        is_last = len(self.inputs["tokens"]) - 1 == i
        # Feature-1: Current word(Wi)
        if "current_word" in self.feature_names:
            key = f"Wi={cur_word}+Ti={cur_tag}"
            add_features(feats, key)
        # Feature-2: Previous Tag(Ti-1)
        if "prev_tag" in self.feature_names:
            key = f"Ti-1={pre_tag}+Ti={cur_tag}"
            add_features(feats, key)
        # Feature-3: Lowercased Word(Oi)
        if "lowercase" in self.feature_names:
            key = f"Oi={cur_word.lower()}+Ti={cur_tag}"
            add_features(feats, key)
        # Feature-4: Current POS Tag(Pi)
        if "pos_tag" in self.feature_names:
            key = f"Pi={pos_tag}+Ti={cur_tag}"
            add_features(feats, key)
        # Feature-5: Shape of Current Word(Si)
        if "word_shape" in self.feature_names:
            word_shape = get_word_shape(cur_word)
            key = f"Si={word_shape}+Ti={cur_tag}"
            add_features(feats, key)
        # Feature-6: (1-4 for prev + for next)
        if "feats_prev_and_next" in self.feature_names:
            prev_word = self.inputs["tokens"][i - 1]
            prev_pos = self.inputs["pos"][i - 1]
            prev_1 = f"Wi-1={prev_word}+Ti={cur_tag}"
            prev_3 = f"Oi-1={prev_word.lower()}+Ti={cur_tag}"
            prev_4 = f"Pi-1={prev_pos}+Ti={cur_tag}"
            add_features(feats, prev_1)
            add_features(feats, prev_3)
            add_features(feats, prev_4)
            if not is_last:
                next_word = self.inputs["tokens"][i + 1]
                next_pos = self.inputs["pos"][i + 1]
                next_1 = f"Wi+1={next_word}+Ti={cur_tag}"
                next_3 = f"Oi+1={next_word.lower()}+Ti={cur_tag}"
                next_4 = f"Pi+1={next_pos}+Ti={cur_tag}"
                add_features(feats, next_1)
                add_features(feats, next_3)
                add_features(feats, next_4)
        # Feature-7: 1,3,4 conjoined with Previous Tag (pre_tag)
        if "feat_conjoined" in self.feature_names:
            conjoined_1 = f"Wi={cur_word}+Ti-1={pre_tag}+Ti={cur_tag}"
            conjoined_3 = f"Oi={cur_word.lower()}+Ti-1={pre_tag}+Ti={cur_tag}"
            conjoined_4 = f"Pi={pos_tag}+Ti-1={pre_tag}+Ti={cur_tag}"
            add_features(feats, conjoined_1)
            add_features(feats, conjoined_3)
            add_features(feats, conjoined_4)
        # Feature-8: Prefix for Current word with lenhth k where k=1,2,3,4
        if "prefix_k" in self.feature_names:
            for k in range(4):
                if k > len(cur_word):
                    break
                prefix = cur_word[: k + 1]
                key = f"PREi={prefix}+Ti={cur_tag}"
                add_features(feats, key)
        # Feature-9: Gazetteer (GAZi)
        if "gazetteer" in self.feature_names:
            key = f"GAZi={is_gazetteer(cur_word)}+Ti={cur_tag}"
            add_features(feats, key)
        # Feature-10: Is capital (CAPi)
        if "capital" in self.feature_names:
            key = f"CAPi={is_capital(cur_word)}+Ti={cur_tag}"
            add_features(feats, key)
        # Feature-11: Position of the current word (indexed from 1)
        if "position" in self.feature_names:
            key = f"POSi={i+1}+Ti={cur_tag}"
            add_features(feats, key)
        return feats

class FeatureVector(object):

    def __init__(self, fdict):
        self.fdict = fdict

    def times_plus_equal(self, scalar, v2):
        """
        self += scalar * v2
        :param scalar: Double
        :param v2: FeatureVector
        :return: None
        """
        for key, value in v2.fdict.items():
            self.fdict[key] = scalar * value + self.fdict.get(key, 0)


    def dot_product(self, v2):
        """
        Computes the dot product between self and v2.  It is more efficient for v2 to be the smaller vector (fewer
        non-zero entries).
        :param v2: FeatureVector
        :return: Int
        """
        retval = 0
        for key, value in v2.fdict.items():
            retval += value * self.fdict.get(key, 0)
        return retval

    def write_to_file(self, filename):
        """
        Writes the feature vector to a file.
        :param filename: String
        :return: None
        """
        print('Writing to ' + filename)
        with open(filename, 'w', encoding='utf-8') as f:
            for key, value in self.fdict.items():
                f.write('{} {}\n'.format(key, value))


    def read_from_file(self, filename):
        """
        Reads a feature vector from a file.
        :param filename: String
        :return: None
        """
        self.fdict = {}
        with open(filename, 'r') as f:
            for line in f.readlines():
                txt = line.split()
                self.fdict[txt[0]] = float(txt[1])
