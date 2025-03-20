import random
from conlleval import evaluate as conllevaluate
import numpy as np
from tqdm import tqdm

random.seed(1234)
START = '<START>'
STOP = '<STOP>'
EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3

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

def sgd(training_size, epochs, gradient, parameters, training_observer, patience=EARLY_STOPPING_PATIENCE):
    """
    Stochastic gradient descent with early stopping and tqdm progress bar.
    """
    step_size = 1.0  # Learning rate
    best_f1 = 0.0  # Track best F1-score
    best_params = FeatureVector(parameters.fdict.copy())  # Store best parameters
    no_improve_count = 0  # Track epochs without improvement

    for epoch in range(epochs):
        indices = list(range(training_size))
        random.shuffle(indices)  # Shuffle data each epoch

        # Show progress bar for each epoch
        with tqdm(total=len(indices), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for i in indices:
                grad = gradient(i)  # Compute gradient
                parameters.times_plus_equal(-step_size, grad)  # Update parameters
                pbar.update(1)  # Update progress bar

        # Evaluate after each epoch
        f1 = training_observer(epoch, parameters)

        # Early stopping check
        if f1 > best_f1:
            best_f1 = f1
            best_params = FeatureVector(parameters.fdict.copy())  # Save best parameters
            no_improve_count = 0  # Reset count
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs. Best F1: {best_f1:.2f}")
            break

    return best_params  # Return best parameters instead of last trained

def adagrad_train(training_size, epochs, gradient, parameters, training_observer, patience=EARLY_STOPPING_PATIENCE):
    """
    Adagrad training with early stopping and tqdm progress bar.
    """
    step_size = 1.0  # Learning rate (Adagrad uses step size 1)
    best_f1 = 0.0  # Track best F1-score
    best_params = FeatureVector(parameters.fdict.copy())  # Store best parameters
    no_improve_count = 0  # Track epochs without improvement
    squared_gradients = FeatureVector({})  # Store sum of squared gradients for each feature

    for epoch in range(epochs):
        indices = list(range(training_size))
        random.shuffle(indices)  # Shuffle data each epoch

        # Show progress bar for each epoch
        with tqdm(total=len(indices), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for i in indices:
                grad = gradient(i)  # Compute gradient
                for key, value in grad.fdict.items():
                    squared_gradients.fdict[key] = squared_gradients.fdict.get(key, 0) + value ** 2
                    adjusted_step = step_size / (np.sqrt(squared_gradients.fdict[key]) + 1e-8)
                    parameters.fdict[key] = parameters.fdict.get(key, 0) - adjusted_step * value  # Update weights
                pbar.update(1)  # Update progress bar

        # Evaluate after each epoch
        f1 = training_observer(epoch, parameters)

        # Early stopping check
        if f1 > best_f1:
            best_f1 = f1
            best_params = FeatureVector(parameters.fdict.copy())  # Save best parameters
            no_improve_count = 0  # Reset count
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs. Best F1: {best_f1:.2f}")
            return best_params

    return best_params


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

    
    return sgd(len(data), epochs, perceptron_gradient, parameters, training_observer) # Use this for SGD
    #return adagrad_train(len(data), epochs, perceptron_gradient, parameters, training_observer) # use for Adagrad
    #return structured_svm_train(len(data), epochs, perceptron_gradient, parameters, training_observer) # use for structured SVM


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
    feature_names = ['tag', 'prev_tag', 'current_word', 'lowercase', 'pos_tag', 'word_shape', 
                 'feats_prev_and_next', 'feat_conjoined', 'prefix_k', 'gazetteer', 'capital', 'position']

    write_predictions(data_filename+'.out', data, parameters, feature_names, tagset)
    evaluate(data, parameters, feature_names, tagset)

    return

def main_train():
    """
    Main function to train the model
    :return: None
    """
    print('Reading training data')
    #train_data = read_data('ner.train')
    #train_data = read_data('ner.train')[1:1] # if you want to train on just one example
    train_data = read_data('ner.train')[:300] # train on first 1000 examples

    tagset = ['B-PER', 'B-LOC', 'B-ORG', 'B-MISC', 'I-PER', 'I-LOC', 'I-ORG', 'I-MISC', 'O']
    # feature_names = ['current_word', 'prev_tag', 'lowercase', 'pos_tag'] # first 4 features
    feature_names = ['current_word', 'prev_tag', 'lowercase', 'pos_tag', 'word_shape', 
     'feats_prev_and_next', 'feat_conjoined', 'prefix_k', 'gazetteer', 'capital', 'position']

    print('Training...')
    parameters = train(train_data, feature_names, tagset, epochs=EPOCHS)
    print('Training done')
    
    #dev_data = read_data('ner.dev')
    dev_data = read_data('ner.dev')[:50]
    evaluate(dev_data, parameters, feature_names, tagset)
    #test_data = read_data('ner.test')
    test_data = read_data('ner.test')[:50]
    evaluate(test_data, parameters, feature_names, tagset)
    
    parameters.write_to_file('model')

    return

###############################################################################################################
'''
Functions for feature class
'''
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

def structured_svm_train(training_size, epochs, gradient, parameters, training_observer, step_size, reg_strength):
    """
    Structured SVM training with early stopping, cost-augmented decoding, and L2 regularization.
    """
    best_f1 = 0.0
    best_params = FeatureVector(parameters.fdict.copy())
    no_improve_count = 0

    for epoch in range(epochs):
        indices = list(range(training_size))
        random.shuffle(indices)

        with tqdm(total=len(indices), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for i in indices:
                grad = gradient(i)
                # Apply L2 Regularization
                for key in parameters.fdict:
                    parameters.fdict[key] -= step_size * reg_strength * parameters.fdict[key]
                parameters.times_plus_equal(-step_size, grad)
                pbar.update(1)

        # Evaluate model
        f1 = training_observer(epoch, parameters)

        if f1 > best_f1:
            best_f1 = f1
            best_params = FeatureVector(parameters.fdict.copy())
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}. Best F1: {best_f1:.2f}")
            break

    return best_params

def train_structured_svm(data, feature_names, tagset, epochs, step_size=1.0, reg_strength=0.1):
    """
    Structured SVM training with early stopping, cost-augmented decoding, and L2 regularization.
    """
    parameters = FeatureVector({})

    def structured_svm_gradient(i):
        inputs = data[i]
        input_len = len(inputs['tokens'])
        gold_labels = inputs['gold_tags']
        features = Features(inputs, feature_names)

        def score(cur_tag, pre_tag, i):
            return parameters.dot_product(features.compute_features(cur_tag, pre_tag, i))

        # Cost-augmented decoding
        tags = decode_cost_augmented(input_len, tagset, score, gold_labels)
        fvector = compute_features(tags, input_len, features)
        fvector.times_plus_equal(-1, compute_features(gold_labels, input_len, features))
        return fvector

    def training_observer(epoch, parameters):
        dev_data = read_data('ner.dev')[:500]
        (_, _, f1) = evaluate(dev_data, parameters, feature_names, tagset)
        parameters.write_to_file(f'model_structured_svm.iter{epoch}')
        return f1

    return structured_svm_train(len(data), epochs, structured_svm_gradient, parameters, training_observer, step_size, reg_strength)

def decode_cost_augmented(input_length, tagset, score, gold_tags=None):
    """
    Viterbi decoding with cost-augmented decoding for Structured SVM.
    If `gold_tags` is provided, it includes a Hamming loss term in the decoding.
    """
    tags = []
    viterbi_matrix = []

    # Initial step
    initial_list = []
    for tag in tagset:
        tag_score = score(tag, START, 1)
        if gold_tags:
            tag_score += 10 if tag != gold_tags[1] else 0  # Cost augmentation
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
                tag_score = score(tag, prev_tag, t) + last_score
                if gold_tags:
                    tag_score += 10 if tag != gold_tags[t] else 0  # Cost-augmented loss
                if tag_score > max_score:
                    max_score = tag_score
                    max_tag = prev_tag
            viterbi_list.append((max_tag, max_score))
        viterbi_matrix.append(viterbi_list)

    # Termination step
    last_viterbi_list = []
    for tag in tagset:
        stop_score = score(STOP, tag, input_length - 1)
        prev_score = viterbi_matrix[-1][tagset.index(tag)][1]
        final_score = stop_score + prev_score
        last_viterbi_list.append((tag, final_score))
    max_tag, _ = max(last_viterbi_list, key=lambda x: x[1])

    tags = backtrack(viterbi_matrix, tagset, max_tag) + [max_tag] + [STOP]
    return tags
