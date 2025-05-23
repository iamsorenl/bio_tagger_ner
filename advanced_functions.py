import random
from conlleval import evaluate as conllevaluate
import os

directory = 'data'

random.seed(1234)

def decode(input_length, tagset, score):
    """
    Compute the highest scoring sequence according to the scoring function.
    :param input_length: int. number of tokens in the input including <START> and <STOP>
    :param tagset: Array of strings, which are the possible tags.  Does not have <START>, <STOP>
    :param score: function from current_tag (string), previous_tag (string), i (int) to the score.  i=0 points to
        <START> and i=1 points to the first token. i=input_length-1 points to <STOP>
    :return: Array strings of length input_length, which is the highest scoring tag sequence including <START> and <STOP>
    """
    # Look at the function compute_score for an example of how the tag sequence should be scored

    # Initialize DP table and backpointer table
    V = [{} for _ in range(input_length)]  # DP table to store best scores
    backpointer = [{} for _ in range(input_length)]  # To track best previous tags

    # Initialize base case (Start state)
    for tag in tagset:
        V[1][tag] = score(tag, "<START>", 1)  # Score transition from <START> to first tag
        backpointer[1][tag] = "<START>"  # <START> is the previous tag for the first token

    # Recursion - Fill DP table
    for i in range(2, input_length - 1):  # Iterate over positions (excluding <START> and <STOP>)
        for cur_tag in tagset:
            best_prev_tag = None
            best_score = float('-inf')

            # Find the best previous tag
            for prev_tag in tagset:
                if prev_tag in V[i - 1]:  # Ensure previous tag exists in DP table
                    new_score = V[i - 1][prev_tag] + score(cur_tag, prev_tag, i)
                    if new_score > best_score:
                        best_score = new_score
                        best_prev_tag = prev_tag

            # Store best score and backpointer
            if best_prev_tag is not None:  # Avoid storing invalid states
                V[i][cur_tag] = best_score
                backpointer[i][cur_tag] = best_prev_tag

    # Transition to STOP state
    best_final_tag = None
    best_final_score = float('-inf')

    for prev_tag in tagset:
        if prev_tag in V[input_length - 2]:  # Ensure last word has a valid tag
            final_score = V[input_length - 2][prev_tag] + score("<STOP>", prev_tag, input_length - 1)
            if final_score > best_final_score:
                best_final_score = final_score
                best_final_tag = prev_tag

    # Handle edge case where no best final tag was found
    if best_final_tag is None:
        best_final_tag = random.choice(tagset)  # Default to a random tag to prevent errors

    # Backtrace to find the best sequence
    best_sequence = ["<STOP>"]  # Start with STOP tag
    best_sequence.insert(0, best_final_tag)  # Add the best final tag

    for i in range(input_length - 2, 1, -1):  # Reverse loop to reconstruct path
        if best_sequence[0] in backpointer[i]:  # Ensure the backpointer exists
            best_sequence.insert(0, backpointer[i][best_sequence[0]])
        else:
            best_sequence.insert(0, random.choice(tagset))  # Handle missing backpointers safely

    # Add START at the beginning
    best_sequence.insert(0, "<START>")

    return best_sequence

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

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}...")

        # Shuffle the training data order for better learning
        indices = list(range(training_size))
        random.shuffle(indices)

        # Perform updates on each training example
        for i in indices:
            grad = gradient(i)  # Compute gradient for example i
            parameters.times_plus_equal(1.0, grad)  # Update parameters

        # Evaluate the model at the end of the epoch
        f1 = training_observer(epoch, parameters)
        print(f"Epoch {epoch+1} completed. Dev F1 Score: {f1:.4f}")

    return parameters  # Return the final trained parameters

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
    full_path = os.path.join(directory, filename)
    data = []
    with open(full_path, 'r') as f:
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
    
    feature_names = [
    'tag', 'prev_tag', 'current_word', 
    'lowercased_word', 'pos_tag', 'word_shape',
    'prev_word', 'prev_pos', 'next_word', 'next_pos',
    'prefix_1', 'prefix_2', 'prefix_3', 'prefix_4',
    'gazetteer_match', 'capitalized', 'position'
    ]

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

class Features(object):
    def __init__(self, inputs, feature_names):
        """
        Creates a Features object
        :param inputs: Dictionary from String to an Array of Strings.
            inputs['tokens'] = Tokens padded with <START> and <STOP>
            inputs['pos'] = POS tags padded with <START> and <STOP>
            inputs['NP_chunk'] = Tags indicating noun phrase chunks, padded with <START> and <STOP>
            inputs['gold_tags'] = DON'T USE! The gold tags padded with <START> and <STOP>
        :param feature_names: List of feature names to compute.
        """
        self.feature_names = feature_names
        self.inputs = inputs

    def compute_features(self, cur_tag, pre_tag, i):
        """
        Computes the local features for the current tag, the previous tag, and position i
        :param cur_tag: String. The current tag.
        :param pre_tag: String. The previous tag.
        :param i: Int. The position in the sentence.
        :return: FeatureVector
        """
        feats = FeatureVector({})

        word = self.inputs['tokens'][i]
        pos = self.inputs['pos'][i] if 'pos' in self.inputs else 'UNK'
        
        if 'tag' in self.feature_names:
            feats.times_plus_equal(1, FeatureVector({'t=' + cur_tag: 1}))
        if 'prev_tag' in self.feature_names:
            feats.times_plus_equal(1, FeatureVector({'ti=' + cur_tag + "+ti-1=" + pre_tag: 1}))
        if 'current_word' in self.feature_names:
            feats.times_plus_equal(1, FeatureVector({'t=' + cur_tag + '+w=' + word: 1}))

        # New Features 👇
        if 'lowercased_word' in self.feature_names:
            feats.times_plus_equal(1, FeatureVector({'t=' + cur_tag + '+lower_w=' + word.lower(): 1}))
        if 'pos_tag' in self.feature_names:
            feats.times_plus_equal(1, FeatureVector({'t=' + cur_tag + '+pos=' + pos: 1}))
        if 'word_shape' in self.feature_names:
            feats.times_plus_equal(1, FeatureVector({'t=' + cur_tag + '+shape=' + self.word_shape(word): 1}))
        if 'capitalized' in self.feature_names and word[0].isupper():
            feats.times_plus_equal(1, FeatureVector({'t=' + cur_tag + '+capitalized': 1}))

        # Contextual Features (Prev/Next)
        if i > 1 and 'prev_word' in self.feature_names:
            feats.times_plus_equal(1, FeatureVector({'t=' + cur_tag + '+prev_w=' + self.inputs['tokens'][i-1]: 1}))
        if i > 1 and 'prev_pos' in self.feature_names:
            feats.times_plus_equal(1, FeatureVector({'t=' + cur_tag + '+prev_pos=' + self.inputs['pos'][i-1]: 1}))
        if i < len(self.inputs['tokens']) - 1 and 'next_word' in self.feature_names:
            feats.times_plus_equal(1, FeatureVector({'t=' + cur_tag + '+next_w=' + self.inputs['tokens'][i+1]: 1}))
        if i < len(self.inputs['tokens']) - 1 and 'next_pos' in self.feature_names:
            feats.times_plus_equal(1, FeatureVector({'t=' + cur_tag + '+next_pos=' + self.inputs['pos'][i+1]: 1}))

        # Prefix Features
        for n in range(1, 5):
            if 'prefix_' + str(n) in self.feature_names:
                feats.times_plus_equal(1, FeatureVector({'t=' + cur_tag + '+prefix_' + str(n) + '=' + word[:n]: 1}))

        # Position Feature (First, Middle, Last)
        if 'position' in self.feature_names:
            position = 'middle'
            if i == 1:
                position = 'first'
            elif i == len(self.inputs['tokens']) - 2:
                position = 'last'
            feats.times_plus_equal(1, FeatureVector({'t=' + cur_tag + '+position=' + position: 1}))

        return feats

    def word_shape(self, word):
        """
        Converts a word into a shape feature (e.g., 'USA' -> 'AAA', 'Apple' -> 'Aaaaa')
        :param word: The input word.
        :return: A string representing its shape.
        """
        shape = []
        for char in word:
            if char.isupper():
                shape.append('A')
            elif char.islower():
                shape.append('a')
            elif char.isdigit():
                shape.append('0')
            else:
                shape.append(char)
        return ''.join(shape)

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