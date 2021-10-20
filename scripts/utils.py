#!/usr/bin/python
import numpy as np
import tensorflow as tf
# import matplotlib.cm as cm
import random
# from matplotlib import pyplot as plt 
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv1D
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from conlleval import conlleval
from collections import Counter

# packages for learning from crowds
from crowd_layer.crowd_layers import CrowdsClassification, MaskedMultiSequenceCrossEntropy
from crowd_layer.crowd_aggregators import CrowdsCategoricalAggregator

def read_conll(filename):
    raw = open(filename, 'r').readlines()
    all_x = []
    point = []
    for line in raw:
        stripped_line = line.strip().split(' ')
        point.append(stripped_line)
        if line == '\n':
            if len(point[:-1]) > 0:
                all_x.append(point[:-1])
            point = []
    # all_x = all_x # TG Note: why is this here??
    return all_x

def find_number_of_annotators(y):
    return len(y[0][0])

def concatenate_answers(annotator_id, y_answers, y_ground_truth):
    """ returns two lists with no missing values; y_true, y_pred """
    y_true = []
    y_pred = []
    for index, work_item in enumerate(y_answers):
        worker_label_list = [x[annotator_id] for x in work_item]
        if not worker_label_list.count('?'):
            y_pred.extend(worker_label_list)
            y_true.extend(y_ground_truth[index])
    return y_true, y_pred

def generate_answers_dict(y_answers, y_ground_truth):
    answers_dict = {}
    for annotator_id in range(find_number_of_annotators(y_answers)):
        y_true, y_pred = concatenate_answers(annotator_id, y_answers, y_ground_truth)
        answers_dict[annotator_id] = {
            "y_true": y_true,
            "y_pred": y_pred
        }
    return answers_dict

def generate_f1_dict(answers_dict, labels, labels_without_o):
    return {
    annotator_id: {
        'f1_with_o': f1_score(
            y_true=answers_dict[annotator_id]["y_true"],
            y_pred=answers_dict[annotator_id]["y_pred"],
            labels=labels,
            average='micro',
        ),
        'f1_without_o': f1_score(
            y_true=answers_dict[annotator_id]["y_true"],
            y_pred=answers_dict[annotator_id]["y_pred"],
            labels=labels_without_o,
            average='micro',
        ),
        'n_without_o': len(answers_dict[annotator_id]["y_true"])
        - answers_dict[annotator_id]["y_true"].count('O'),
    }
    for annotator_id in answers_dict
}

def find_mean_f1(f1_dict):
    return np.mean([f1_dict[x]['f1_without_o'] for x in f1_dict])

def find_std_f1(f1_dict):
    return np.std([f1_dict[x]['f1_without_o'] for x in f1_dict])

def get_majority_label(y_answers):
    y_mv = []
    for sentence in y_answers:
        sentence_label_list = []
        for entity in sentence:
            top_2_labels = Counter(entity).most_common(2)
            if top_2_labels[0][0] != '?':
                sentence_label_list.append(top_2_labels[0][0])
            else:
                sentence_label_list.append(top_2_labels[1][0])
        y_mv.append(sentence_label_list)
    return y_mv

def generate_noise_dict(labels_without_o):
    shift = np.random.randint(len(labels_without_o))
    return {
        labels_without_o[index]: labels_without_o[
            (index + shift) % len(labels_without_o)
        ]
        for index in range(len(labels_without_o))
    }

def random_bool(noise_proportion):
    return np.random.choice([True, False], 1, p=[noise_proportion, 1-noise_proportion])[0]

def add_noise(y_answers, noise_proportion, labels_without_o, noise_type='random'):
    
    noise_dict = generate_noise_dict(labels_without_o)
    y_answers_noised = []
    for sentence in y_answers:
        noised_sentence = []
        for entity in sentence:
            noised_entity = []
            for label in entity:
                noised_label = label
                if label in noise_dict and random_bool(noise_proportion):
                    if noise_type == 'systematic':
                        noised_label = noise_dict[label]
                    elif noise_type == 'random':
                        noised_label = random.choice(labels_without_o)
                noised_entity.append(noised_label)
            noised_sentence.append(noised_entity)
        y_answers_noised.append(noised_sentence)
    return y_answers_noised

def de_noise(y_answers, denoise_proportion, labels_without_o, y_ground_truth):
   
    y_answers_denoised = []
    for sentence_index, sentence in enumerate(y_answers):
        denoised_sentence = []
        for entity_index, entity in enumerate(sentence): 
            denoised_entity = []
            for label in entity:
                denoised_label = label
                if label != '?' and random_bool(denoise_proportion): # consider "label in labels_without_o"
                    denoised_label = y_ground_truth[sentence_index][entity_index]
                denoised_entity.append(denoised_label)
            denoised_sentence.append(denoised_entity)
        y_answers_denoised.append(denoised_sentence)
    return y_answers_denoised

def encode(x, n):
    result = np.zeros(n)
    result[x] = 1
    return result

def onehot_encode_x(x, maxlen, word2ind):
    """ includes padding """
    x_enc = [[word2ind[c] for c in x_i] for x_i in x]
    return pad_sequences(x_enc, maxlen=maxlen)

def onehot_encode_y(y, maxlen, label2ind):
    """ includes padding """
    max_label = max(label2ind.values()) + 1
    y_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y]
    y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]
    return pad_sequences(y_enc, maxlen=maxlen)

def onehot_encode_y_answers(y_answers, maxlen, label2ind):
    """ includes padding """
    y_answers_enc = []
    n_annot = find_number_of_annotators(y_answers)
    for r in range(n_annot):
        annot_answers = []
        for i in range(len(y_answers)):
            seq = []
            for j in range(len(y_answers[i])):
                enc = -1
                if y_answers[i][j][r] != "?":
                    enc = label2ind[y_answers[i][j][r]]
                seq.append(enc)
            annot_answers.append(seq)
        y_answers_enc.append(annot_answers)
    y_answers_enc_padded = []
    for r in range(n_annot):
        padded_answers = pad_sequences(y_answers_enc[r], maxlen=maxlen)
        y_answers_enc_padded.append(padded_answers)
    y_answers_enc_padded = np.array(y_answers_enc_padded)
    return np.transpose(np.array(y_answers_enc_padded), [1, 2, 0])

def score(yh, pr):
    coords = [np.where(yhh > 0)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    return fyh, fpr

class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(
            s['Word'].values.tolist(),
            s['POS'].values.tolist(),
            s['Tag'].values.tolist())]
        self.grouped = self.data.groupby('Sentence #').apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2]
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1, 
            '-1:postag[:2]': postag1[:2]
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1, 
            '+1:postag[:2]': postag1[:2]
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels
