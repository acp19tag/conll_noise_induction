#!/usr/bin/python2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv1D
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from conlleval import conlleval
from collections import Counter

from tqdm import tqdm # for progress bar

from scripts.utils import *

# packages for learning from crowds
from crowd_layer.crowd_layers import CrowdsClassification, MaskedMultiSequenceCrossEntropy
from crowd_layer.crowd_aggregators import CrowdsCategoricalAggregator

class NoiseInductor():

    def __init__(self, config_params):

        # load config from parameters
        self.config = config_params

        # start tf session
        self.start_tf_session()

        # load word vectors
        self.load_word_vectors()

        # load data
        self.load_data()

        # prepare embedding matrix
        self.prepare_embedding_matrix()

        # one-hot encode data
        self.onehot_encoding()

        # set headers
        self.headers = [
            'noise_prop', 
            'worker_avg_f1', 
            'worker_std_f1', 
            'test_accuracy', 
            'model_precision', 
            'model_recall', 
            'model_f1'
            ]

    def start_tf_session(self):
        """ prevent tensorflow from allocating the entire GPU memory at once """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)

    def load_word_vectors(self):
        """ loads indexing word vectors """
        embeddings_index = {}
        with open("data/glove.6B/glove.6B.%dd.txt" % (self.config["EMBEDDING_DIM"],)) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        self.embeddings_index = embeddings_index

    def load_data(self):
        """ loads data from path in config """
        self.data = {}
        path = self.config["DATA_PATH"]
        all_answers = read_conll(path+"answers.txt")
        all_ground_truth = read_conll(path+"ground_truth.txt")
        all_test = read_conll(path+"testset.txt")
        all_docs = all_ground_truth + all_test

        self.data["X_train"] = [[c[0] for c in x] for x in all_answers]
        self.data["y_answers"] = [[c[1:] for c in y] for y in all_answers]
        self.data["y_ground_truth"] = [[c[1] for c in y] for y in all_ground_truth]
        self.data["X_test"] = [[c[0] for c in x] for x in all_test]
        self.data["y_test"] = [[c[1] for c in y] for y in all_test]
        self.data["X_all"] = [[c[0] for c in x] for x in all_docs]
        self.data["y_all"] = [[c[1] for c in y] for y in all_docs]
        
        self.data["N_ANNOT"] = find_number_of_annotators(self.data["y_answers"])
        self.data["lengths"] = [len(x) for x in all_docs]
        self.data["all_text"] = [c for x in self.data["X_all"] for c in x]
        
        words = list(set(self.data["all_text"]))
        self.data["word2ind"] = {word: index for index, word in enumerate(words)}
        self.data["ind2word"] = {index: word for index, word in enumerate(words)}
        self.data["labels"] = list({c for x in self.data["y_all"] for c in x})
        self.data["labels_without_o"] = list(set(self.data["labels"]) - {'O'})

        self.data["label2ind"] = {label: (index + 1) for index, label in enumerate(self.data["labels"])}
        self.data["ind2label"] = {(index + 1): label for index, label in enumerate(self.data["labels"])}
        self.data["ind2label"][0] = "O" # padding index
        self.data["N_CLASSES"] = len(self.data["label2ind"]) + 1

        self.data["max_label"] = max(self.data["label2ind"].values()) + 1
        self.data["maxlen"] = max(len(x) for x in self.data["X_all"])

        self.data["num_words"] = len(self.data["word2ind"])

    def prepare_embedding_matrix(self):
        """ prepares the embedding matrix """
        self.embedding_matrix = np.zeros((self.data["num_words"], self.config["EMBEDDING_DIM"]))
        for word, i in self.data["word2ind"].items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector

    def onehot_encoding(self):
        """ converts data to one-hot encoding & pads sequences """

        self.data["X_train_enc"] = onehot_encode_x(
            self.data["X_train"], 
            self.data["maxlen"], 
            self.data["word2ind"]
            )

        self.data["y_ground_truth_enc"] = onehot_encode_y(
            self.data["y_ground_truth"], 
            self.data["maxlen"],
            self.data["label2ind"]
            )

        self.data["X_test_enc"] = onehot_encode_x(
            self.data["X_test"], 
            self.data["maxlen"],
            self.data["word2ind"]
            )

        self.data["y_test_enc"] = onehot_encode_y(
            self.data["y_test"], 
            self.data["maxlen"],
            self.data["label2ind"]
            )
        
    def build_base_model(self):
        """ builds the baseline model (see Rodrigues & Pereira paper) """
        base_model = Sequential()
        base_model.add(Embedding(self.data["num_words"],
                            300,
                            weights=[self.embedding_matrix],
                            input_length=self.data["maxlen"],
                            trainable=True))
        base_model.add(Conv1D(512, 5, padding="same", activation="relu"))
        base_model.add(Dropout(0.5))
        base_model.add(GRU(50, return_sequences=True))
        base_model.add(TimeDistributed(Dense(self.data["N_CLASSES"], activation='softmax')))
        base_model.compile(loss='categorical_crossentropy', optimizer='adam')

        return base_model

    def eval_model(self, model):
        """ returns accuracy, precision, recall, f1 """
        pr_test = model.predict(self.data["X_test_enc"], verbose=0)
        pr_test = np.argmax(pr_test, axis=2)

        yh = self.data["y_test_enc"].argmax(2)
        fyh, fpr = score(yh, pr_test)

        preds_test = []
        for i in xrange(len(pr_test)):
            row = pr_test[i][-len(self.data["y_test"][i]):]
            row[np.where(row == 0)] = 1
            preds_test.append(row)
        preds_test = [ list(map(lambda x: self.data["ind2label"][x], y)) for y in preds_test]

        results_test = conlleval(preds_test, self.data["y_test"], self.data["X_test"], 'r_test.txt')

        return accuracy_score(fyh, fpr), results_test['p'], results_test['r'], results_test['f1']

    def run_noise_induction_model(
        self,
        noise_type,
        output_dir
        ):
        """ 
        runs the noise induction model.

        Parameters:

        noise_type: string, one of {"random", "systematic"} 
        output_dir: string, filename of df results output
        
        """

        first_time = True

        noise_prop_list = [np.round(x, 2) for x in np.arange(
            self.config["MIN_NOISE_PROP"],
            self.config["MAX_NOISE_PROP"] + self.config["INC_NOISE_PROP"],
            self.config["INC_NOISE_PROP"]
        )]

        for noise_prop in tqdm(noise_prop_list):

            # add noise
            y_answers_noised = add_noise(
                self.data["y_answers"], 
                noise_prop, 
                self.data["labels_without_o"],
                noise_type=noise_type
                )
            
            answers_dict_noised = generate_answers_dict(y_answers_noised, self.data["y_ground_truth"])
            f1_dict_noised = generate_f1_dict(answers_dict_noised, self.data["labels"], self.data["labels_without_o"])
            mean_f1 = find_mean_f1(f1_dict_noised)
            std_f1 = find_std_f1(f1_dict_noised)

            y_answers_noised_enc = onehot_encode_y_answers(y_answers_noised, self.data["maxlen"], self.data["label2ind"])
            y_mv_noised = get_majority_label(y_answers_noised)
            y_mv_noised_enc = onehot_encode_y(y_mv_noised, self.data["maxlen"], self.data["label2ind"])

            for _ in range(self.config["NUM_REPEATS"]): # run each model N times

                model = self.build_base_model()

                # pre-train base model for using the output of majority voting
                model.fit(
                    self.data["X_train_enc"], 
                    y_mv_noised_enc, 
                    batch_size=self.config["BATCH_SIZE"], 
                    epochs=self.config["PRETRAIN_EPOCHS"], 
                    verbose=0
                    )

                # add crowds layer on top of the base model
                model.add(
                    CrowdsClassification(
                        self.data["N_CLASSES"], 
                        self.data["N_ANNOT"], 
                        conn_type="MW"
                        )
                    )

                # instantiate specialized masked loss to handle missing answers
                loss = MaskedMultiSequenceCrossEntropy(self.data["N_CLASSES"]).loss

                # compile model with masked loss and train
                model.compile(optimizer='adam', loss=loss)
                model.fit(
                    self.data["X_train_enc"], 
                    y_answers_noised_enc, 
                    batch_size=self.config["BATCH_SIZE"], 
                    epochs=self.config["TRAIN_EPOCHS"], 
                    verbose=0
                    ) 

                model.pop()

                model.compile(
                    optimizer='adam', 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy']
                    )

                test_accuracy, model_precision, model_recall, model_f1 = self.eval_model(model)

                df = pd.DataFrame(
                    [
                        [
                            noise_prop, 
                            mean_f1, 
                            std_f1, 
                            test_accuracy, 
                            model_precision, 
                            model_recall, 
                            model_f1
                            ]
                        ], 
                        columns=self.headers
                    )

                if first_time:
                    df.to_csv(output_dir, index=False, header=True)
                    first_time = False
                else:
                    existing_df = pd.read_csv(output_dir)
                    existing_df = existing_df.append(df, ignore_index = True)
                    existing_df.to_csv(output_dir, index=False, header=True)

    def run_denoise_induction_model(
        self,
        output_dir
        ):
        """ 
        runs the DEnoise induction model.

        Parameters:

        output_dir: string, filename of df results output
        
        """

        first_time = True

        denoise_prop_list = [np.round(x, 2) for x in np.arange(
            self.config["MIN_NOISE_PROP"],
            self.config["MAX_NOISE_PROP"] + self.config["INC_NOISE_PROP"],
            self.config["INC_NOISE_PROP"]
        )]

        for denoise_prop in tqdm(denoise_prop_list):

            # add de-noise
            y_answers_denoised = de_noise(
                self.data["y_answers"], 
                denoise_prop, 
                self.data["labels_without_o"], 
                self.data["y_ground_truth"]
                )

            answers_dict_denoised = generate_answers_dict(
                y_answers_denoised, 
                self.data["y_ground_truth"]
                )

            f1_dict_denoised = generate_f1_dict(
                answers_dict_denoised, 
                self.data["labels"], 
                self.data["labels_without_o"]
                )
            mean_f1 = find_mean_f1(f1_dict_denoised)
            std_f1 = find_std_f1(f1_dict_denoised)

            y_answers_denoised_enc = onehot_encode_y_answers(
                y_answers_denoised, 
                self.data["maxlen"], 
                self.data["label2ind"]
                )
            y_mv_denoised = get_majority_label(y_answers_denoised)
            y_mv_denoised_enc = onehot_encode_y(
                y_mv_denoised, 
                self.data["maxlen"], 
                self.data["label2ind"]
                )

            model = self.build_base_model()

            # pre-train base model for a few iterations using the output of majority voting
            model.fit(
                self.data["X_train_enc"], 
                y_mv_denoised_enc, 
                batch_size=self.config["BATCH_SIZE"], 
                epochs=self.config["PRETRAIN_EPOCHS"], 
                verbose=0
                ) 

            # add crowds layer on top of the base model
            model.add(
                CrowdsClassification(
                    self.data["N_CLASSES"], 
                    self.data["N_ANNOT"], 
                    conn_type="MW"
                    )
                )

            # instantiate specialized masked loss to handle missing answers
            loss = MaskedMultiSequenceCrossEntropy(self.data["N_CLASSES"]).loss

            # compile model with masked loss and train
            model.compile(optimizer='adam', loss=loss)
            model.fit(
                self.data["X_train_enc"], 
                y_answers_denoised_enc, 
                batch_size=self.config["BATCH_SIZE"], 
                epochs=self.config["TRAIN_EPOCHS"], 
                verbose=0
                ) 

            model.pop()

            model.compile(
                optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=['accuracy']
                )

            test_accuracy, model_precision, model_recall, model_f1 = self.eval_model(model)

            df = pd.DataFrame(
                [
                    [
                        denoise_prop, 
                        mean_f1, 
                        std_f1, 
                        test_accuracy, 
                        model_precision, 
                        model_recall, 
                        model_f1
                        ]
                    ], 
                    columns=self.headers
                )

            if first_time:
                df.to_csv(output_dir, index=False, header=True)
                first_time = False
            else:
                existing_df = pd.read_csv(output_dir)
                existing_df = existing_df.append(df, ignore_index = True)
                existing_df.to_csv(output_dir, index=False, header=True)