#!/usr/bin/python

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from gensim.models import KeyedVectors
import numpy as np
from dialog_acts import DialogueActSample
import pandas as pd
import nltk

class DaClassifier():
    def __init__(self):
        #in the terminal load the glove txt file and save it
        #we = KeyedVectors.load_word2vec_format('/home/maitreyee/Development/glove/glove.6B.300d.txt')
        #we.save_word2vec_format('/home/maitreyee/Development/glove/glove.6B.300d.bin', binary=True)
        self.wrdembedding = KeyedVectors.load_word2vec_format('/home/maitreyee/Development/glove/glove.6B.300d.bin',
                                                              binary=True)

        self.classes_int_encoder = LabelEncoder()

        self.classes_encoder = OneHotEncoder()

        self.utt_we, self.acts_de = [], []

        self.classifier = Sequential([Dense(units=200, activation='relu'),  # input_dim=100-n_removed_cmps,
                                      Dropout(.5),
                                      Dense(units=150, activation='relu'),
                                      Dropout(.5),
                                      Dense(units=26, activation='sigmoid')])

    #encodes the classes in the dataset, encoding here:we encode the 20 classes into n-1 or 19 classes.
    # Becuase the classes start at 0. (-1 says that first dimnesion is not
    # known and the second states that the array should be one dimnesion.)

    def classesEncod(self,sampled_data):
        resampled_annotate = sampled_data
        acts = [str(a).lower().strip() for a in resampled_annotate.commfunct.values]
        classes_int = self.classes_int_encoder.fit_transform(acts).reshape(-1, 1)
        classes = self.classes_encoder.fit_transform(classes_int).toarray()
        return classes

    #for graphical presentation of data that extract the principle values from the numerical dataset.

    def average_words(self, tokens, noise=0.):

        tokens = [t.strip(')( ., ?') for t in tokens]

        if noise <= 0.:
            we = [self.wrdembedding[w].reshape(1,-1) for w in tokens if w in self.wrdembedding]
        else:
            we = []
            for t in tokens:
                if np.random.uniform(size=1) <= noise:
                    idx = np.random.choice(self.wrdembedding.vectors.shape[0])
                    we.append(self.wrdembedding.vectors[idx, :])
                else:
                    if t in self.wrdembedding:
                        we.append(self.wrdembedding[t])

        if len(we) > 0:
            mean_we = np.mean(we, axis=0, keepdims=True)
            return mean_we
        else:
            return None
    #generate the train and test samples for x i.e utterances and y i.e classes
    def traintestgenerate(self, df_cleaned):
        classesx = self.classesEncod(df_cleaned)
        utt = df_cleaned.utterance.values
        for i in range(len(utt)):
            u = utt[i]
            a = classesx[i]
            tokens = str(u).strip().lower().split()
            mean_we = self.average_words(tokens, noise=.15)

            if mean_we is None:
                continue

            if mean_we.shape == (1, 1):
                continue
            self.utt_we = self.utt_we + [mean_we]
            self.acts_de = self.acts_de + [a]

        self.acts_de = np.vstack(self.acts_de)
        self.utt_we = np.vstack(self.utt_we)

        size_utt = self.utt_we.shape[1]

        dataset = np.hstack([self.utt_we, self.acts_de])
        np.random.shuffle(dataset)

        self.utt_we, self.acts_de = dataset[:, :size_utt], dataset[:, size_utt:]
        perc_train = .5

        idx_train = int(self.utt_we.shape[0] * perc_train)

        x_train, x_test = self.utt_we[:idx_train, :], self.utt_we[idx_train:, :]

        y_train, y_test = self.acts_de[:idx_train, :], self.acts_de[idx_train:, :]
        yield x_train,x_test,y_train,y_test
    #perform the classification and print the log, some metrics to perform the classification is provided here
    #the classification neural layer is defined in the __init__ function.
    def classifier_n(self, frmtraintest):
        data_toClassify = frmtraintest

        # We compile it
        self.classifier.compile(optimizer=Adam(1e-4),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        for xytrain in data_toClassify:
            x_train, x_test, y_train,y_test = xytrain
        # We fit the network
            h = self.classifier.fit(x=x_train,
                               y=y_train,
                               validation_data=[x_test, y_test],
                               epochs=100,
                               batch_size=10,
                               verbose=2,
                               callbacks=[])
        # its for logging, and showing results

    #analyse the results of the neural network
    def confmatrx(self, xandytest):
        for traintest in xandytest:
            x_train, x_test, y_train, y_test = traintest
            y_true = y_test
            y_pred = self.classifier.predict(x_test)

            idx_max_true = np.argmax(y_true, axis=1)
            idx_max_pred = np.argmax(y_pred, axis=1)

            pd.options.display.max_columns = 500

            conf_mat = confusion_matrix(idx_max_true, idx_max_pred)
            np.core.arrayprint._line_width = 160
            print(conf_mat, )

    def test_on_brkdown(self):
        test_file = []
        utt_test = []
        pd_test = pd.read_csv('./brkdown_corpora1.csv',sep='\t')

        sents = list(pd_test.utterance.str.strip().str.replace(r'[^\w\s]','',regex=True).dropna())

        for i in range(len(sents)):
            u = sents[i]
            tokens = str(u).lower().split()
            we = [self.wrdembedding[w] for w in tokens if w in self.wrdembedding]
            mean_we = np.mean(we, axis=0, keepdims=True).reshape(1, -1)
            if mean_we.shape == (1, 1):
                continue
            utt_test = utt_test + [mean_we]

        utt_test = np.concatenate(utt_test, axis=0)
        utt_test = np.vstack(utt_test)
        res = self.classifier.predict(utt_test)
        pred_class = [list(zip(*[(a, r[a]) for a in range(len(r)) if r[a] > .20])) for r in res]
        res = [list(zip(self.classes_int_encoder.
                        inverse_transform(np.array(p[0], dtype='int')), p[1])) for p in pred_class
               if len(p) > 0]
        p_res = list(zip(sents, res))
        df_das = pd.DataFrame(p_res)
        df_das.to_csv('./da_classified_brkdown_corpora1.csv', sep='\t')
        print(df_das.head())

if __name__ == '__main__':
    sampled_data = DialogueActSample()
    df_cleaned1 = pd.read_csv('./cleaned.csv',sep='\t')
    sampling = sampled_data.samplingFeatures(df_cleaned1)
    classifier = DaClassifier()
    traintest = classifier.traintestgenerate(sampling)
    logging_classifier = classifier.classifier_n(traintest)
    #brkdown = classifier.test_on_brkdown()
