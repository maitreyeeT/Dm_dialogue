import numpy as np
from gensim.models import KeyedVectors
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from dataReductionAndSampling import DialogueActSample
import pandas as pd



class DataTransform():
    def __init__(self):
    # in the terminal load the glove txt file and save it
    # we = KeyedVectors.load_word2vec_format('/home/maitreyee/Development/glove/glove.6B.300d.txt')
    # we.save_word2vec_format('/home/maitreyee/Development/glove/glove.6B.300d.bin', binary=True)
        self.wrdembedding = KeyedVectors.load_word2vec_format('/home/maitreyee/Development/glove/glove.6B.300d.bin',
                                                          binary=True)

        self.classes_int_encoder = LabelEncoder()

        self.classes_encoder = OneHotEncoder()

        self.utt_we, self.acts_de = [], []



# encodes the classes in the dataset, encoding here:we encode the 20 classes into n-1 or 19 classes.
# Becuase the classes start at 0. (-1 says that first dimnesion is not
# known and the second states that the array should be one dimnesion.)

    def classesEncod(self ,sampled_data):
        resampled_annotate = sampled_data
        acts = [str(a).lower().strip() for a in resampled_annotate.feats.values]
        classes_int = self.classes_int_encoder.fit_transform(acts).reshape(-1, 1)
        classes = self.classes_encoder.fit_transform(classes_int).toarray()
        return classes


# for graphical presentation of data that extract the principle values from the numerical dataset.

    def average_words(self, tokens, noise=0.):

        tokens = [t.strip(')( ., ?') for t in tokens]

        if noise <= 0.:
            we = [self.wrdembedding[w].reshape(1 ,-1) for w in tokens if w in self.wrdembedding]
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


# generate the train and test samples for x i.e utterances and y i.e classes
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
        yield x_train ,x_test ,y_train ,y_test
# perform the classification and print the log, some metrics to perform the classification is provided here
# the classification neural layer is defined in the __init__ function.

if __name__ == '__main__':
    data_sampling = DialogueActSample()
    #df_cleaned1 = pd.read_csv('./cleaned.csv', sep='\t')
    #sampling = data_sampling.samplingFeatures(df_cleaned1)

