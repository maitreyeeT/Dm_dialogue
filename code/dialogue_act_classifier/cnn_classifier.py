import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split


class CNN():
  def __init__(self):
    self.tokenizeX = Tokenizer(num_words=10000)
    self.tokenizeY = Tokenizer()
    self.vocab_size = len(self.tokenizeX.word_index) + 1
    self.model = Sequential()

  # import a csv file and append the columns to be classified into class, data
  def read_data(self):
    filepath = ''  # give the filepath
    classify_data = pd.read_csv(filepath,
                                sep='\t')  # change the seperator dependening on your file
    # preprocess the data before converting them to below list of values if required.
    classes = [str(a).lower().strip() for a in
               classify_data['classes_col'].values]
    data = [str(a).lower().strip() for a in classify_data['data_col'].values]
    yield classes, data

  def transform_data(self):
    class_val, data_val = self.read_data()
    utt_train, utt_test, acts_train, acts_test = train_test_split(data_val,
                                                                  class_val,
                                                                  test_size=0.4,
                                                                  random_state=1000)  # modify the parameters here

    tokenizer = self.tokenizeX
    tokenizer.fit_on_texts(data_val)
    x_trainCon = tokenizer.texts_to_sequences(utt_train)
    x_testCon = tokenizer.texts_to_sequences(utt_test)
    self.maxlen = max([len(utt) for utt in x_trainCon + x_testCon])
    self.X_train = pad_sequences(x_trainCon, maxlen=self.maxlen)
    self.X_test = pad_sequences(x_testCon, maxlen=self.maxlen)
    y_tokenizer = self.tokenizeY
    y_tokenizer.fit_on_texts(acts_train)
    y_tokenizer.fit_on_texts(acts_test)
    self.y_train = np.zeros((len(acts_train), len(y_tokenizer.word_index)),
                            dtype='float')
    self.y_test = np.zeros((len(acts_test), len(y_tokenizer.word_index)),
                           dtype='float')
    for i, w in enumerate(acts_train):
      self.y_train[i, y_tokenizer.word_index[w] - 1] = 1
    for i, w in enumerate(acts_test):
      self.y_test[i, y_tokenizer.word_index[w] - 1] = 1

  def create_embeding(self):
    word_emb = KeyedVectors.load_word2vec_format(
        "/home/maitreyee/Development/glove/glove.6B.300d.bin", binary=True)

    embed_dim = 300
    # gensim_matrix = word_emb.vectors

    embed_matrix = np.zeros((self.vocab_size, embed_dim), dtype='float')

    for _, wi in enumerate(self.tokenizeX.word_index.items()):
      w, i = wi
      embed_matrix[i, :] = word_emb[w] if w in word_emb else np.random.normal(0,
                                                                              1,
                                                                              (
                                                                              1,
                                                                              embed_dim))
    yield embed_matrix, embed_dim

  def model(self):
    for matrices in self.create_embeding():
      embed_matrix, embed_dim = matrices

      self.model.add(
        layers.Embedding(self.vocab_size, embed_dim, weights=[embed_matrix],
                         input_length=self.maxlen))
      self.model.add(layers.Conv1D(100, 2, activation='relu'))
      self.model.add(layers.GlobalMaxPool1D())
      self.model.add(layers.Dense(100, activation='relu'))
      self.model.add(layers.Dropout(.5))
      self.model.add(layers.Dense(5, activation='softmax'))

      self.model.compile(optimizer=Adam(1e-3),
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])

      cnn = self.model.fit(x=self.X_train,
                           y=self.y_train,
                           validation_data=[self.X_test, self.y_test],
                           epochs=100,
                           batch_size=128,
                           verbose=2,
                           callbacks=[])


if __name__ == '__main__':
  pass
