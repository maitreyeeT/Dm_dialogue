# import all the required libraries
# spacy for lemmatization
import en_core_web_sm
import numpy as np
import pandas as pd

nlp = en_core_web_sm.load()
# plotting tool
# import pyLDAvis
# import pyLDAvis.gensim

import nltk
from nltk import word_tokenize

import re

from nltk.metrics import edit_distance
from nltk.corpus import wordnet as wsn

filepath = '/home/maitreyee/Development/Dm_develop/code/plotdata1_test.csv'
data_brkdwn = pd.read_csv(filepath, sep='\t')


def sent_to_wrd(q1, q2):
  return word_tokenize(q1), word_tokenize(q2)


# perform post tagging and stemming

# perform stemming


class Lesk():
  def __init__(self, sentence):
    self.sentence = sentence
    self.meaning = {}
    for word in sentence:
      self.meaning[word] = ''

  def get_senses(self, words):
    return wsn.synsets(words.lower())

  def gloss(self, senses):

    gloss = {}

    for sense in senses:
      gloss[sense.name()] = []

    for sense in senses:
      gloss[sense.name()] += word_tokenize(sense.definition())

    return gloss

  def getAllsenses(self, word):
    senses = self.get_senses(word)

    if senses == []:
      return {word.lower(): senses}

    return self.gloss(senses)

  def score(self, set1, set2):
    # Base
    overlap = 0

    # step
    for word in set1:
      for word in set2:
        overlap += 1

    return overlap

  def overlapScore(self, word1, word2):
    gloss_set1 = self.getAllsenses(word1)

    if self.meaning[word2] == '':
      gloss_set2 = self.getAllsenses(word2)

    else:
      # print
      gloss_set2 = self.gloss([wsn.synset(self.meaning[word2])])

    score = {}

    for i in gloss_set1.keys():
      score[i] = 0
      for j in gloss_set2.keys():
        score[i] += self.score(gloss_set1[i], gloss_set2[j])

    bestSense = None
    max_Score = 0
    for i in gloss_set1.keys():
      if score[i] > max_Score:
        max_Score = score[i]
        bestSense = i

    return bestSense, max_Score

  def lesk(self, word, sentence):
    maxOverlap = 0
    context = sentence
    word_sense = []
    meaning = {}

    senses = self.get_senses(word)

    for sense in senses:
      meaning[sense.name()] = 0

    for word_context in context:
      if not word == word_context:
        score = self.overlapScore(word, word_context)

        if score[0] == None:
          continue
        meaning[score[0]] += score[1]

    if senses == []:
      return word, None, None

    self.meaning[word] = max(meaning.keys(), key=lambda x: meaning[x])

    return word, self.meaning[word], wsn.synset(self.meaning[word]).definition()


def path(set1, set2):
  return wsn.path_similarity(set1, set2)


def wup(set1, set2):
  return wsn.wup_similarity(set1, set2)


def edit(word1, word2):
  if float(edit_distance(word1, word2)) == 0.0:
    return 0.0
  return 1.0 / float(edit_distance(word1, word2))


def compute_path(q1, q2):
  R = np.zeros((len(q1), len(q2)))

  for i in range(len(q1)):
    for j in range(len(q2)):
      if q1[i][1] == None or q2[j][1] == None:
        sim = edit(q1[i][0], q2[j][0])

        R[i, j] = sim

  return R


# compute WUP distance
def computeWUP(q1, q2):
  R = np.zeros((len(q1), len(q2)))
  for i in range(len(q1)):
    for j in range(len(q2)):
      if q1[i][1] == None or q2[j][1] == None:
        sim = edit(q1[i][0], q2[j][0])

      else:
        sim = wup(wsn.synset(q1[i][1]), wsn.synset(q2[j][1]))

      if sim == None:
        sim = edit(q1[i][0], q2[j][0])

        R[i, j] = sim

  return R


def overallSim(q1, q2, R):
  sum_X = 0.0
  sum_Y = 0.0
  for i in range(len(q1)):
    max_i = 0.0
    for j in range(len(q2)):
      if R[i, j] > max_i:
        sum_X += max_i

  for i in range(len(q1)):
    max_j = 0.0
    for j in range(len(q2)):
      if R[i, j] > max_j:
        max_j = R[i, j]
      sum_Y += max_j

  if (float(len(q1)) + float(len(q2))) == 0.0:
    return 0.0

  overall = (sum_X + sum_Y) / (2 * (float(len(q1)) + float(len(q2))))

  return overall


def semanticSimilarity(q1, q2):
  tokens_q1, tokens_q2 = sent_to_wrd(q1, q2)
  sentences = []
  sentencex = []
  for words in tokens_q1:
    tag_q1 = nlp(words)
    for token in tag_q1:
      if 'NOUN' in token.dep_ or 'ADJ' in token.pos_:
        sentences.append(token.text)

  for wordx in tokens_q2:
    tag_q2 = nlp(wordx)
    for token in tag_q2:
      if 'NOUN' in token.dep_ or 'ADJ' in token.pos_:
        sentencex.append(token.text)

  sense1 = Lesk(sentences)
  sentence1means = []
  for word in sentences:
    sentence1means.append(sense1.lesk(word, sentences))

  sense2 = Lesk(sentencex)
  sentence2means = []
  for word in sentencex:
    sentence2means.append(sense2.lesk(word, sentencex))

    R1 = compute_path(sentence1means, sentence2means)
    R2 = computeWUP(sentence1means, sentence2means)

    R = (R1 + R2) / 2

    return overallSim(sentence1means, sentence2means, R)


STOP_WORDS = nltk.corpus.stopwords.words()


def clean_sentence(val):
  regex = re.compile('([^\s\w]|_)+')
  sentence = regex.sub('', val).lower()
  sentence = sentence.split(" ")

  for word in list(sentence):
    if word in STOP_WORDS:
      sentence.remove(word)

  sentence = " ".join(sentence)
  return sentence


# from sklearn.metrics import log_loss

df_sim = pd.read_csv(
  '/home/maitreyee/Development/python_notebook/brkdwn_similarity.csv', sep=',')
X_train = df_sim
X_train = X_train.dropna(how='any')

# y = X_train['is_duplicate']
print('Exported Cleaned train Data, no need for cleaning')
for col in ['utterance1', 'utterance2']:
  X_train[col] = X_train[col].apply(clean_sentence)
y_pred = []
count = 0
print('calculating similarity for the training data, please wait.')
for row in X_train.itertuples():
  q1 = str(row[2])
  q2 = str(row[3])

  sim = semanticSimilarity(q1, q2)
  count += 1
  if count % 1000 == 0:
    print(str(count) + ", " + str(sim) + ", " + str(row[3]))
  y_pred.append(sim)
output = pd.DataFrame(list(zip(X_train['utterance1'].tolist(), y_pred)),
                      columns=['Utterance1', 'similarity'])
output.to_csv('semantic_sim.csv', index=False, sep='\t')
