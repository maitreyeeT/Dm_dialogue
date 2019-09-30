import hdbscan
import numpy as np
import pandas as pd
from featureExtract_dialPheno import TopicExtractDep
from gensim.models import KeyedVectors
from nltk import ngrams
from scipy.cluster.hierarchy import ward
# import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
# from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics.pairwise import cosine_similarity


class BrkdownClassifier():

  def __init__(self):
    self.wrdembedding = KeyedVectors.load_word2vec_format(
      '/home/maitreyee/Development/glove/glove.6B.300d.bin',
      binary=True)

    self.brkdown_corpora1 = pd.read_csv('/home/maitreyee/Development'
                                        '/Dm_develop/code/dialogue_act_classifier/brkdown_corpora1.csv',
                                        sep='\t')
    utterances = list(ngrams(self.brkdown_corpora1['utterance'].astype(str), 2))

    top_extr = TopicExtractDep()
    self.extract_feats = top_extr.nouns_extract(utterances)
    self.datadf = pd.read_csv('./plotdata.csv', sep='\t')
    self.uttwe = []

  def vectorize_data(self):
    tfidf_vect = TfidfVectorizer(max_df=0.8, max_features=1000,
                                 min_df=0.01, use_idf=True,
                                 ngram_range=(1, 2))
    self.datadf['nouns_kewords'] = self.datadf['noun'] + ' ' + self.datadf[
      'keywords']
    feats3 = list(self.datadf.nouns_kewords.astype(str))
    tfidf_matrix1 = tfidf_vect.fit_transform(feats3)

    tfidf_array = tfidf_matrix1.toarray()
    cosine_dist = cosine_similarity(tfidf_matrix1)
    self.linkage_mat_v1v2 = ward(cosine_dist)
    return zip(self.linkage_mat_v1v2, tfidf_array)

  def word2vec(self):
    self.datadf.combined = self.datadf.noun + ' ' + self.datadf.keywords + ' ' + self.datadf.subobjvrb
    utt1 = self.datadf.combined
    for i in range(len(utt1)):
      u = utt1[i]
      tokens = str(u).strip().lower().split()
      we = [self.wrdembedding[w] for w in tokens if w in self.wrdembedding]
      mean_we = np.mean(we, axis=0, keepdims=True).reshape(1, -1)
      if mean_we.shape == (1, 1):
        continue
      self.uttwe = self.uttwe + [mean_we]
      X = np.array(self.uttwe)
      return X

  def hierarchical_clustering(self):
    tfidf_data = self.vectorize_data()
    wrdemd_data = self.word2vec()
    for matrixplotarray in tfidf_data:
      matrix, tfidf_array = matrixplotarray
      clusterAglo_tfidf = AgglomerativeClustering(n_clusters=5,
                                                  affinity='euclidean',
                                                  linkage='complete')
      clusterAglo_tfidf.fit_predict(matrix)
      clusterAglo_wrdembd = AgglomerativeClustering(n_clusters=5,
                                                    affinity='euclidean',
                                                    linkage='complete')
      clusterAglo_wrdembd.fit_predict(wrdemd_data)
      return zip(clusterAglo_tfidf, clusterAglo_wrdembd)
      # plt.scatter(tfidf_array[:, 0], tfidf_array[:, 1],
      #       c=clusterAglo.labels_, cmap='rainbow', s=15, alpha=1)

  def density_clutering(self):
    X = self.word2vec()
    X = np.array(X)
    wrd_Embd = X.reshape(-1, 1)
    tfidf_data = self.vectorize_data()
    clusterHdbscn_tfidf = hdbscan.HDBSCAN(min_cluster_size=2)
    clusterHdbscn_wrdembd = hdbscan.HDBSCAN(min_cluster_size=2)
    clusterHdbscn_wrdembd.fit_predict(wrd_Embd)
    for xy in tfidf_data:
      matrix, array = xy
      clusterHdbscn_tfidf = clusterHdbscn_tfidf.fit_predict(matrix)

      yield clusterHdbscn_tfidf

    yield clusterHdbscn_wrdembd

  def evaluation(self):
    Agglocluster_result = self.hierarchical_clustering()
    vector_data = self.vectorize_data()
    for matrixplotarray in vector_data:
      matrix, tfidf_array = matrixplotarray
      for tfidfwrdembd in Agglocluster_result:
        cluster_tfidf, cluster_wrdembd = tfidfwrdembd
        labels_tfidf = cluster_tfidf.labels_
        labels_wrdembd = cluster_wrdembd.labels_
        classes = list(self.datadf.nouns_kewords)
        print("Silhouette Score is {}, "
              "and calinski score is {}".format(
            # metrics.homogeneity_score(classes[0:], labels),
            #   metrics.completeness_score(classes, labels),
            #   metrics.v_measure_score(classes,labels),
            metrics.calinski_harabasz_score(matrix, labels_tfidf),
            metrics.calinski_harabasz_score(matrix, labels_wrdembd)))


if __name__ == '__main__':
  classifier = BrkdownClassifier()
  cluster = classifier.evaluation()
