#import the required python libraries in this case pandas and Spacy and instantiate it by the variable nlp
import pandas as pd
import en_core_web_sm
nlp = en_core_web_sm.load()

#a class is defined where the data is parsed, and appended to different columns of a dataframe to be processed for
#cluster and classification
class TopicExtractDep():
    def __init__(self):
        #here we initialize lists for parsed values in an utterance.
        self.data_words1 = []
        self.uttrence_bigrams, self.intj, self.adv, self.subobjverb, self.keywords = [],[],[],[],[]
    # in this function, noun, co-ordinating conjunction,
    # tokens(n to n+2) of direct-object, verb, adverb, tokens(n to n+2) indirect object
    # are extracted and appended to the lists above and then transformed to a dataframe.
    def nouns_extract(self,df_col):
        filepath = '/home/maitreyee/Development/Dm_develop' \
                   '/code/dialogue_act_classifier/' \
                   '/data_brkdown_annotated/brkdown_feats4.4.csv'
        for words in df_col:
            doc = nlp(words)
            tmp_intj = []
            tmp_verb = []
            tmp_adv = []
            try:
               intj = [val.text for val in doc if len(val)<2 or val.pos_=='NOUN' or val.pos_=='CCONJ']
               tmp_intj.append(intj)
               dobj = [[doc[token.text.right_edge.i],doc[token.text.right_edge.i+1]]
                          for token in doc if token.pos_=='dobj']
               adv_adj_noun = [token.text for token in doc if token.pos_=='ADJ' or
                               token.pos_=='ADV' or token.pos_=='NOUN' or token.dep_ =='dobj'
                               or token.dep_=='iobj' or token.pos_=='VERB']
               tmp_adv.append(adv_adj_noun)
               nsubj = [token.text for token in doc if token.dep_=='nsubj']
               tmp_verb.append([nsubj,dobj])
            except:
                pass

            tmp2 = []
            for sent in doc.sents:
                # print(sent)
                tmp = []
                try:
                    for token1 in sent:
                        if 'neg' in token1.dep_:  # or token1.dep_ == 'nsubj':
                            tmp.append([[doc[token1.right_edge.i], doc[token1.right_edge.i + 1]]])  # or token1.dep_ == 'intj' or token1.dep_ == 'advmod'#
                        elif 'aux' in token1.dep_ and 'neg' not in token1.dep_:
                            tmp.append([doc[token1.right_edge.i], doc[token1.right_edge.i + 1]])
                    if len(tmp) == 0:
                        tmp.append([[token2.head.text for token2 in doc]])
                    import numpy as np
                    tmp.append(np.nan)
                    tmp2.append(tmp)
                except:
                    pass

            self.intj.append(tmp_intj)
            self.adv.append(tmp_adv)
            self.subobjverb.append(tmp_verb)
            self.keywords.append(tmp2)
            self.data_words1.append([words for words in doc])

        data_for_plots = pd.DataFrame({
            'utterances': self.data_words1,
            'interjection': self.intj,
            'adverb': self.adv,
            'subobjvrb': self.subobjverb,
            'neg-keywords': self.keywords,
        })

        return(data_for_plots.to_csv(filepath,sep='\t'))
    #for two sequences of utterances find out the similarity of topics: co-reference, cosine, hyponym.


    #def WrdntSimilarity(self, zipped):
        #topics = self.nouns_extract(zipped)
       # noun_list = []
       # ngram_list = []


        #tmpx = []
        #for word in segment:
         #   for wordx in segment2:
          #      tokens1 = wordnet.synsets(word)
           #     tokens2 = wordnet.synsets(wordx)
        #find out tokens that have a similarity
            #    for token1 in tokens1:
             #       for token2 in tokens2:
              #          tmpx.append([token1,token2,token1.wup_similarity(token2)])
        #return(pd.DataFrame(tmpx))

if __name__ == '__main__':
    #fetch csv files from directory one by one, the file path here must be changed with the local path
    filepath = '/home/maitreyee/Development/Dm_develop/' \
               'code/dialogue_act_classifier/brkdown_data/brkdown_corpora4.csv'

    brkdown_corpora1 = pd.read_csv(filepath,
                                   sep='\t')
    #transform the utterance column to a list and of type string to perform the parsing
    utterances = list(brkdown_corpora1['utterance'].astype(str))

    top = TopicExtractDep()
    extract = top.nouns_extract(utterances)


