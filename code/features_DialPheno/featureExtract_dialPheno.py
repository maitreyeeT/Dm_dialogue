#import the required python libraries in this case pandas and Spacy and instantiate it by the variable nlp
import pandas as pd
import en_core_web_sm
nlp = en_core_web_sm.load()

#a class is defined where the data is parsed, and appended to different columns of a dataframe to be processed for
#cluster and classification
class TopicExtractDep():
    def __init__(self,id): #,id
        #here we initialize lists for parsed values in an utterance.
        self.data_words1 = []
        self.id = id
        self.uttrence_bigrams, self.intj, self.adv, self.subobjverb, self.keywords = [],[],[],[],[]
    # in this function, noun, co-ordinating conjunction,
    # tokens(n to n+2) of direct-object, verb, adverb, tokens(n to n+2) indirect object
    # are extracted and appended to the lists above and then transformed to a dataframe.
    def nouns_extract(self,df_col):
        filepath = '/home/maitreyee/Development/Dm_develop/code/features_DialPheno/data_brkdown_annotated/' \
                   'brkdown_feats4.7.csv'
        for words in df_col:
            doc = nlp(words)
            tmp_intj = []
            tmp_verb = []
            tmp_adv = []
            try:
               intj = [val.text for val in doc if len(val)<2 or val.pos_=='NOUN' or val.pos_=='CCONJ' or val.pos_=='INTJ']
               tmp_intj.append(intj)
               dobj = [token.text for token in doc if token.dep_ =='dobj' or token.dep_=='iobj']
               adv_adj_noun = [token.text for token in doc if #token.pos_=='ADJ' or
                               #token.pos_=='ADV' or
                              token.pos_=='NOUN' or token.dep_ =='dobj'
                              or token.dep_=='iobj']
               tmp_adv.append(adv_adj_noun)
               sovs = [(token.text, token.head.text) for token in doc if
                       token.dep_ == 'nsubj' and token.head.pos_ == 'VERB']
               tmp_verb.append([sovs,dobj])
            except:
                pass

            tmp2 = []
            for sent in doc.sents:
                # print(sent)
                tmp = []
                try:
                    for token1 in sent:
                        if 'neg' in token1.dep_:  # or token1.dep_ == 'nsubj':
                            tmp.append([doc[token1.right_edge.i], doc[token1.right_edge.i + 1]])  # or token1.dep_ == 'intj' or token1.dep_ == 'advmod'#
                       # if 'aux' in token1.dep_ and 'neg' not in token1.dep_:
                        #    tmp.append([doc[token1.right_edge.i], doc[token1.right_edge.i + 1], doc[token1.right_edge.i + 2]])
                    tmp2.append([[token2.text for token2 in sent if token2.dep_ =='aux'],tmp])
                except:
                    pass

            self.intj.append(tmp_intj)
            self.adv.append(tmp_adv)
            self.subobjverb.append(tmp_verb)
            self.keywords.append(tmp2)
            self.data_words1.append([words for words in doc])

        data_for_plots = pd.DataFrame({
            'id':self.id,
            'utterances': self.data_words1,
            'interjection': self.intj,
            'adverb': self.adv,
            'subobjvrb': self.subobjverb,
            'neg-keywords': self.keywords,
        })

        return(data_for_plots.to_csv(filepath,sep='\t'))


if __name__ == '__main__':
    #fetch csv files from directory one by one, the file path here must be changed with the local path
    filepath = '/home/maitreyee/Development/Dm_develop/code/features_DialPheno/' \
               'brkdown_data/brkdown_corpora4.csv'

    brkdown_corpora1 = pd.read_csv(filepath,
                                   sep='\t')
    #transform the utterance column to a list and of type string to perform the parsing
    utterances = list(brkdown_corpora1['utterance'].astype(str).
                      str.replace(r'can t|ca not|cant','cannot',regex=True).str.replace(r'don t','dont',regex=True)
                      .values)
    id = list(brkdown_corpora1['id'])

    top = TopicExtractDep(id)
    extract = top.nouns_extract(utterances)


