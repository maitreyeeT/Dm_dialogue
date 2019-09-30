class keyphrase_topic_categories():

  def __init__(self):
    self.cntrdct_wrds = ['yet', 'nevertheless', 'nontheless', 'even so',
                         'however',
                         'still', 'despite that', 'inspite of that', 'though',
                         'although']
    self.non_affrmng_wrds = ['no', 'no indeed', 'absolutely not',
                             'most certainly not',
                             'ofcourse not', 'under no circumstance',
                             'by no means',
                             'not at all', 'not really', 'no thanks', 'nae',
                             'nope', 'no way', 'nah']
    self.clarifying_wrds = ['sorry', 'did not get', 'did not understand',
                            'what', 'did not follow', 'come again'
      , 'say that again', 'sorry what', 'execuse me', 'eh?', 'hmm?', 'say what']
    # self.contracted_words = {"didn't":'did not',"couldn't":'could not', "i'm": 'i am', "can't":'cannot'}

  def keyphrase_category(self, df, df_col1, df_col2):
    # for replacing contracted words
    # df = df.assign(Utterance_text_new=df_col1.apply(
    #    lambda x: [key for key, value in self.contracted_words.items() if x in value]))
    # assign category to keyphrases above for the classes mentioned
    df['keyphrase_category'] = df_col2.apply(lambda x: x for x in df_col1)
    print(df.Contradicting_keyphrase)


if __name__ == '__main__':
  import pandas as pd

  filepath = 'dials_for_dialPheno.csv'
  dataHypo = pd.read_csv(filepath, sep="\t")
  classy = keyphrase_topic_categories()
  keyphrase = classy.keyphrase_category(dataHypo, dataHypo['Utterance_text'],
                                        dataHypo['dial_keyphrases'])
