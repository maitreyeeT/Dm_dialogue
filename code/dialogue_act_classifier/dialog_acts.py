#!/usr/bin/python


import pandas as pd
import re

class DialogueActSample():

    def __init__(self):
        self.filepath = '/home/maitreyee/Development/Dm_develop/data/cleaned_corpora_for_DA_annotation2.csv'
        self.lookups = [('(^.*answer.*$)', 'answer'),('(^.*answe.*$)', 'answer'),('(^.*question.*$)', 'question'),
                      ('(^.*greeting.*$)', 'greeting'),('(^.*thanking.*$)', 'thanking'),
                        ('(^.*turn.*$)', 'turnmanagement'), ('(^.*suggest.*$)', 'suggest'),
                        ('(^.*goodbye.*$)', 'greeting'), ('(^.*agreement.*$)', 'agreement'),('(^.*agree.*$)', 'agreement'),
                        ('(^.*disagreement.*$)', 'disagreement'), ('(^.*instruct.*$)', 'instruct'),
                        ('(^.*inform.*$)', 'inform'),('(^.*apology.*$)', 'apology'),('(^.*apologize.*$)', 'apology')
                      ,('(^.*correct.*$)', 'correction'),('(^.*selfcorrection.*$)', 'correction'),
                      ('(^.*retraction.*$)', 'correction'),('(^.*decline.*$)', 'reject'),('(^.*accept.*$)', 'accept'),
                      ('(^.*feedback.*$)', 'feedback'),('(^.*allo.*$)', 'feedback'),('(^.*auto.*$)', 'feedback'),
                      ('(^.*address.*$)', 'address'),('(^.*opening.*$)', 'interactionstructuring'),
                      ('(^.*introduction.*$)', 'interactionstructuring'),('(^.*selferror)','error'),
                      ('(^.*pausing.*$)', 'request'),('(^.*nan.*$)', 'question'),('(^.*completion.*$)', 'correction'),
                      ('(^.*goodbye.*$)', 'greeting'),('(^.*confirm.*$)', 'confirm'),('(^.*disconfirm.*$)', 'disconfirm')]





    def readAndconvert_data(self):
        dataset = pd.read_csv(self.filepath, sep=',')
        dataset['utterance'] = dataset['utterancetext'].astype(str).str.lower()
        dataset['commfunct'] = dataset['communicativefunction'].astype(str).str.lower().str.strip()
        return zip(dataset.utterance,dataset.commfunct)


    def modify_data(self, readData):
        reading = readData
        for uttcomfun in reading:
            utter, commfunct = uttcomfun
            found = False
            for lookup_match, lookup_trg in self.lookups:
                #print('{} {}'.format(lookup_match, commfunct))
                if re.match(lookup_match, commfunct):
                    yield (utter, lookup_trg)
                    found = True
                    break
            if not found:
                yield (utter, commfunct)


    def samplingFeatures(self, df):
        sample_size = 400
        sampled_clz = []
        reading_df = df
        for clz in reading_df.commfunct.unique():
            print(clz)
            df_class = reading_df[reading_df['commfunct'] == clz]
            if len(df_class) <= sample_size:
                df_class_under = df_class.sample(sample_size, replace=True)

            else:
                df_class_under = df_class.sample(sample_size)
            sampled_clz.append(df_class_under)

        resampled_annotate = pd.concat(sampled_clz)
        return(resampled_annotate)


if __name__ == '__main__':
    classifier = DialogueActSample()

    read = classifier.readAndconvert_data()
    modification = pd.DataFrame(classifier.modify_data(read))
    modification.columns = [['utterance','commfunct']]
    modification.to_csv('./cleaned.csv', sep='\t')
    cleaned = pd.read_csv('./cleaned.csv', sep='\t')
    sampled = classifier.samplingFeatures(cleaned)
