import csv
import pandas as pd

class Analysis():
    def __init__(self):
        filepath = './data/1.tsv'
        with open(filepath) as tsvread:
            reader = csv.reader(tsvread, delimiter='\t')
            utterances = []
            for row in reader:
                utterances.append(row)

            self.df = pd.DataFrame(data=utterances,columns=['timestamp','p1','p2','utterance'])


    def analyse_df(self):
        #find the average number of words, shortest and longest sentences
        df_utt_length = self.df['utterance'].str.split().str.len()
        df_shortest_utterance = min(df_utt_length)
        df_longest_utterance = max(df_utt_length)
        #df_avg_utt_length = (df_utt_length)
        print([df_longest_utterance,df_shortest_utterance])

analysis_class = Analysis()
print(analysis_class.analyse_df())