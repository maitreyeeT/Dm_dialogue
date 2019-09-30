import json
import os

import pandas as pd
from pandas.io.json import json_normalize

path_to_json = '/home/maitreyee/Development/Dm_develop/data/DBDC3/dbdc3_revised/en/dev/IRIS_100'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if
              pos_json.endswith('.json')]
dfs = []
mrgd_brkdwn_chts = pd.DataFrame()
for index, js in enumerate(json_files):
  try:
    with open(os.path.join(path_to_json, js), 'r') as jd:
      jdata = jd.read()
      load_json = json.loads(jdata)
      brkdwn_chts = json_normalize(load_json['turns'])
      try:
        annotations_data = json_normalize(data=load_json['turns'],
                                          record_path=['annotations'],
                                          meta=['turn-index', 'speaker',
                                                'breakdown', 'annotator-id',
                                                'utterance'])
      except:
        pass
      dfs.append(brkdwn_chts)

      mrgd_brkdwn_chts = pd.concat(dfs, ignore_index=True)
      # print(brkdwn_chts.head())
  except ValueError:
    print('decoding json failed')

mrgd_brkdwn_chts.to_csv('brkdown_corpora2.csv', sep='\t')
