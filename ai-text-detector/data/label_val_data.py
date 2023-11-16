import glob
import json

import numpy as np
import pandas as pd

elems = dict()

for idx, p in enumerate(sorted(glob.glob('../data_scraping/*.json')), start=-1):
    with open(p, 'r') as f:
        file = json.load(f)
    elems[idx] = file
    
    if idx == -1:
        continue
    assert file['idx'] == idx

print(np.unique([v['score'] for v in elems.values()], return_counts=True))

baseline = elems[0]['score']
labels = elems[0]['answer']

for idx in range(len(elems) - 1):
    labels[idx] = int(elems[idx]['score'] > baseline)

print(len(labels) - sum(labels), 1, sum(labels))

data = pd.read_csv('./val_data.tsv', sep='\t')

data.columns = ['text']
data['is_generated'] = labels

data.to_csv('./labelled_validation_data.tsv', sep='\t', index=False)
