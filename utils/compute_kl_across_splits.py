from data import DATASET_DIR
import json, os
from collections import Counter
from tensorflow.keras.losses import kl_divergence
import numpy as np
import random

document_ids = {}
dist = {}
for subset in ['train', 'dev', 'test']:
    with open(os.path.join(DATASET_DIR, 'multi-eurlex', f'multi_eurlex_{subset}_ids.json')) as file:
        document_ids[subset] = json.load(file)

document_ids_all = document_ids['train'] + document_ids['dev'] + document_ids['test']
random.shuffle(document_ids_all)
random.seed(1)
document_ids['train'] = document_ids_all[:55000]
document_ids['dev'] = document_ids_all[55000:60000]
document_ids['test'] = document_ids_all[60000:]


for subset in ['train', 'dev', 'test']:
    dist[subset] = []
    for filename in document_ids[subset]:
        with open(os.path.join(DATASET_DIR, 'multi-eurlex', 'train', f'{filename}.json')) as file:
            labels = json.load(file)['concepts']['original']
            dist[subset].extend(labels)

    dist[subset] = Counter(dist[subset])

keys = dist['train'].keys()
for subset in ['train', 'dev', 'test']:
    dist[subset] = [dist[subset][label_id] for label_id in keys]
    dist[subset] = [val/np.sum(dist[subset]) for val in dist[subset]]

for subset in ['dev', 'test']:
    print(kl_divergence(dist['train'], dist[subset]))
print()