import os
from data import DATASET_DIR, TOKENIZER_DIR
import numpy as np
import json
import tqdm
from collections import Counter


CORPUS_DIR = os.path.join(DATASET_DIR, 'multi-eurlex', 'corpus')
LANGS = ['en', 'bg', 'cs', 'da', 'de', 'el', 'es', 'et', 'fi', 'fr', 'hr', 'hu',
         'it', 'lt', 'lv', 'mt', 'nl', 'pl', 'pt', 'ro', 'sk', 'sl', 'sv']

filenames = []
for SUBSET in ['train', 'dev', 'test']:
    with open(os.path.join(DATASET_DIR, 'multi-eurlex', f'multi_eurlex_{SUBSET}_ids.json')) as file:
        filenames += [os.path.join(CORPUS_DIR, f'{idx}.json') for idx in json.load(file)]


with open(os.path.join(DATASET_DIR, 'multi-eurlex', 'eurovoc_concepts.json')) as file:
    eurovoc_concepts = json.load(file)
    eurovoc_concepts.pop('level_4', None)
    eurovoc_concepts.pop('level_5', None)
    eurovoc_concepts.pop('level_6', None)
    eurovoc_concepts.pop('level_7', None)
    eurovoc_concepts.pop('level_8', None)

avg_concepts = {key: [] for key in eurovoc_concepts.keys()}
unique_concepts = {key: set() for key in eurovoc_concepts.keys()}
count_concepts = {key: [] for key in eurovoc_concepts.keys()}
for filename in tqdm.tqdm(filenames):
    with open(filename) as file:
        data = json.load(file)
    for key in avg_concepts.keys():
        avg_concepts[key].append(len(data['concepts'][key]))
        unique_concepts[key].update(data['concepts'][key])
        count_concepts[key].extend(data['concepts'][key])

for key, value in avg_concepts.items():
    print(f'{key:<10}\t{len(eurovoc_concepts[key]):<4}\t\t{len(unique_concepts[key]):<4}\t{np.mean(value):.1f}')

for key, values in count_concepts.items():
    counts = Counter(values)
    freq = sum([1 for key_1, value in counts.items() if value >= 10])
    infreq = len(eurovoc_concepts[key]) - freq
    print(f'{key:<10}\t{freq}\t{infreq}')
