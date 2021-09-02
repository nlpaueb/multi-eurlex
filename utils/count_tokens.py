from datasets import load_dataset
import tqdm
import numpy as np
LANGS = ['en', 'da', 'de', 'nl', 'sv', 'bg', 'cs', 'hr', 'pl', 'sk', 'sl', 'es',
         'fr', 'it', 'pt', 'ro', 'et', 'fi', 'hu', 'lt', 'lv', 'el', 'mt']


total_labels = 0
document_lens = {LANG:[] for LANG in LANGS}
dataset = load_dataset('multi_eurlex', 'all_languages')

for subset in ['train', 'dev', 'test']:
    for document in tqdm.tqdm(dataset):
        for LANG in LANGS:
            if document['text'][LANG] is not None:
                document_lens[LANG].append(len(document['text'][LANG] .split()))

for LANG in LANGS:
    print(f'{LANG}:\t {np.mean(document_lens[LANG]):.1f} / {np.median(document_lens[LANG]):.1f} '
          f'/ {np.sum((np.asarray(document_lens[LANG]) <= 500).astype("int32"))*100/ len(document_lens[LANG]):.1f}')
