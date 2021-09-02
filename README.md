# Multi-EURLEX

## MultiEURLEX - A multi-lingual and multi-label legal document classification dataset for zero-shot cross-lingual transfer

This is the code used for the experiments described in the following paper:


> I. Chalkidis, M. Fergadiotis, and I. Androutsopoulos, "MultiEURLEX - A multi-lingual and multi-label legal document classification dataset for zero-shot cross-lingual transfer". Proceedings of the 2021 Conference on Empirical Methods
               in Natural Language Processing, Punta Cana, Dominican Republic, 2021 (xxx)

## Requirements:

* tensorflow==2.3.1
* tensorflow-addons==0.5
* transformers==4.3.3
* tokenizers==0.10.1
* scipy==1.6.1
* torch==1.7.1
* tqdm==4.43.0
* cudatoolkit==10.1.243 (for GPU acceleration)
* cudnn==7.6.0 (for GPU acceleration)

## Quick start:

### Install python requirements:

```
pip install -r requirements.txt
```

### Download dataset (MultiEURLEX):

The dataset is hosted and been described in detail in the Hugging Face Datasets (https://huggingface.co/datasets/multi_eurlex). It is automatically downloaded and used by the Trainer. 
If you want to review and familiarize your self with the dataset, you can download it usingthe following Python code:

```python
from datasets import load_dataset
dataset = load_dataset('multi_eurlex', languages='all_languages')
```

### Train a model:

The following configuration (command-line) arguments can be used:

* **'bert_path'** (default='xlm-roberta-base'): The name of the pretrained transformer-based model hosted by Hugging Face, or the full path to a local directory.
* **'native_bert'** (default=False): If the ISO code of a language (e.g., 'en') is provided, then the relevant monolingual model will be fine-tuned.
* **'multilingual_train'** (default=False): If True, the model will be trained across multiple languages ('train_langs').
* **'use_adapters'** (default=False) If True, the model will be fine-tuned using Adapter modules (Houlsby et al., 2019).
* **'use_ln'** (default=False) If True, only the parameter of the LayerNorm layers of the the model will be fine-tuned
* **'bottleneck_size'** (default=256) The size of the bottleneck layer in Adapter modules (if used).
* **'n_frozen_layers'** (default=0) The number of the initial layers that will remain frozen in fine-tuning.
* **'epochs'** (default=70) The number of the maximum training epochs (Early stopping with patience 5 is used by default).
* **'batch_size'** (default=8) The number of the samples in a single batch.
* **'learning_rate'** (default=3e-5) The initial learning rate to be used by the Adam optimizer.
* **'label_smoothing'** (default=0.0) The rate of label smoothing (Szegedy  et  al.,2016).
* **'max_document_length'** (default=512) The maximum length of tokens to be considered per document.
* **'monitor'** (default='val_rp') The score to be monitored for early stopping ('val_rp' or 'val_loss')
* **'train_lang'** (default='en') The ISO code of the training language (e.g., 'en') in a *one-to-many* setting.
* **'train_langs'** (default=['en']) The list of languages to be used for fine-tuning, in *many-to-one* setting.
* **'eval_langs'** (default='all') The list of languages to be used for evaluation.
* **'label_level'** (default='level_3') The level of EUROVOC (e.g., 'level_1', 'level_2', 'level_3', 'all') used for the classification task.

You can run experiments by simply calling:

```
python trainer.py --bert_path 'xlm-roberta-base' --use_adapters True --train_lang 'en' --label_level 'level_1'
```

