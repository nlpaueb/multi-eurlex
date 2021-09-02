from typing import Dict
import tensorflow as tf
import numpy as np
import random
from transformers import AutoTokenizer
from datasets import Dataset


class SampleGenerator(tf.keras.utils.Sequence):

    def __init__(self, dataset: Dataset,
                 label_index: Dict[str, int],
                 bert_model_path: str,
                 lang: str,
                 multilingual_train: bool = False,
                 batch_size: int = 8,
                 shuffle: bool = True,
                 max_document_length: int = 512):
        self.documents = dataset
        self.batch_size = batch_size
        self.lang = lang
        self.label_index = label_index
        self.indices = np.arange(len(dataset))
        self.max_document_length = max_document_length
        self.shuffle = shuffle
        self.multilingual_train = multilingual_train
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_path)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.documents) / self.batch_size))

    def vectorize_data(self, documents, special_token=''):
        x = np.asarray(self.tokenizer.batch_encode_plus(
            [special_token + document['text'][self.lang if not self.multilingual_train else
            random.choice([lang for lang in document['text'] if document['text'][lang] is not None and lang in self.lang])] for document in documents],
            max_length=self.max_document_length, padding='max_length',
            truncation=True)['input_ids'],
                       dtype=np.int32)
        y = np.zeros((len(documents), len(self.label_index)), dtype=np.float32)
        for i, document in enumerate(documents):
            for j, concept_id in enumerate(sorted(document['labels'])):
                if concept_id in self.label_index:
                    y[i][self.label_index[concept_id]] = 1
        return [x, y]

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of batch's sequences + targets
        documents = [self.documents[k] for k in indices]
        # Clear empty documents
        if not self.multilingual_train:
            valid_docs = [document for document in documents if document['text'][self.lang] is not None]
        else:
            valid_docs = [document for document in documents if document['text'][self.lang[0]] is not None]
        if len(valid_docs) == 0:
            return [], []
        x_batch, y_batch = self.vectorize_data(valid_docs, special_token='<extra_id_0>'
                                               if 'mt5' in self.tokenizer.name_or_path else '')
        return x_batch, y_batch

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)

