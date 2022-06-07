import os
import json

from tqdm import tqdm
from scipy.special import expit
from datasets import load_dataset
from experiments.model import Classifier
from experiments.data_loader import SampleGenerator

from data import DATA_DIR


def get_scores(generator, classifier, language):
    """
    Feed forward each batch from the models and updates the document object with the predicted scores
    :param generator: The actual generator
    :param classifier: Out pre-trained models
    :param language: The language of the data we feed to the model
    :return: A list with the updated documents
    """
    updated_documents = []
    for x_batch, documents in tqdm(generator):
        if len(x_batch):
            predictions = expit(classifier(x_batch))
            for doc, prediction in zip(documents, predictions):
                scores = [float(p) for p in prediction]
                try:
                    doc['scores'][language] = scores
                except KeyError:
                    doc.update({'scores': {language: scores}})
                updated_documents.append(doc)
        else:
            for doc in documents:
                updated_documents.append(doc)

    return updated_documents


def scoring(bert_path, loading, language, use_adapters, use_ln, bottleneck_size,
            n_frozen_layers, label_level, model_name, max_document_length, output_name, **kwargs):

    with open(os.path.join(DATA_DIR, 'eurovoc_concepts.json')) as file:
        label_index = {concept: idx for idx, concept in enumerate(json.load(file)[label_level])}

    dataset = load_dataset('nlpaueb/multi_eurlex', language=language)

    if loading:
        # Initialize a model
        classifier = Classifier(bert_model_path=bert_path, num_labels=len(label_index))
        classifier.adapt_model(use_adapters=use_adapters,
                               use_ln=use_ln,
                               bottleneck_size=bottleneck_size,
                               num_frozen_layers=n_frozen_layers)

        # Load the weights
        classifier.load_model(model_name)
    else:
        classifier = kwargs['classifier']

    # Instantiate development generator
    dev_generator = SampleGenerator(
        dataset=dataset['validation'],
        label_index=label_index, lang=language,
        bert_model_path=bert_path, batch_size=8, shuffle=False,
        max_document_length=max_document_length, scoring=True)

    # Get the dev scores
    dev_documents = get_scores(generator=dev_generator, classifier=classifier, language=language)

    # Instantiate train generator
    train_generator = SampleGenerator(
        dataset=dataset['train'],
        label_index=label_index, lang=language,
        bert_model_path=bert_path, batch_size=8, shuffle=False,
        max_document_length=max_document_length, scoring=True)

    # Get the train scores
    train_documents = get_scores(generator=train_generator, classifier=classifier, language=language)

    updated_data = {'train': train_documents, 'validation': dev_documents, 'test': dataset['test']}

    with open(f'{DATA_DIR}/{output_name}', 'w') as f:
        json.dump(updated_data, f, sort_keys=True, indent=4, ensure_ascii=False)

    print('The scores were saved.')
