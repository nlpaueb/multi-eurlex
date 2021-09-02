import logging
import click
import os
import json
import time
import numpy as np
from scipy.special import expit
import tensorflow as tf
from tensorflow_addons.metrics import MeanMetricWrapper
from experiments.model import Classifier, NATIVE_BERT
from experiments.data_loader import SampleGenerator
from data import MODELS_DIR, DATA_DIR
from datasets import load_dataset
from utils.logger import setup_logger
from utils.retrieval_metrics import mean_rprecision, mean_ndcg_score, mean_recall_k
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
log_name = setup_logger()
LOGGER = logging.getLogger(__name__)
cli = click.Group()


@cli.command()
@click.option('--bert_path', default='xlm-roberta-base')
@click.option('--native_bert', default=False)
@click.option('--multilingual_train', default=False)
@click.option('--use_adapters', default=False)
@click.option('--use_ln', default=False)
@click.option('--bottleneck_size', default=256)
@click.option('--n_frozen_layers', default=0)
@click.option('--epochs', default=70)
@click.option('--batch_size', default=8)
@click.option('--learning_rate', default=3e-5)
@click.option('--label_smoothing', default=0.0)
@click.option('--max_document_length', default=512)
@click.option('--monitor', default='val_rp')
@click.option('--train_lang', default='en')
@click.option('--train_langs', default=['en'])
@click.option('--eval_langs', default=['en', 'da', 'de', 'nl', 'sv', 'bg', 'cs', 'hr', 'pl', 'sk', 'sl', 'es',
                                       'fr', 'it', 'pt', 'ro', 'et', 'fi', 'hu', 'lt', 'lv', 'el', 'mt'])
@click.option('--label_level', default='level_3')
@click.option('--train_sample', default=10)
@click.option('--eval_samples', default=10)
def train(bert_path, native_bert, use_adapters, use_ln, bottleneck_size, n_frozen_layers, epochs, batch_size,
          learning_rate, label_smoothing, monitor, train_lang, train_langs, eval_langs, label_level, multilingual_train,
          max_document_length, train_samples, eval_samples):

    bottleneck_size = int(bottleneck_size)
    n_frozen_layers = int(n_frozen_layers)
    epochs = int(epochs)
    batch_size = int(batch_size)
    learning_rate = float(learning_rate)
    label_smoothing = float(label_smoothing)

    if multilingual_train and not train_langs:
        train_langs = eval_langs

    with open(os.path.join(DATA_DIR, 'eurovoc_concepts.json')) as file:
        label_index = {concept: idx for idx, concept in enumerate(json.load(file)[label_level])}

    # Load native bert
    if native_bert in NATIVE_BERT:
        bert_path = NATIVE_BERT[native_bert]

    # Log configuration
    LOGGER.info('\n----------------------------------------')
    LOGGER.info(log_name)
    LOGGER.info('----------------------------------------')
    LOGGER.info('Task Parameters')
    LOGGER.info('----------------------------------------')
    LOGGER.info(f'     Training Language: {train_lang if not multilingual_train else train_langs}')
    LOGGER.info(f'  Evaluation Languages: {eval_langs}')
    LOGGER.info(f'      EUROVOC Concepts: {label_level} ({len(label_index)})')
    LOGGER.info(f'   Max Document Length: {max_document_length}')
    LOGGER.info('----------------------------------------')
    LOGGER.info('Model Parameters')
    LOGGER.info('----------------------------------------')
    LOGGER.info(f'            Bert Model: {bert_path}')
    LOGGER.info(f'         Frozen Layers: {n_frozen_layers}')
    LOGGER.info(f'          Use Adapters: {use_adapters}')
    LOGGER.info(f'LayerNorm ONLY (LNFIT): {use_ln}')
    LOGGER.info(f'      Bottle-neck Size: {bottleneck_size}')
    LOGGER.info('----------------------------------------')
    LOGGER.info('Training Parameters')
    LOGGER.info('----------------------------------------')
    LOGGER.info(f'                Epochs: {epochs}')
    LOGGER.info(f'            Batch Size: {batch_size}')
    LOGGER.info(f'         Learning Rate: {learning_rate}')
    LOGGER.info(f'       Label Smoothing: {label_smoothing}')
    LOGGER.info(f'     EarlyStop Monitor: {monitor}')
    LOGGER.info('----------------------------------------\n')

    # Load dataset
    train_dataset = load_dataset('multi_eurlex', language='all_languages',
                                 languages=train_langs if multilingual_train else [train_lang],
                                 label_level=label_level, split='train')
    eval_dataset = load_dataset('multi_eurlex', language='all_languages',
                                languages=eval_langs, label_level=label_level)

    # Instantiate training / development generators
    LOGGER.info(f'{len(train_dataset)} documents will be used for training')
    train_generator = SampleGenerator(dataset=train_dataset[:train_samples if train_samples else len(train_dataset)],
                                      label_index=label_index,
                                      lang=train_langs if multilingual_train else train_lang,
                                      bert_model_path=bert_path, batch_size=batch_size, shuffle=True,
                                      multilingual_train=multilingual_train, max_document_length=max_document_length)

    LOGGER.info(f'{len(eval_dataset["dev"])} documents will be used for development')
    dev_generator = SampleGenerator(dataset=eval_dataset['dev'][:eval_samples if eval_samples else len(eval_dataset)],
                                    label_index=label_index,
                                    lang=train_langs if multilingual_train else train_lang,
                                    bert_model_path=bert_path, batch_size=batch_size, shuffle=False,
                                    multilingual_train=multilingual_train, max_document_length=max_document_length)
    # Instantiate Model
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))
        if monitor == 'val_loss':
            monitor_metric = None
            monitor_mode = 'min'
        elif monitor == 'val_rp':
            monitor_metric = MeanMetricWrapper(fn=r_precision, name='rp')
            monitor_mode = 'max'
        else:
            raise Exception(f'Monitor "{monitor}" is not supported')

        classifier = Classifier(bert_model_path=bert_path, num_labels=len(label_index))
        classifier.adapt_model(use_adapters=use_adapters,
                                    use_ln=use_ln,
                                    bottleneck_size=bottleneck_size,
                                    num_frozen_layers=n_frozen_layers)

        classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                                        label_smoothing=label_smoothing),
                                metrics=[monitor_metric])
        classifier(tf.zeros((1, 10), dtype='int32'))
        classifier.summary(print_fn=LOGGER.info, line_length=100)

    # Train model
    start_training_time = time.time()
    history = classifier.fit(train_generator, validation_data=dev_generator,
                                  epochs=epochs,
                                  callbacks=[tf.keras.callbacks.EarlyStopping(
                                      patience=5, monitor=monitor, mode=monitor_mode, restore_best_weights=True)])
    end_training_time = time.time()

    # Log training history
    LOGGER.info(f"\n{'-' * 100}")
    LOGGER.info('Training History')
    LOGGER.info('-' * 100)
    if monitor == 'val_loss':
        LOGGER.info('           \tLoss   \t\tVal Loss')
        for n_epoch, _ in enumerate(history.history['loss']):
            LOGGER.info(f'EPOCH #{n_epoch+1:<2}: \t{history.history["loss"][n_epoch]:.5f}'
                        f'\t\t{history.history["val_loss"][n_epoch]:.5f}')
    elif monitor == 'val_rp':
        LOGGER.info('           \tLoss   \t\tVal Loss\tRP     \t\tVal RP')
        for n_epoch, _ in enumerate(history.history['loss']):
            LOGGER.info(f'EPOCH #{n_epoch+1:<2}: \t{history.history["loss"][n_epoch]:.5f}'
                        f'\t\t{history.history["val_loss"][n_epoch]:.5f}'
                        f'\t\t{history.history["rp"][n_epoch]:.5f}'
                        f'\t\t{history.history["val_rp"][n_epoch]:.5f}')

    # Evaluate model
    LOGGER.info(f"\n{'-' * 100}")
    LOGGER.info('Evaluation Metrics')
    LOGGER.info('-' * 100)
    LOGGER.info('\nDevelopment')
    LOGGER.info('-' * 100)

    # Re-instantiate development generator
    dev_generator = SampleGenerator(dataset=eval_dataset["dev"][:eval_samples if eval_samples else len(eval_dataset)],
                                    label_index=label_index, lang=train_lang,
                                    bert_model_path=bert_path, batch_size=batch_size, shuffle=False,
                                    max_document_length=max_document_length)

    for lang_code in eval_langs:
        # Set target language
        dev_generator.lang = lang_code
        # Initialize score matrices
        n_documents = sum([1 for document in dev_generator.documents if document.text[lang_code] is not None])
        y_true = np.zeros((n_documents, len(label_index)), dtype=np.float32)
        y_pred = np.zeros((n_documents, len(label_index)), dtype=np.float32)
        count = 0
        # Predict labels batch-wise
        for x_batch, y_batch in dev_generator:
            LEN_BATCH = len(x_batch)
            if LEN_BATCH:
                yp_batch = classifier.predict(x_batch)
                y_true[count:count + LEN_BATCH] = y_batch
                y_pred[count:count + LEN_BATCH] = yp_batch
            count += LEN_BATCH
        # Log evaluation scores
        y_pred = expit(y_pred)
        scores = f'"{lang_code}": R-Precision: {mean_rprecision(y_true, y_pred)[0]*100:2.2f}\t'
        for k in range(1, 6):
            scores += f'NDCG@{k}: {mean_ndcg_score(y_true, y_pred, k=k)[0]*100:2.2f}\t'
        for k in range(1, 6):
            scores += f'R@{k}: {mean_recall_k(y_true, y_pred, k=k)[0]*100:2.2f}\t'
        LOGGER.info(scores)

    LOGGER.info('-' * 100)
    LOGGER.info('\nTest')
    LOGGER.info('-' * 100)

    # Instantiate test generator
    test_generator = SampleGenerator(dataset=eval_dataset['test'][:eval_samples if eval_samples else len(eval_dataset)],
                                     label_index=label_index, lang=train_lang,
                                     bert_model_path=bert_path, batch_size=batch_size, shuffle=False,
                                     max_document_length=max_document_length)
    for lang_code in eval_langs:
        # Set target language
        test_generator.lang = lang_code
        # Initialize score matrices
        n_documents = sum([1 for document in test_generator.documents if document.text[lang_code] is not None])
        y_true = np.zeros((n_documents, len(label_index)), dtype=np.float32)
        y_pred = np.zeros((n_documents, len(label_index)), dtype=np.float32)
        count = 0
        # Predict labels batch-wise
        for x_batch, y_batch in test_generator:
            LEN_BATCH = len(x_batch)
            if LEN_BATCH:
                yp_batch = classifier.predict(x_batch)
                y_true[count:count + LEN_BATCH] = y_batch
                y_pred[count:count + LEN_BATCH] = yp_batch
            count += LEN_BATCH
        # Log evaluation scores
        y_pred = expit(y_pred)
        scores = f'"{lang_code}": R-Precision: {mean_rprecision(y_true, y_pred)[0] * 100:2.2f}\t'
        for k in range(1, 6):
            scores += f'NDCG@{k}: {mean_ndcg_score(y_true, y_pred, k=k)[0] * 100:2.2f}\t'
        for k in range(1, 6):
            scores += f'R@{k}: {mean_recall_k(y_true, y_pred, k=k)[0] * 100:2.2f}\t'
        LOGGER.info(scores)
    LOGGER.info('-' * 100)

    LOGGER.info(f'\nTraining time: {time.strftime("%H:%M:%S", time.gmtime(end_training_time - start_training_time))} sec')
    LOGGER.info(f'Training + Evaluation time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_training_time))} sec\n')

    # Save model
    classifier.save_model(os.path.join(MODELS_DIR, LOGGER.name))


def r_precision(y_true, y_pred):
    positives = tf.reduce_sum(tf.cast(tf.equal(y_true, 1.0), dtype=tf.keras.backend.floatx()), axis=-1)
    y_true_ranked = tf.cast(tf.gather(y_true, tf.argsort(y_true, direction='DESCENDING'), batch_dims=1), dtype=tf.bool)
    y_true_pred_ranked = tf.cast(tf.gather(y_true, tf.argsort(y_pred, direction='DESCENDING'), batch_dims=1), dtype=tf.bool)
    relevant = tf.reduce_sum(tf.cast(tf.logical_and(y_true_ranked, y_true_pred_ranked), dtype=tf.keras.backend.floatx()), axis=-1)
    return tf.math.divide_no_nan(relevant, positives)


if __name__ == '__main__':
    train()
