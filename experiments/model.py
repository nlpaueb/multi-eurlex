# Adapters code inspired by
# Hosein Mohebbi's project:
# https://github.com/hmohebbi/TF-Adapter-BERT

import tensorflow as tf
import copy
import os
from data import MODELS_DIR
from transformers.activations_tf import get_tf_activation
from transformers import AutoConfig, TFAutoModel, TFMT5EncoderModel

NATIVE_BERT = {
    'en': 'bert-base-uncased',
    'da': 'DJSammy/bert-base-danish-uncased_BotXO,ai',
    'de': 'deepset/gbert-base',
    'nl': 'pdelobelle/robbert-v2-dutch-base',
    'sv': 'KB/bert-base-swedish-cased',
    'es': 'BSC-TeMU/roberta-base-bne',
    'fr': 'camembert-base',
    'it': 'Musixmatch/umberto-commoncrawl-cased-v1',
    'pt': 'neuralmind/bert-base-portuguese-cased',
    'ro': 'dumitrescustefan/bert-base-romanian-uncased-v1',
    'bg': 'Geotrend/bert-base-bg-cased',
    'cs': 'UWB-AIR/Czert-B-base-cased',
    'hr': 'EMBEDDIA/crosloengual-bert',
    'pl': 'dkleczek/bert-base-polish-uncased-v1',
    'sl': 'EMBEDDIA/crosloengual-bert',
    'et': 'tartuNLP/EstBERT',
    'fi': 'TurkuNLP/bert-base-finnish-uncased-v1',
    'hu': 'SZTAKI-HLT/hubert-base-cc',
    'lt': 'Geotrend/bert-base-lt-cased',
    'lv': 'EMBEDDIA/litlat-bert',
    'el': 'nlpaueb/bert-base-greek-uncased-v1'
}


class Adapter(tf.keras.Model):
    """Adapter layer (Houlsby et al., 2019)
    (https://arxiv.org/abs/1902.00751)
    Adapters rely on a bottleneck architecture. The adapters first project the original d-dimensional
    features into a smaller dimension, m, apply a nonlinearity, then project back to d dimensions.
    """

    def __init__(self, input_size, bottleneck_size, non_linearity, *inputs, **kwargs):
        super(Adapter, self).__init__(name="Adapter")

        self.non_linearity = get_tf_activation(non_linearity)

        self.down_project = tf.keras.layers.Dense(
            bottleneck_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3),
            bias_initializer="zeros",
            name="feedforward_downproject")

        self.up_project = tf.keras.layers.Dense(
            input_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3),
            bias_initializer="zeros",
            name="feedforward_upproject")

    def call(self, inputs, **kwargs):
        output = self.down_project(inputs=inputs)
        output = self.non_linearity(output)
        output = self.up_project(inputs=output)
        output = output + inputs
        return output

    def get_config(self):
        pass


class T5Adapter(tf.keras.Model):
    """Adapter layer (Houlsby et al., 2019)
    (https://arxiv.org/abs/1902.00751)
    Adapters rely on a bottleneck architecture. The adapters first project the original d-dimensional
    features into a smaller dimension, m, apply a nonlinearity, then project back to d dimensions.
    """

    def __init__(self, input_size, bottleneck_size, *inputs, **kwargs):
        super(T5Adapter, self).__init__(name="Adapter")
        self.wi_0 = tf.keras.layers.Dense(bottleneck_size, use_bias=False, name="wi_0",
                                          kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3))
        self.wi_1 = tf.keras.layers.Dense(bottleneck_size, use_bias=False, name="wi_1",
                                          kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3))
        self.wo = tf.keras.layers.Dense(input_size, use_bias=False, name="wo",
                                        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=1e-3))
        self.act = get_tf_activation("gelu_new")

    def call(self, inputs, **kwargs):
        hidden_gelu = self.act(self.wi_0(inputs))
        hidden_linear = self.wi_1(inputs)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.wo(hidden_states)
        output = hidden_states + inputs
        return output


class TFBertSelfOutput(tf.keras.layers.Layer):
    def __init__(self, pretrained_self_dense: tf.keras.layers.experimental.EinsumDense,
                 pretrained_self_ln: tf.keras.layers.LayerNormalization, config: AutoConfig,
                 bottleneck_size: int, **kwargs):
        super().__init__(**kwargs)

        self.dense = copy.deepcopy(pretrained_self_dense)
        self.LayerNorm = copy.deepcopy(pretrained_self_ln)
        self.adapter = Adapter(input_size=config.hidden_size,
                               bottleneck_size=bottleneck_size,
                               non_linearity=config.hidden_act)
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states


class TFBertOutput(tf.keras.layers.Layer):
    def __init__(self, pretrained_out_dense: tf.keras.layers.experimental.EinsumDense,
                 pretrained_out_ln: tf.keras.layers.LayerNormalization, config: AutoConfig,
                 bottleneck_size: int, **kwargs):
        super().__init__(**kwargs)
        self.dense = copy.deepcopy(pretrained_out_dense)
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.adapter = Adapter(input_size=config.hidden_size,
                               bottleneck_size=bottleneck_size,
                               non_linearity=config.hidden_act)
        self.LayerNorm = copy.deepcopy(pretrained_out_ln)

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states


class TFT5LayerFF(tf.keras.layers.Layer):
    def __init__(self, pretrained_self_dense: tf.keras.layers.experimental.EinsumDense,
                 pretrained_self_ln: tf.keras.layers.LayerNormalization, config: AutoConfig,
                 bottleneck_size: int, **kwargs):
        super().__init__(**kwargs)
        self.DenseReluDense = copy.deepcopy(pretrained_self_dense)
        self.layer_norm = copy.deepcopy(pretrained_self_ln)
        self.adapter = Adapter(input_size=config.hidden_size,
                               bottleneck_size=bottleneck_size,
                               non_linearity='gelu_new')
        self.dropout = tf.keras.layers.Dropout(rate=config.dropout_rate)

    def call(self, hidden_states, training=False):
        normed_hidden_states = self.layer_norm(hidden_states)
        dense_output = self.DenseReluDense(normed_hidden_states, training=training)
        dense_output = self.adapter(self.dropout(dense_output, training=training))
        hidden_states = hidden_states + dense_output
        return hidden_states


class Classifier(tf.keras.Model):

    def __init__(self, bert_model_path, num_labels, *inputs, **kwargs):
        super(Classifier, self).__init__(name="BertClassifier")
        if 'bert' in bert_model_path or bert_model_path in NATIVE_BERT.values():
            self.bert = TFAutoModel.from_pretrained(bert_model_path, from_pt=True)
        else:
            self.bert = TFMT5EncoderModel.from_pretrained(bert_model_path, from_pt=True)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.classifier = tf.keras.layers.Dense(
            num_labels,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            use_bias=True,
            name="classifier")

    def call(self, inputs, **kwargs):
        # First Hidden state
        outputs = self.bert(inputs, **kwargs)
        pooled_out = outputs[0][:, 0]

        dropped_out = self.dropout(pooled_out, training=kwargs.get("training", False))
        output = self.classifier(dropped_out)
        return output

    def adapt_model(self, num_frozen_layers=None, bottleneck_size=256, use_adapters=False, use_ln=False):
        if self.bert.config.model_type in ['xlm-roberta', 'roberta']:
            self.adapt_model_roberta(num_frozen_layers=num_frozen_layers,
                                     bottleneck_size=bottleneck_size,
                                     use_adapters=use_adapters,
                                     use_ln=use_ln)
        elif self.bert.config.model_type == 'bert':
            self.adapt_model_bert(num_frozen_layers=num_frozen_layers,
                                  bottleneck_size=bottleneck_size,
                                  use_adapters=use_adapters,
                                  use_ln=use_ln)
        elif self.bert.config.model_type == 'mt5':
            self.adapt_model_t5(num_frozen_layers=num_frozen_layers,
                                bottleneck_size=bottleneck_size,
                                use_adapters=use_adapters,
                                use_ln=use_ln)
        else:
            raise NotImplementedError(f'Model type {self.bert.config.model_type} is not supported for adaptation')

    def adapt_model_bert(self, num_frozen_layers=None, bottleneck_size=256, use_adapters=False, use_ln=False):
        if use_adapters:
            # Add Adapters
            for i in range(self.bert.bert.config.num_hidden_layers):
                # instantiate
                self.bert.bert.encoder.layer[i].attention.dense_output = TFBertSelfOutput(
                    self.bert.bert.encoder.layer[i].attention.dense_output.dense,
                    self.bert.bert.encoder.layer[i].attention.dense_output.LayerNorm,
                    self.bert.bert.config,
                    bottleneck_size)
                self.bert.bert.encoder.layer[i].bert_output = TFBertOutput(
                    self.bert.bert.encoder.layer[i].bert_output.dense,
                    self.bert.bert.encoder.layer[i].bert_output.LayerNorm,
                    self.bert.bert.config,
                    bottleneck_size)
            # Freeze all hidden layers
            num_frozen_layers = self.bert.bert.config.num_hidden_layers

        if num_frozen_layers:
            # Freeze BERT layers
            self.bert.bert.embeddings.trainable = False
            self.bert.bert.pooler.trainable = False
            for i in range(num_frozen_layers):
                self.bert.bert.encoder.layer[i].attention.self_attention.trainable = False
                self.bert.bert.encoder.layer[i].attention.dense_output.dense.trainable = False
                if not (use_adapters or use_ln):
                    self.bert.bert.encoder.layer[i].attention.dense_output.LayerNorm.trainable = False
                self.bert.bert.encoder.layer[i].intermediate.trainable = False
                self.bert.bert.encoder.layer[i].bert_output.dense.trainable = False
                if not (use_adapters or use_ln):
                    self.bert.bert.encoder.layer[i].bert_output.LayerNorm.trainable = False

    def adapt_model_roberta(self, num_frozen_layers=None, bottleneck_size=256, use_adapters=False, use_ln=False):
        if use_adapters:
            # Add Adapters
            for i in range(self.bert.roberta.config.num_hidden_layers):
                # instantiate
                self.bert.roberta.encoder.layer[i].attention.dense_output = TFBertSelfOutput(
                    self.bert.roberta.encoder.layer[i].attention.dense_output.dense,
                    self.bert.roberta.encoder.layer[i].attention.dense_output.LayerNorm,
                    self.bert.roberta.config,
                    bottleneck_size)
                self.bert.roberta.encoder.layer[i].bert_output = TFBertOutput(
                    self.bert.roberta.encoder.layer[i].bert_output.dense,
                    self.bert.roberta.encoder.layer[i].bert_output.LayerNorm,
                    self.bert.roberta.config,
                    bottleneck_size)
            # Freeze all hidden layers
            num_frozen_layers = self.bert.roberta.config.num_hidden_layers

        if num_frozen_layers:
            # Freeze RoBERTa layers
            self.bert.roberta.embeddings.trainable = False
            if use_adapters or use_ln:
                self.bert.roberta.embeddings.LayerNorm.trainable = True
            self.bert.roberta.pooler.trainable = False
            for i in range(num_frozen_layers):
                self.bert.roberta.encoder.layer[i].attention.self_attention.trainable = False
                self.bert.roberta.encoder.layer[i].attention.dense_output.dense.trainable = False
                if not (use_adapters or use_ln):
                    self.bert.roberta.encoder.layer[i].attention.dense_output.LayerNorm.trainable = False
                self.bert.roberta.encoder.layer[i].intermediate.trainable = False
                self.bert.roberta.encoder.layer[i].bert_output.dense.trainable = False
                if not (use_adapters or use_ln):
                    self.bert.roberta.encoder.layer[i].bert_output.LayerNorm.trainable = False

    def adapt_model_t5(self, num_frozen_layers=None, bottleneck_size=256, use_adapters=False, use_ln=True):
        if use_adapters:
            # Add Adapters
            for i in range(self.bert.config.num_layers):
                # instantiate
                self.bert.encoder.block[i].layer[1] = TFT5LayerFF(
                    self.bert.encoder.block[i].layer[1].DenseReluDense,
                    self.bert.encoder.block[i].layer[1].layer_norm,
                    self.bert.encoder.config,
                    bottleneck_size)
            # Freeze all hidden layers
            num_frozen_layers = self.bert.config.num_layers

        if num_frozen_layers:
            # Freeze mT5 layers
            self.bert.shared.trainable = False
            for i in range(num_frozen_layers):
                self.bert.encoder.block[i].layer[0].SelfAttention.trainable = False
                if not (use_adapters or use_ln):
                    self.bert.encoder.block[i].layer[0].layer_norm.trainable = False
                self.bert.encoder.block[i].layer[1].DenseReluDense.trainable = False
                if not (use_adapters or use_ln):
                    self.bert.encoder.block[i].layer[1].layer_norm.trainable = False

    def save_model(self, filepath: str):
        os.mkdir(filepath)
        self.save_weights(os.path.join(filepath, 'bert'))

    def load_model(self, filepath: str):
        self.load_weights(os.path.join(MODELS_DIR, filepath, 'bert'))

    def get_config(self):
        pass


if __name__ == '__main__':
    classifier = Classifier(bert_model_path='xlm-roberta-base', num_labels=100)
    classifier.adapt_model(use_adapters=True, num_frozen_layers=None)
    classifier(tf.zeros((1, 10), dtype='int32'))
    classifier.summary()
    classifier.save_model(os.path.join(MODELS_DIR, 'test_model'))

    classifier2 = Classifier(bert_model_path='xlm-roberta-base', num_labels=100)
    classifier2.load_model(os.path.join(MODELS_DIR, 'test_model'))
    preds = classifier2(tf.zeros((1, 10), dtype='int32'))
    print(preds.shape)

