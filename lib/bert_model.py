import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
import lib.tokenization as tokenization
import tensorflow_hub as hub
import numpy as np

"""
For a better view of the explaination look at the notebook.

BERT: Bidirectional Encoder Representations from Transformers

@author: Rohan Deshpande

Vocab:
    1. Masked Language Model: Mask random words in a sentence(fill in the blanks) to understand bidirectional context.
    2. Next sentence prediction: Two sentences and detect which sentences follows each other.

Process:
    Token inputs -> BERT Transformer Encoders -> Output as next sentence prediction and mask language modeling -> Pass to feedforward for classification.

Token Inputs: To generate token inputs, we need token embeddings(WordPiece Embedding) + Segment Embeddings(distinguish sentences) 
    + Position Embeddings(position of word within a sentence encoded as a vector).

1. token embeddings: [CLS] token in the beginning of the sentence and [SEP] token at the end of the sentence
2. segment embeddings: distinguish sentences with tokens assigned to each sentence
3. position embedding: unique tokens assigned to each word in the sentence.
"""

# # BERT Layer and Tokenizer from BERT team

# Bert Pretrained Layer from Google
bert_layer = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=True
)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)


# Test tokenizer
def test_encode_mechanism():
    normal_sentence = "covid-19 sucks like hell!"
    tokenize_sentence = tokenizer.tokenize(normal_sentence)
    print("Tokens for BERT: ", tokenize_sentence)
    bert_input_sentence = ["[CLS]"] + tokenize_sentence + ["[SEP]"]
    print("Token Ids for BERT: ", tokenizer.convert_tokens_to_ids(bert_input_sentence))


# BERT Custom Built Encoder
def bert_encode(texts, max_length=128):
    all_token_ids = []
    all_input_ids = []
    all_segment_ids = []
    for sentence in texts:
        tokenize_sentence = tokenizer.tokenize(sentence)[
            :max_length
        ]  # limit sentence to only 128 tokens
        bert_input_sentence = ["[CLS]"] + tokenize_sentence + ["[SEP]"]

        unused_sentence_length = max_length - len(
            bert_input_sentence
        )  # Max sentence length is 128, input sentence length is x so unused length is 128-x
        unused_length_array = [0] * unused_sentence_length

        token_ids = (
            tokenizer.convert_tokens_to_ids(bert_input_sentence) + unused_length_array
        )
        bert_input_ids = ([1] * len(bert_input_sentence)) + unused_length_array
        bert_segment_ids = [0] * max_length

        all_token_ids.append(token_ids)
        all_input_ids.append(bert_input_ids)
        all_segment_ids.append(bert_segment_ids)
    return (
        np.array(all_token_ids),
        np.array(all_input_ids),
        np.array(all_segment_ids),
    )

# BERT Pretrained Layer with Custom Feed-Forward Neural Network. hp is for hyper-parameter tuning for keras-tuner
def build_model(hp):
    input_word_ids = keras.Input(
        shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name="input_word_ids"
    )
    input_mask = keras.Input(
        shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name="input_mask"
    )
    segment_ids = keras.Input(
        shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name="segment_ids"
    )

    pooled_output, sequence_output = bert_layer(
        [input_word_ids, input_mask, segment_ids]
    )
    clf_output = sequence_output[:, 0, :]
    neural_inputs = keras.layers.Dense(
        units=hp.Int(
            "units_hidden_1", min_value=32, max_value=256, step=32, default=128
        ),
        activation="elu",
        kernel_initializer="he_normal",
    )(clf_output)
    neural_inputs = keras.layers.Dropout(
        rate=hp.Float("dropout_1", min_value=0.1, max_value=0.9, default=0.2, step=0.1)
    )(neural_inputs)
    neural_inputs = keras.layers.Dense(
        units=hp.Int(
            "units_hidden_2", min_value=32, max_value=256, step=32, default=64
        ),
        activation="elu",
        kernel_initializer="he_normal",
    )(neural_inputs)
    neural_inputs = keras.layers.Dropout(
        rate=hp.Float("dropout_2", min_value=0.1, max_value=0.9, default=0.2, step=0.1)
    )(neural_inputs)
    out = keras.layers.Dense(
        1,
        activation="sigmoid",
    )(neural_inputs)
    model = keras.models.Model(
        inputs=[input_word_ids, input_mask, segment_ids], outputs=out
    )
    model.compile(
        keras.optimizers.Adam(hp.Choice("learning_rate", [5e-5, 3e-5, 1e-5])),
        loss=keras.losses.binary_crossentropy,
        metrics=["accuracy"],
    )

    return model
