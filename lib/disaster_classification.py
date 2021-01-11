#!/usr/bin/env python
# coding: utf-8

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from matplotlib import pylab
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from kerastuner.tuners import Hyperband
from sklearn.model_selection import train_test_split
import time

# Local imports
import lib.tokenization as tokenization
import lib.preprocess as preprocess

'''
Jupyter Config
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'nb_black')
'''

# Path to Train and Test Files
curr_dir = ".."
already_trained = True # set this to false to retrain.
train_file = os.path.join(curr_dir, "dataset", "train.csv")
test_file = os.path.join(curr_dir, "dataset", "test.csv")
sample_submission_file = os.path.join(
    curr_dir, "dataset", "sample_submission.csv"
)

# Matplotlib and Seaborn Configurations
params = {
    "legend.fontsize": "x-large",
    "figure.figsize": (16, 5),
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
}
pylab.rcParams.update(params)

df = pd.read_csv(train_file)
df.head()


df.info()


df.isnull().sum()

# Data Analysis Start

def plotBarGraph(data, labelx, labely, title, switch_axis=False):
    y = data.index.tolist()
    x = data.tolist()
    if switch_axis:
        x = data.index.tolist()
        y = data.tolist()
    sns.barplot(y=y, x=x)
    plt.title(title)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.show()


# # Tweet Locations Graph

locations_map = df.location.value_counts().sort_values(ascending=False)[:10]
plotBarGraph(locations_map, "Frequency", "Countries", "Tweet locations")


# # Top 20 Disaster-Related Keywords
keywords_map = (
    df.keyword.loc[df.target == 1].value_counts().sort_values(ascending=False)[:20]
)
plotBarGraph(keywords_map, "Frequency", "Keywords", "Top 20 disaster related keywords")


# Top 20 Non-Disaster-Related Keywords
# Notice non-disaster keywords are not actual disasters but rather general keywords that could describe a disaster. 
# These keywords can be used in our daily lives, for instance "I started panicing when I got into MIT", which clearly isn't a natural disaster lol :)
# Top 20 Non-Disaster related Keywords
keywords_non_disaster_map = df.keyword.loc[df.target == 0].value_counts(
    ascending=False
)[:20]
plotBarGraph(
    keywords_non_disaster_map,
    "Frequency",
    "Keywords",
    "Top 20 non-disaster related keywords",
)


# # Target Distrubution

sns.countplot(x="target", data=df)
plt.show()


# Removing Duplicate Text Data
text_data_duplicate_map = df.text.duplicated().value_counts()
print(text_data_duplicate_map)
plotBarGraph(
    text_data_duplicate_map,
    "Duplicate",
    "Frequency",
    "Text Data Duplicate or Not",
    switch_axis=True,
)


# Remove Duplicates and only Keeping Original Text

print("Original dataframe shape: ", df.shape)
df.drop_duplicates(subset="text", keep="first", inplace=True)
print("Dropping duplicate rows and keeping original value shape: ", df.shape)
df.text.duplicated().value_counts()

 
 # Search Keywords in Text Column for Debugging Purposes

def search_text_data(query, column="text"):
    return df[df[column].str.contains(query)][column]


print(search_text_data("volcano")[:5])

def clean_text_column(col_name, df=df):
    df["clean_text"] = ""
    df["clean_text"] = df[col_name].apply(preprocess.clean_sentence_pipeline)


clean_text_column("text")


# Comparison of cleaned texts
df[["text", "clean_text"]]


# # BERT: Bidirectional Encoder Representations from Transformers
# 
# ## Vocab:
# 1. Masked Language Model: Mask random words in a sentence(fill in the blanks) 
#     to understand bidirectional context.
# 2. Next sentence prediction: Two sentences and detect which sentences follows each other.
# 
# ## Process:
# **Token inputs -> BERT Transformer Encoders -> Output as next sentence prediction and mask language modeling
# -> Pass to feedforward for classification.**
# 
# ## Token Inputs:
# To generate token inputs, we need token embeddings(WordPiece Embedding) + Segment Embeddings(distinguish sentences) + Position Embeddings(position of word within a sentence encoded as a vector).
# 
# 1. token embeddings: [CLS] token in the beginning of the sentence and [SEP] token at the end of the sentence
# 2. segment embeddings: distinguish sentences with tokens assigned to each sentence
# 3. position embedding: unique tokens assigned to each word in the sentence.

# # BERT Layer and Tokenizer from BERT team

# Bert config
bert_layer = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=True
)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)


# ## Experiment with Tokenizer and it's functionalities

# test tokenizer
normal_sentence = "covid-19 sucks like hell!"
tokenize_sentence = tokenizer.tokenize(normal_sentence)
print("Tokens for BERT: ", tokenize_sentence)
bert_input_sentence = ["[CLS]"] + tokenize_sentence + ["[SEP]"]
print("Token Ids for BERT: ", tokenizer.convert_tokens_to_ids(bert_input_sentence))


# # BERT Encoder

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


# # BERT Pretrained Layer with Custom Feed-Forward Neural Network

MAX_SEQUENCE_LENGTH = 128


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
    out = keras.layers.Dense(1, activation="sigmoid",)(neural_inputs)
    model = keras.models.Model(
        inputs=[input_word_ids, input_mask, segment_ids], outputs=out
    )
    model.compile(
        keras.optimizers.Adam(hp.Choice("learning_rate", [5e-5, 3e-5, 1e-5])),
        loss=keras.losses.binary_crossentropy,
        metrics=["accuracy"],
    )

    return model


# # Hyperband Hyperparameter Tuning Mechanism
# I don't know why kaggle couldn't save the hyperparameter folder.

tuner = Hyperband(
    build_model,
    objective="val_accuracy",
    max_epochs=3,
    project_name="../hyperparameter/hyperband",
)


# # Train-Test Split

# Prepare final datasplits
x_train, x_val, y_train, y_val = train_test_split(
    df.clean_text, df.target, train_size=0.85
)


# # BERT Encoding Calls on Train and Validation

bert_xtrain = bert_encode(x_train)
bert_xval = bert_encode(x_val)


# # Keras Callbacks

# model_path = os.path.join("..", "models", "keras_hyperband_disaster_prediction.h5") for local development
tensorboard_logs_path = os.path.join("..", "tensorboard_logs")
# Keras callbacks
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath="../models/keras_hyperband_disaster_prediction.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)
# Tensorboard callback
tensorboard_cb = keras.callbacks.TensorBoard(
    os.path.join(tensorboard_logs_path, time.strftime("run_%Y_%m_%d_at_%H_%M_%S"))
)


# # Type change for passing it in Tuner
# * This was a bug where a numpy array couldn't be converted to a tenson before passing it into the model, for more reference check out [link](https://stackoverflow.com/a/60750937/10016132)

bert_xtrain = np.asarray(bert_xtrain).astype(np.float32)
bert_xval = np.asarray(bert_xval).astype(np.float32)


# # Training! Best Validation Accuracy is 83.46%

if not already_trained:
    tuner.search(
        {
            "input_word_ids": bert_xtrain[0],
            "input_mask": bert_xtrain[1],
            "segment_ids": bert_xtrain[2],
        },
        y_train,
        epochs=3,
        batch_size=32,
        validation_data=(
            {
                "input_word_ids": bert_xval[0],
                "input_mask": bert_xval[1],
                "segment_ids": bert_xval[2],
            },
            y_val,
        ),
        callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb],
    )
else:
    None


# # Load Saved Model

bert_model = keras.models.load_model(
    "../models/keras_hyperband_disaster_prediction.h5",
    custom_objects={"KerasLayer": hub.KerasLayer},
)
bert_model.summary()


# # My Custom BERT Model Image

tf.keras.utils.plot_model(bert_model, to_file="bert_model.png", dpi=96)


# # Test set preprocessing

# Read Test Set
df_test = pd.read_csv(test_file)
# Preprocess Test Set
clean_text_column("text", df_test)
bert_test = bert_encode(df_test.clean_text)


# # Compute Predictions

pred = bert_model.predict(bert_test)


# # Submission

round_predictions = pred.round().astype("int")
df_submit = pd.read_csv(sample_submission_file)
df_submit["target"] = round_predictions
df_submit.to_csv("submission.csv", index=False, header=True)
