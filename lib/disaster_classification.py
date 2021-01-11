#!/usr/bin/env python
# coding: utf-8

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from matplotlib import pylab
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from kerastuner.tuners import Hyperband
from sklearn.model_selection import train_test_split
import time

# Local imports
import lib.tokenization as tokenization
import lib.preprocess as preprocess
import lib.bert_model as bert_model

# Path to Train and Test Files
curr_dir = ".."
already_trained = True  # set this to false to retrain.
train_file = os.path.join(curr_dir, "dataset", "train.csv")
test_file = os.path.join(curr_dir, "dataset", "test.csv")
sample_submission_file = os.path.join(curr_dir, "dataset", "sample_submission.csv")

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


def analyticGraphs():
    # Tweet Locations Graph
    locations_map = df.location.value_counts().sort_values(ascending=False)[:10]
    plotBarGraph(locations_map, "Frequency", "Countries", "Tweet locations")

    # Top 20 Disaster-Related Keywords
    keywords_map = (
        df.keyword.loc[df.target == 1].value_counts().sort_values(ascending=False)[:20]
    )
    plotBarGraph(
        keywords_map, "Frequency", "Keywords", "Top 20 disaster related keywords"
    )

    # Top 20 Non-Disaster related Keywords
    """
    Notice non-disaster keywords are not actual disasters but rather general keywords that could describe a disaster. 
    These keywords can be used in our daily lives, for instance "I started panicing when I got into MIT", which clearly isn't a natural disaster lol :)
    """
    keywords_non_disaster_map = df.keyword.loc[df.target == 0].value_counts(
        ascending=False
    )[:20]
    plotBarGraph(
        keywords_non_disaster_map,
        "Frequency",
        "Keywords",
        "Top 20 non-disaster related keywords",
    )
    # Target Distrubution
    sns.countplot(x="target", data=df)
    plt.show()

    # Duplicate Text Data Distrubution
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

# Preprocess dataframe text column
def clean_text_column(col_name, df=df):
    df["clean_text"] = ""
    df["clean_text"] = df[col_name].apply(preprocess.clean_sentence_pipeline)


# Data Splits
clean_text_column("text")
x_train, x_val, y_train, y_val = train_test_split(
    df.clean_text, df.target, train_size=0.85
)

# BERT Transform the Data
def bert_transform_for_train():
    # BERT Encoding Calls on Train and Validation
    bert_xtrain = bert_model.bert_encode(x_train)
    bert_xval = bert_model.bert_encode(x_val)

    # Type change for passing it in Tuner Reference: https://stackoverflow.com/a/60750937/10016132
    bert_xtrain = np.asarray(bert_xtrain).astype(np.float32)
    bert_xval = np.asarray(bert_xval).astype(np.float32)
    return bert_xtrain, bert_xval


# Training! Best Validation Accuracy is 83.46%
def train():
    bert_xtrain, bert_xval = bert_transform_for_train()
    tuner = Hyperband(
        bert_model.build_model,
        objective="val_accuracy",
        max_epochs=3,
        project_name="../hyperparameter/hyperband",
    )
    # Keras Callbacks
    model_path = os.path.join("..", "models", "keras_hyperband_disaster_prediction.h5")
    tensorboard_logs_path = os.path.join("..", "tensorboard_logs")
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=model_path, save_best_only=True
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True
    )
    tensorboard_cb = keras.callbacks.TensorBoard(
        os.path.join(tensorboard_logs_path, time.strftime("run_%Y_%m_%d_at_%H_%M_%S"))
    )
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


# Train neural network.
if not already_trained:
    train()
else:
    None


# # Load Saved Model

bert_model_existing = keras.models.load_model(
    "../models/keras_hyperband_disaster_prediction.h5",
    custom_objects={"KerasLayer": hub.KerasLayer},
)
bert_model_existing.summary()


# Visual Model Image
tf.keras.utils.plot_model(
    bert_model_existing, to_file="../images/my_generated_model.png", dpi=96
)


# Test set preprocessing

# Read Test Set
df_test = pd.read_csv(test_file)

# Preprocess Test Set
clean_text_column("text", df_test)
bert_test = bert_model.bert_encode(df_test.clean_text)

# Compute Predictions
test_pred = bert_model_existing.predict(bert_test)

# Prediction Submission
round_predictions = test_pred.round().astype("int")
df_submit = pd.read_csv(sample_submission_file)
df_submit["target"] = round_predictions
df_submit.to_csv("submission.csv", index=False, header=True)
