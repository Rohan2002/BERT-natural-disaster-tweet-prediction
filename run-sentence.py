import warnings
warnings.filterwarnings("ignore")

from tensorflow import keras
import tensorflow_hub as hub
from lib.preprocess import clean_sentence_pipeline
from lib.bert_model import bert_encode

input_sentence = input("Type in a natural disaster-tweet:")

# TODO: create caching mechanism for optimizing model runtime
bert_model_saved = keras.models.load_model(
    "./models/keras_hyperband_disaster_prediction.h5",
    custom_objects={"KerasLayer": hub.KerasLayer},
)
print(
    "\n------------------------------------------------------- BERT Model -------------------------------------------------------\n"
)
bert_model_saved.summary()
print(
    "\n------------------------------------------------------- End of Model Summary -------------------------------------------------------\n"
)

# Preprocessing
input_sentence_preprocess = clean_sentence_pipeline(input_sentence)
bert_encoded_sentence = bert_encode([input_sentence_preprocess])

# Predictions
pred = bert_model_saved.predict(bert_encoded_sentence)
# Probability
formatted_prob = round(pred[0][0] * 100, 0)
print("\n--------------------------------------------Result Start--------------------------------------------\n")
print(f"The tweet's \"{input_sentence}\" relation to a disaster is {formatted_prob}%")
print("\n--------------------------------------------Result End--------------------------------------------\n")