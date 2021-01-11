from tensorflow import keras
import tensorflow_hub as hub
from lib.preprocess import clean_sentence_pipeline

# bert_model = keras.models.load_model(
#     "./models/keras_hyperband_disaster_prediction.h5",
#     custom_objects={"KerasLayer": hub.KerasLayer},
# )
# print("\n------------------------------------------------------- BERT Model -------------------------------------------------------\n")
# bert_model.summary()
# print("\n------------------------------------------------------- End of Model Summary -------------------------------------------------------\n")

input_sentence="Hello!3(02"

# Preprocessing
input_sentence = clean_sentence_pipeline(input_sentence)
print(input_sentence)