# Fine-Tuned BERT model to Classify Natural-Disaster-Tweets

This project is the part of a Kaggle NLP competition to classify wether a series of tweets related to a natural disaster or not. My results were in the top 25% category according to kaggle. (In reality, my position is more higher because submissions by bots have an unfeasible accuracy of 1.0 and get ranked high)

## Brief Summary
Using natural language processessing, I build a text pipeline to clean text by removing special characters, lemmatization, and extensively use the nltk package for preprocessing. Moreover, I encode the clean english text and use BERT encoding to make the text ready for the neural network. Finally, I feed the text into a custom built feed-forward model I build on top of a pretrained BERT (Bidirectional Encoder Text Representation) model made by Google. I also tune my custom built model using Keras Tuner and tune the dropout regularization rate, number neurons in the hidden layer, learning rate alpha.

## Model
The feed-forward model has 2 hidden layers with activation "ELU" and one classification layer with the activation of sigmoid. The model uses the ADAM optimization algorithm and uses 3 epochs to train the model(reccomended by Google Research team)

My model image generated from Keras plot utils:

![model_plot](./images/my_generated_model.png)

## Want to Try out?