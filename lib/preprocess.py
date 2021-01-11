import warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.corpus import stopwords
import emoji
import contractions
from bs4 import BeautifulSoup
import unicodedata
from nltk.stem import WordNetLemmatizer
import re

# nltk.download("wordnet")


# Lower case text data
def lower_case_data(data=""):
    data = data.lower()
    return data


# Handle Emojis
def sentences_with_emojis(id_texts):
    sentences = []
    indeces = id_texts[0]
    texts = id_texts[1]
    for index, sentence in zip(indeces, texts):
        has_emoji = bool(emoji.get_emoji_regexp().search(sentence))
        if has_emoji:
            sentences.append((index, sentence))
    if len(sentences) == 0:
        return "Sentences are clean and don't have emojis!"
    else:
        return sentences


# Source: https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
def clean_emojis(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = " ".join(
        [str for str in text.split() if not any(i in str for i in emoji_list)]
    )
    return clean_text


# Clean urls
def clean_urls(text):
    text = re.sub(r"https?://\S+", "", text)
    return text


# Remove all sorts of special characters and punctuations.
def removeSpecialChar(text):
    sentence = []
    for s in text:
        if s == " ":
            sentence.append(s)
        if s.isalnum():
            sentence.append(s)
    return "".join(sentence)


# Check any html text
def checkHtml(text):
    return bool(BeautifulSoup(text, "html.parser").find())


# Remove accented text
def remove_accented_chars(text):
    new_text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )
    return new_text


# Explore Stopwords list from nltk
def remove_words_having_not(words=[""]):
    for word in words:
        if "not" in word:
            words.remove(word)
    return words


# Custom stopwords list
def stopwords_list():
    stopwords_list = stopwords.words("english")
    stopwords_list = remove_words_having_not(
        [removeSpecialChar(contractions.fix(word)) for word in stopwords_list]
    )
    return stopwords_list


# Remove stopwords from nltk corpus
def remove_stopwords(text):
    new_sentence = ""
    stop_words = stopwords_list()
    for word in text.split():
        if word not in stop_words:
            new_sentence += word + " "
    return new_sentence


# Remove numbers from text
def remove_numbers(text=""):
    new_sentence = ""
    for word in text.split():
        num_free_word = "".join([i for i in word if not i.isdigit()])
        new_sentence += num_free_word + " "
    return new_sentence


# lemmatization
def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    new_sentence = ""
    for word in text.split():
        lematized_word = lemmatizer.lemmatize(word)
        new_sentence += lematized_word + " "
    return new_sentence


# Final data cleaning step is removing non-essential whitespaces
def remove_white_space(text):
    return " ".join(text.split())


# Text cleaning pipeline
def clean_sentence_pipeline(text):
    text = lower_case_data(text)
    text = clean_emojis(text)
    text = clean_urls(text)
    text = contractions.fix(text)
    text = removeSpecialChar(text)
    text = remove_accented_chars(text)
    text = remove_stopwords(text)
    text = remove_numbers(text)
    text = lemmatize(text)
    return remove_white_space(text)