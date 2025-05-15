import streamlit as st
import json
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data once
nltk.download('punkt_tab')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load intents
with open('intents.json') as file:
    intents = json.load(file)

# Prepare data (you can move this to a function if you want)
all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

all_words = [lemmatizer.lemmatize(w.lower()) for w in all_words if w.isalnum()]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

def bag_of_words(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [0] * len(all_words)
    for s in sentence_words:
        for i, w in enumerate(all_words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Streamlit UI

st.title("ChatBot - Chatbot")

user_input = st.text_input("You:")

if user_input:
    bow = bag_of_words(user_input)

    best_tag = None
    for (pattern_words, tag) in xy:
        pattern_bow = bag_of_words(" ".join(pattern_words))
        if np.array_equal(bow, pattern_bow):
            best_tag = tag
            break

    if best_tag:
        for intent in intents['intents']:
            if intent['tag'] == best_tag:
                response = random.choice(intent['responses'])
                st.text(f"ChatBot: {response}")
                break
    else:
        st.text("ChatBot: I'm not sure I understand.")
