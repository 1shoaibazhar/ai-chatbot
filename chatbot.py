import random
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model('chatbot_model.keras')

def cleanUpSentence(sentence):
    sentenceWords = nltk.word_tokenize(sentence)
    sentenceWords = [lemmatizer.lemmatize(word.lower()) for word in sentenceWords]
    return sentenceWords


def bagOfWords(sentence):
    sentenceWords = cleanUpSentence(sentence)
    bag = [0] * len(words)
    for w in sentenceWords:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predictClass(sentence):
    bow = bagOfWords(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    returnList = []
    for result in results:
        returnList.append({'intent': classes[result[0]], 'probability': str(result[1])})
    return returnList


while True:
    message = input('You: ')
    if message == 'exit':
        break
    intentsList = predictClass(message)
    for intent in intentsList:
        if intent['probability'] != '0':
            for i in intents['intents']:
                if i['tag'] == intent['intent']:
                    print(f'Chatbot: {random.choice(i["responses"])}')
    if intentsList[0]['probability'] == '0':
        print('Chatbot: I do not understand')

        