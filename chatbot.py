import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('tempFiles\\words.pkl', 'rb'))
classes = pickle.load(open('tempFiles\\classes.pkl', 'rb'))
model = load_model('tempFiles\\chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    # prepare sentence
    sentence_words = clean_up_sentence(sentence)
    # create a bag of all known words
    bag = [0] * len(words)
    # if the current word matches any known word, set the position of that word to 1
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    # prepare the sentence and let the model predict the class
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    # remove all predictions below Threshold and add the number of the position so we can assign the class
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort in descending order depending on r[1] (the result probability)
    results.sort(key=lambda x: x[1], reverse=True)
    # cast result into an array of objects that we can use. we also switch back to see the classes in full text
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intent_list, intent_json):
    # set the most likely intent as tag
    tag = intent_list[0]['intent']
    list_of_intents = intent_json['intents']
    # if probability of the most likely intent is below 50% we set it to repeatPlease
    if float(intent_list[0]['probability']) < 0.5:
        tag = 'repeatPlease'
    # get a random response of the selected tag
    result = {'response': '', 'action': 0}
    for i in list_of_intents:
        if i['tag'] == tag:
            result['response'] = random.choice(i['responses'])
            break
    # set an action flag if required
    if tag == 'goodbye':
        result['action'] = 1
    return result


def run_chatbot():
    running = True
    print('Hi, I am the Allu. You can aks me anything about YMCA. Please ask a question.')
    while running:
        message = input("")
        ints = predict_class(message)
        res = get_response(ints, intents)
        print(res['response'])
        if res['action'] == 1:
            running = False
