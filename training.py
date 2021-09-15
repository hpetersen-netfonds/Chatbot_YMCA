import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD


def train():
    lemmatizer = WordNetLemmatizer()
    intents = json.loads(open('intents.json').read())
    words = []
    classes = []
    documents = []
    ignore_letters = ['?', '!', '.', ',', ';', ':']

    # from all patterns extract all known classes and words. (also prepare documents[all patterns with their intent])
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # lemmatize (break down words to their core)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]

    # remove duplicates and turn back into a list
    words = sorted(set(words))
    classes = sorted(set(classes))
    # save words and classes in Files
    pickle.dump(words, open('tempFiles\\words.pkl', 'wb'))
    pickle.dump(classes, open('tempFiles\\classes.pkl', 'wb'))

    training = []
    output_empty = [0] * len(classes)

    # for every entry in documents we create a bag of words
    # and add it together with the right class into the training array
    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])

    # we shuffle the data and cast it into an array
    random.shuffle(training)
    training = np.array(training)
    # we then split the array into an x and y axis. x are the bag of words and y are the corresponding intent classes
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    # tensorflow black magic
    # setting up the model
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # assign a optimizer
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # compile
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # train the model with our training data over 200 epochs
    chat_model = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
    # save the model for later use
    model.save('tempFiles\\chatbot_model.h5', chat_model)
    print('Done Training')