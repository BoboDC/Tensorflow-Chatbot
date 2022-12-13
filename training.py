import json 
import numpy as np
import pandas as pd 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Input, LSTM, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model, save_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

# opening the json file that contains the data for the model
file = open("intents.json")
data = json.load(file)    
file.close()

# creating varibles that will store the data from the json
greetings = []
greetingLabels = []
tags = []
responses = []

# appending each list with the data that fits 
for intent in data['intents']:
    for pattern in intent['patterns']:
        greetings.append(pattern)
        greetingLabels.append(intent['tag'])
    responses.append(intent['responses'])
    
    if intent['tag'] not in tags:
        tags.append(intent['tag'])
              
classes = len(tags)

# storing the greetings and the labels in a pandas data frame that is easier to work with 
dataset = pd.DataFrame({"greetings": greetings, "tags": greetingLabels})

xData = dataset["greetings"]
yData = dataset["tags"]

# tokenizing the data
tokenizer = Tokenizer()
tokenizer = Tokenizer(num_words=2000)

# fitting on text the data from the pandas as well as making all lower case
tokenizer.fit_on_texts(xData.astype(str).str.lower())

# keeping all the sentences the same size
tokenSentence = tokenizer.texts_to_sequences(xData)
totalWords = len(tokenizer.word_index)+1
allWordsCount = tokenizer.word_counts
x = pad_sequences(tokenSentence)

# encoding the labels
labelEnc = LabelEncoder()
y = labelEnc.fit_transform(yData)
xShape = x.shape[1]

# model
model = Sequential()
model.add(Input(shape=(xShape,)))
model.add(Embedding(input_dim=totalWords,
                    output_dim=10))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
model.add(Flatten())
model.add(Dense(units=50, activation='relu'))
model.add(Dense(classes, activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy",
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=30)
mc = ModelCheckpoint('model.h5', monitor='accuracy', mode='max', verbose=1, save_best_only=True)

train = model.fit(x, y, epochs=400, callbacks=[es,mc])

