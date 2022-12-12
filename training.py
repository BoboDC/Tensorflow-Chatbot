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

file = open("intents.json")
data = json.load(file)    
file.close()
    
greetings = []
greetingLabels = []
tags = []
responses = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        greetings.append(pattern)
        greetingLabels.append(intent['tag'])
    responses.append(intent['responses'])
    
    if intent['tag'] not in tags:
        tags.append(intent['tag'])
        
classes = len(tags)
dataset = pd.DataFrame({"greetings": greetings, "tags": greetingLabels})

xData = dataset["greetings"]
yData = dataset["tags"]

tokenizer = Tokenizer()
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(xData.astype(str).str.lower())
tokenSentence = tokenizer.texts_to_sequences(xData)
totalWords = len(tokenizer.word_index)+1
allWordsCount = tokenizer.word_counts
x = pad_sequences(tokenSentence)

labelEnc = LabelEncoder()
y = labelEnc.fit_transform(yData)
input_shape = x.shape[1]


model = Sequential()
model.add(Input(shape=(input_shape,)))
model.add(Embedding(totalWords, 10))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(Flatten())
model.add(Dense(units=50, activation='relu'))
model.add(Dense(classes, activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
model.summary()

es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=30)
mc = ModelCheckpoint('model.h5', monitor='accuracy', mode='max', verbose=1, save_best_only=True)

train = model.fit(x, y, epochs=400, callbacks=[es,mc])

