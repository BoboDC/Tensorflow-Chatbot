import random
import json 
import numpy as np
import pandas as pd 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model, save_model

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
xShape = x.shape[1]

model = load_model("model.h5")
# t = "hello man"
def chatting():
    while True:
        # print("User:", end=" ")
        t = input("User: ")
        tokenizedText = tokenizer.texts_to_sequences([t])[0]
        tokenizedText = np.array(tokenizedText).reshape(-1)
        tokenizedText = pad_sequences([tokenizedText],xShape)

        prediction = model.predict(tokenizedText, verbose=0)
        prediction = prediction.argmax()
        responseClass = labelEnc.inverse_transform([prediction])[0]

        for i in data["intents"]: 
            if i["tag"] == responseClass:
                print("AI: ", random.choice(i["responses"]))
                
    
chatting()
