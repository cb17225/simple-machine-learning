import pandas as pd
import numpy as np

# textFile = open(<filename>, "r")
lines = textFile.read().split("\n")
#textFile = open(<filename>, "r")
lines = lines + textFile.read().split("\n")
# textFile = open(<filename>, "r")
lines = lines + textFile.read().split("\n")

newLines = []
for line in lines:
  if len(line.split("\t")) == 2 and line.split("\t")[1] != "":
    newLines.append(line.split("\t"))

sentences = [line[0] for line in newLines]
labels = [int(line[1]) for line in newLines]

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')

trainingSentences = sentences[:750]
testingSentences = sentences[750:]
trainingLabels = labels[:750]
testingLabels = labels[750:]

tokenizer = Tokenizer(num_words = 10000, oov_token = '<OOV>')

tokenizer.fit_on_texts(trainingSentences)
wordIndex = tokenizer.word_index

trainingSequences = tokenizer.texts_to_sequences(trainingSentences)
trainingPadded = pad_sequences(trainingSequences, maxlen = 100, padding = 'post', truncating = 'post')

tokenizer.fit_on_texts(trainingSentences)
wordIndex = tokenizer.word_index

testingSequences = tokenizer.texts_to_sequences(testingSentences)
testingPadded = pad_sequences(testingSequences, maxlen = 100, padding = 'post', truncating = 'post')

import numpy as np

trainingPadded = np.array(trainingPadded)
trainingLabels = np.array(trainingLabels)

testingPadded = np.array(testingPadded)
testingLabels = np.array(testingLabels)

model = tf.keras.Sequential([
                             tf.keras.layers.Embedding(10000,16, input_length = 100),
                             tf.keras.layers.GlobalAveragePooling1D(),
                             tf.keras.layers.Dense(36, activation = 'relu'),
                             tf.keras.layers.Dense(1,activation = 'sigmoid')
])
model.summary()

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(
    trainingPadded,
    trainingLabels,
    epochs = 30
)