import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, GRU
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
import pickle
import heapq

path = "new_cup.txt"
rawtxt = open(path).read().lower()
datas = rawtxt.split('\n')

chars = sorted(list(set(rawtxt)))
vocab = len(chars)

char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

sentences = []
next_chars = []

for data in datas:

	if len(list(data)) > 30:

		#print data
		continue

	for i in range(0,len(data),3):

		sentences.append([char_to_int[char] for char in data[:i]])
		next_chars.append([char_to_int[data[i]]])

print('num training examples: ',len(sentences))

#print sentences.shape, '\n', next_chars.shape, '\n'
#print type(sentences)
#input()

x = tf.one_hot(sentences, depth = vocab)
y = tf.one_hot(next_chars, depth = vocab)

print x.shape, '\n', y.shape


model = Sequential()
model.add(GRU(128, input_shape=(sequence_len, vocab)))
#model.add(Dropout(0.2))
model.add(Dense(vocab, activation = 'softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
#model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

filepath = "wt-imp1.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
callbacks_list = [checkpoint]

history = model.fit(x, y, epochs = 1, validation_split = 0.1,  batch_size = 10000, callbacks = callbacks_list, shuffle=True).history

model.save('keras_model3.h5')
pickle.dump(history, open("history3.p", "wb"))