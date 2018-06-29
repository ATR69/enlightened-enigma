import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Embedding, GRU, CuDNNGRU
from keras.layers import TimeDistributed
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras.preprocessing.text import one_hot
import pickle
import heapq

np.random.seed(47)
path = "i.txt"
rawtxt = open(path).read().lower()
sentences = rawtxt.split('\n')

#print sentences[4][5],'\n', len(sentences[4])


chars = sorted(list(set(rawtxt)))
vocab = len(chars)

char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

print"Unique Chars: ", vocab
#print "Raw Data: ", len(rawtxt)

sequence_len = 30

def get_sequence(rawtxt, chars, sequence_len):

	step = 2
	datax = []
	datay = []

	for i in range(len(sentences)):

		for j in range(0, len(sentences[i]), step):

			seq_in = sentences[i][:j]
			seq_out = sentences[i][j]
			datax.append([char_to_int[char] for char in seq_in])
			datay.append([char_to_int[seq_out]])

	n_patterns = len(datax)

	# x = np_utils.to_categorical(datax)
	# y = np_utils.to_categorical(datay)

		
	x = np.zeros((n_patterns, sequence_len, len(chars)), dtype=np.bool)

	y = np.zeros((n_patterns, len(chars)), dtype=np.bool)

	for i, sentence in enumerate(datax):

		for t, word in enumerate(sentence):

			x[i, t, word] = 1

	for i, w in enumerate(datay):

		#print w
		y[i, w] = 1



	return x, y, n_patterns

x, y, n = get_sequence(rawtxt, chars, sequence_len)
#print (x.shape, '\n', y.shape)

model = Sequential()
#model.add(CuDNNGRU(256, input_shape=(sequence_len, vocab)))
#model.add(Embedding(sequence_len, vocab, input_shape=(n, sequence_len)))
#model.add(Bidirectional(LSTM(256, return_sequences = True), input_shape=(sequence_len, vocab)))
#model.add(Dropout(0.2))
#model.add(LSTM(128))
#model.add(TimeDistributed(Dense(vocab, activation = 'softmax')))
model.add(GRU(128, input_shape=(sequence_len, vocab)))
model.add(Dropout(0.2))
# model.add(GRU(64))
# model.add(Dropout(0.2))
model.add(Dense(vocab, activation = 'softmax'))
model.summary()
# optimizer = RMSprop(lr=0.01)
# model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
model.compile(loss = 'categorical_crossentropy', optimizer = 'adagrad', metrics = ['accuracy'])

filepath = "wt-imp2.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
callbacks_list = [checkpoint]

x, y, n = get_sequence(rawtxt, chars, sequence_len)

history = model.fit(x, y, epochs = 5, validation_split = 0.1,  batch_size = 10000, callbacks = callbacks_list, shuffle=True).history

# for epoch in range(2):

# 	x, y, n = get_sequence(rawtxt, chars, sequence_len)
# 	history = model.fit(x, y, epochs = 1, validation_split = 0.1,  batch_size = 10000, callbacks = callbacks_list, shuffle=True).history
# 	#model.fit(x, y, epochs = 1, validation_split = 0.05,  batch_size = 20, callbacks = callbacks_list)

model.save('keras_model4.h5')
pickle.dump(history, open("history4.p", "wb"))
