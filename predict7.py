import sys
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.layers import TimeDistributed
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop
from keras.preprocessing.text import one_hot
import pickle
import heapq

numpy.random.seed(47)
path = "new_cup.txt"
rawtxt = open(path).read().lower()

chars = sorted(list(set(rawtxt)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((c, i) for i, c in enumerate(chars))

print"Unique Chars: ", len(chars)
print "Raw Data: ", len(rawtxt)

sequence_len = 30

def get_sequence(rawtxt, chars, sequence_len):

	#step = 1
	datax = []
	datay = []

	for data in rawtxt:
		
		if data != "\n":
			
			for i in range (0, len(data), 1):

				seq_in = data[:i]
				#print seq_in
				seq_out = data[i]
				datax.append([char_to_int[char] for char in seq_in])
				datay.append([char_to_int[char] for char in seq_out])
				

	n_patterns = len(datax)
	print ("Total Pattern : ", n_patterns)

	x = np.zeros((n_patterns, sequence_len, char), dtype=np.bool)
	y = np.zeros((n_patterns, chars), dtype=np.bool)

	for i, sentence in enumerate(datax):
		for t, word in enumerate(sentence):
			x[i, t, chars[word]] = 1
	y[i, chars[datay[i]]] = 1


	#x = np_utils.to_categorical(datax)


	#y = np_utils.to_categorical(datay)

	return x, y

x, y = get_sequence(rawtxt, chars, sequence_len)
print (x.shape, '\n', y.shape)
model = Sequential()
model.add(Bidirectional(LSTM(256, return_sequences = True, activation = 'relu'), input_shape = (x.shape[1], x.shape[2])))
model.add(Dropout(0.4))
model.add(Dense(len(chars),activation = 'softmax'))
model.summary()
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

filepath = "wt-imp1.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
callbacks_list = [checkpoint]

for epoch in range(2):

	x, y = get_sequence(rawtxt, chars, sequence_len)
	history = model.fit(x, y, epochs = 1, validation_split = 0.1,  batch_size = 1500, callbacks = callbacks_list, shuffle=True).history
	#model.fit(x, y, epochs = 1, validation_split = 0.05,  batch_size = 20, callbacks = callbacks_list)

model.save('keras_model1.h5')
pickle.dump(history, open("history1.p", "wb"))
