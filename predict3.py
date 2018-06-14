import sys
import numpy
from keras.models import Sequential
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

sequence_len = 40

def get_sequence(rawtxt, chars, sequence_len):

	step = 3
	datax = []
	datay = []

	for i in range( 0, len(rawtxt) - sequence_len - 1, step):
		
		#seq_in = rawtxt[i : i + sequence_len]
		#seq_out = rawtxt[i+1 : i + sequence_len+1]
		datax.append(rawtxt[i : i + sequence_len])
		datay.append(rawtxt[i+1 : i + sequence_len+1])

	#n_patterns = len(datax)
	
	#print ("Total Pattern : ", n_patterns)

x = np.zeros((len(datax), sequence_len, len(chars)), dtype=np.bool)
y = np.zeros((len(datax), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_int[char]] = 1
    y[i, char_to_int[datay[i]] = 1

	return x, y

x, y = get_sequence(rawtxt, chars, sequence_len)
print (x.shape, '\n', y.shape)
model = Sequential()
model.add(Bidirectional(LSTM(256, return_sequences = True), input_shape = (x.shape[1], x.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(len(chars),activation = 'softmax'))
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

filepath = "wt-imp1.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
callbacks_list = [checkpoint]

for epoch in range(10):

	x, y = get_sequence(rawtxt, chars, sequence_len)
	history = model.fit(x, y, epochs = 1, validation_split = 0.05,  batch_size = 500, callbacks = callbacks_list).history
	#model.fit(x, y, epochs = 1, validation_split = 0.05,  batch_size = 20, callbacks = callbacks_list)

model.save('keras_model1.h5')
pickle.dump(history, open("history1.p", "wb"))
