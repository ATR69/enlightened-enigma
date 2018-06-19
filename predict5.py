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
print type(rawtxt)
input()

words = []
datas = rawtxt.split("\n")
for i in datas:
	words=i.split(' ')


print len(words)
print words
word_to_int = dict((c, i) for i, c in enumerate(vocab))
int_to_word = dict((i, c) for i, c in enumerate(vocab))

print "Unique Words: ", len(vocab)
print "No of words: ", len(words)
input()
sequence_len = 40

def get_sequence(datas):

	datax = []
	datay = []

	for data in datas:

		if len(list(data)) > sequence_len:

			continue

		for i in range(0,len(data),3):

			datax.append(data[:i])
			#print datax, '\n'
			datay.append(data[i])
			#print datay


	x = np_utils.to_categorical(datax)
	
	y = np_utils.to_categorical(datay)

	return x, y


#print('num training examples: ',len(sentences))


x, y = get_sequence(datas)
#print (x.shape, '\n', y.shape)
model = Sequential()
model.add(Bidirectional(LSTM(256, return_sequences = True), input_shape = (x.shape[1], x.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(len(chars),activation = 'softmax'))
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

filepath = "wt-imp.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
callbacks_list = [checkpoint]


for epoch in range(2):

	x, y = get_sequence(datas)
	history = model.fit(x, y, epochs = 1, validation_split = 0.05,  batch_size = 1000, callbacks = callbacks_list).history
	#model.fit(x, y, epochs = 1, validation_split = 0.05,  batch_size = 20, callbacks = callbacks_list)

model.save('keras_model.h5')
pickle.dump(history, open("history.p", "wb"))

model = load_model('keras_model.h5')
history = pickle.load(open("history.p", "rb"))
