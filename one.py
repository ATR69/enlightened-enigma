import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras.preprocessing.text import one_hot


path = "in1.txt"
z=0
rawtxt = open(path).read().lower()
sentences = rawtxt.split('\n')

chars = sorted(list(set(rawtxt)))
vocab = len(chars)

char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

print"Unique Chars: ", vocab
print "Raw Data: ", len(rawtxt)

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

xx = np.array(datax)
yy = np.array(datay)

print xx.shape, ' -> ', yy.shape, '\n'

x = to_categorical(xx)
y = to_categorical(yy)

print x.shape, '->', y,shape