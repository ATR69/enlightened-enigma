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


np.random.seed(47)

model = load_model('keras_model1.h5')
history = pickle.load(open("history1.p", "rb"))

sequence_len = 30
#datax = []
#seq_in = [30]

t = "cup"
#text = ''.join(t)

path = "new_cup.txt"
rawtxt = open(path).read().lower()

chars = sorted(list(set(rawtxt)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

def prepare_input(text):
    x = np.zeros((1, SEQUENCE_LENGTH, len(char_indices)))
    for t, char in enumerate(text):
        x[0, t, char_indices[char]] = 1.        
    return x
	

def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    return heapq.nlargest(top_n, range(len(preds)), preds.take)
	
	
def predict_completion(text):
    original_text = text
    generated = text
    completion = ''
    while True:
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, top_n=1)[0]
        next_char = indices_char[next_index]
        text = text[1:] + next_char
        completion += next_char
        
        if len(original_text + completion) + 2 > len(original_text) and (next_char == ' ' or next_char == '\n'):
            return completion
			
def predict_completions(text, n=3):
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    r=[indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]
    return [indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]
	
res = print(predict_completions(t, 3))

ret=[]

for i in res:
    k=q+i
    ret.append(k)
    
print ret
