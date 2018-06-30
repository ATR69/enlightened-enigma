import numpy as np
from keras.models import load_model
import pickle
import sys
import heapq

np.random.seed(47)


model = load_model('keras_model4.h5')
history = pickle.load(open("history4.p", "rb"))

sequence_len = 17

path = "in.txt"
rawtxt = open(path).read().lower()

chars = sorted(list(set(rawtxt)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

def prepare_input(text):

    x = np.zeros((1, sequence_len, len(chars)))
    for t, char in enumerate(text):
        x[0, t, char_to_int[char]] = 1.        
    return x
	

def sample(preds, top_n=2):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    return heapq.nlargest(top_n, range(len(preds)), preds.take)
	
	
def predict_completion(text):
    original_text = text
    print text
    generated = text
    completion = ''
    while True:
        print "Text : ", text
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        #print "Pred: ", preds
        next_index = sample(preds, top_n=1)[0]
        next_char = int_to_char[next_index]
        #print "Next Char: ", next_char
        text = text[1:] + next_char
        completion += next_char
        #print len(original_text), '-', len(completion), '-', len(original_text+completion)

        if len(original_text + completion) + 2 > len(original_text) and (next_char == ' ' or next_char == '\n'):
            return completion
			
def predict_completions(text, n=3):
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    print next_indices
    r=[int_to_char[idx] + predict_completion(text[1:] + int_to_char[idx]) for idx in next_indices]
    return [int_to_char[idx] + predict_completion(text[1:] + int_to_char[idx]) for idx in next_indices]
	
q=sys.argv[1:]
q=" ".join(q)
#q="1 ch"
seq = q.lower()
print seq
res = predict_completions(seq, 3)
ret=[]
for i in res:
    k=q+i
    ret.append(k)
print ret
