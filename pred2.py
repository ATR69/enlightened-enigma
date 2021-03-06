import numpy as np
from keras.models import load_model
import pickle
import sys
import heapq

np.random.seed(47)


model = load_model('keras_model2.h5')
history = pickle.load(open("history2.p", "rb"))

sequence_len = 40

path = "BGAE.txt"
rawtxt = open(path).read().lower()

chars = sorted(list(set(rawtxt)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

def prepare_input(text):

    x = np.zeros((1, sequence_len, len(chars)))
    for t, char in enumerate(text):
        x[0, t, char_to_int[char]] = 1.        
    return x
	

def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    return heapq.nlargest(top_n, range(len(preds)), preds.take)
	
	
def predict_completion(text):
    original_text = text
    #print text
    generated = text
    completion = ''
    while True:
        #print "Text : ", text
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
    #print next_indices
    r=[int_to_char[idx] + predict_completion(text[1:] + int_to_char[idx]) for idx in next_indices]
    return [int_to_char[idx] + predict_completion(text[1:] + int_to_char[idx]) for idx in next_indices]
	
quotes = [
    "It is not a lack of love, but a lack of friendship that makes unhappy marriages.",
    "That which does not kill us makes us stronger.",
    "I'm not upset that you lied to me, I'm upset that from now on I can't believe you.",
    "And those who were seen dancing were thought to be insane by those who could not hear the music.",
    "It is hard enough to remember my opinions, without also remembering my reasons for them!"
    ]


for q in quotes:
    seq = q[:40].lower()
    print(seq)
    print(predict_completions(seq, 5))
    print()

# q=sys.argv[1:]
# q=" ".join(q)
# #q="1 ch"
# seq = q[:40].lower()
# print seq
# print(predict_completions(seq, 5))
# print()
# res = predict_completions(seq, 5)
# ret=[]
# for i in res:
#     k=q+i
#     ret.append(k)
# print ret
