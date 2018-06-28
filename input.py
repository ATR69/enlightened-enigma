
f = open("i.txt", "w+")
path = "new_cup.txt"
rawtxt = open(path).read().lower()
sentences = rawtxt.split('\n')

print len(sentences)

for i in range(len(sentences)):

	if (len(sentences[i]) < 30):

		f.write(sentences[i]+'\n')

f.close()









			