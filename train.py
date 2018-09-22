import numpy as np
import model
import time
import re
import gensim
import gensim.models.keyedvectors as word2vec
from keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt

print('Loading Word Embedding model...')

start_time = time.time()
embedding = word2vec.KeyedVectors.load_word2vec_format('wiki.ar.vec', binary=False)
# embedding = gensim.models.Word2Vec.load('tweet_cbow_300/tweets_cbow_300')
load_time = time.time() - start_time

print('loaded model in ' + str(load_time) + ' seconds')

dump_chars = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~’،ـ؟؛«» '

def clean_word(word):
    word = word.translate(str.maketrans({key: None for key in dump_chars}))
    
    #remove tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    word = re.sub(p_tashkeel,"", word)
    
    return word

def load_data():
    path = 'ANERCorp'
    f = open(path, 'r')
    sents = f.read().split('\n. O\n')
    f.close()
    
    # tokenize words
    words = [None]*len(sents)
    tokens = [None]*len(sents)
    for i, sent in enumerate(sents):
        sent = sent.split('\n')
        words[i] = []
        tokens[i] = []
        for word in sent:
            line = word.rsplit(' ', 1)
            line[0] = clean_word(line[0])
            if len(line[0]) > 0:
                words[i].append(line[0])
                tokens[i].append(line[1])
                    
                
    return [d for d in words if len(d) > 0], [d for d in tokens if len(d) > 0]

# load data
sents, labels = load_data()


tag_classes = ['B-LOC', 'B-MISC', 'B-ORG', 'B-PERS', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PERS', 'O']

# embed words
for i, sent in enumerate(sents):
    for j, word in enumerate(sent):
        try:
            sents[i][j] = embedding[word]
        except KeyError:
            sents[i][j] = embedding['unk']

# embed labels
for i, tokens in enumerate(labels):
    labels[i] = [to_categorical(tag_classes.index(tag), num_classes=len(tag_classes)) for tag in tokens]
        
################################
# No. sentences: 4898
# No. all words: 135717
# No. 3/4 all words: 101787
# Index of 3/4 sentences: 3569
################################

# pad sequences
for i, sent in enumerate(sents):
    l = 212 - len(sent)
    sents[i] += [[0]*300]*l
    
for i, label in enumerate(labels):
    l = 212 - len(label)
    labels[i] += [[0]*8+[1]]*l
    
# build model
train_model, crf_layer = model.build_model()
train_model.compile(optimizer="rmsprop", loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])
train_model.summary()

# train model
history = train_model.fit(np.array(sents, dtype='float64'), np.array(labels, dtype='float64'), epochs=10, verbose=1, validation_split=0.25)

# save weights
train_model.save_weights('weights.hd5f')

# plot accuracy
hist = pd.DataFrame(history.history)
plt.style.use("ggplot")
plt.figure(figsize=(24,24))
plt.plot(hist["acc"])
plt.plot(hist["val_acc"])
plt.show()
