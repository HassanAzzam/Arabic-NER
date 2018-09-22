
# coding: utf-8

# In[1]:


import numpy as np
import model
import time
import re
import gensim
import gensim.models.keyedvectors as word2vec

print('Loading Word Embedding model...')

start_time = time.time()
embedding = word2vec.KeyedVectors.load_word2vec_format('wiki.ar.vec', binary=False)
# embedding = gensim.models.Word2Vec.load('tweet_cbow_300/tweets_cbow_300')
load_time = time.time() - start_time

print('loaded model in ' + str(load_time) + ' seconds')


# In[23]:


dump_chars = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~’،ـ؟؛«» '

def clean_word(word):
    word = word.translate(str.maketrans({key: None for key in dump_chars}))
    
    #remove tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    word = re.sub(p_tashkeel,"", word)
    
    return word

raw_sent = input()
raw_sent = raw_sent.split()
sent = []
X = []
for i, word in enumerate(raw_sent):
    word = clean_word(word)
    if len(word) > 0:
        sent.append(word)
        try:
            X.append(embedding[word])
        except KeyError:
            X.append(embedding['unk'])

X += [[0]*300]*(212 - len(sent))
# print(len(X))


# In[24]:


test_model, _ = model.build_model()
test_model.load_weights('weights.hd5f')
pred = test_model.predict(np.array([X]))


# In[26]:


tag_classes = ['B-LOC', 'B-MISC', 'B-ORG', 'B-PERS', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PERS', 'O']
from terminaltables import AsciiTable
table_data = [['word', 'prediction']]
for i, word in enumerate(sent):
    table_data.append([word, tag_classes[np.argmax(pred[0][i])]])
table = AsciiTable(table_data)
print(table.table)


# In[8]:





# In[12]:





# In[16]:




