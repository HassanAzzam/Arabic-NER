import numpy as np
import tensorflow as tf
import model
import time
import re
import gensim
import gensim.models.keyedvectors as word2vec
from keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

# from keras.callbacks import Callback
# # from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
# class Metrics(Callback):
#     def on_train_begin(self, logs={}):
#         self.val_f1s = []

#     def on_epoch_end(self, epoch, logs={}):
#         f_score = 2*logs['val_precision']*logs['val_recall']/(logs['val_precision']+logs['val_recall'])
#         self.val_f1s.append(f_score)
#         print("f_score: {0}".format(f_score))
#         return

# metrics = Metrics()

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
max_sent_length = 212
sents_lengths = []
for i, sent in enumerate(sents):
    sents_lengths.append(len(sent))
    l = max_sent_length - len(sent)
    sents[i] += [[0]*300]*l
    
for i, label in enumerate(labels):
    l = max_sent_length - len(label)
    labels[i] += [[0]*8+[0]]*l
    
    
# split data
train_x, train_y = sents[:3673], labels[:3673]
test_x, test_y = sents[3674:], labels[3674:]

# tf metrics wrapper
def as_keras_metric(method):
    import functools
    from keras import backend as K
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

# build model
train_model, crf_layer = model.build_model()
train_model.compile(optimizer="rmsprop", loss=crf_layer.loss_function, metrics=[crf_layer.accuracy, as_keras_metric(tf.metrics.precision), as_keras_metric(tf.metrics.recall), as_keras_metric(tf.contrib.metrics.f1_score)])
train_model.summary()

# train model
history = train_model.fit(np.array(train_x, dtype='float64'), np.array(train_y, dtype='float64'), epochs=20, verbose=1, validation_data=(np.array(test_x, dtype='float64'), np.array(test_y, dtype='float64')))

# save weights
train_model.save_weights('weights.hd5f')

# plot accuracy
hist = pd.DataFrame(history.history)
plt.style.use("ggplot")
plt.figure(figsize=(12,12))
plt.plot(hist["val_f1_score"])
plt.plot(hist["val_acc"])
plt.plot(hist["val_loss"])
plt.show()


# testing
pred = train_model.predict(np.array(test_x, dtype='float64'))
pred_x = []
pred_y = []
for i, sent in enumerate(pred):
    pred_x.append([tag_classes[np.argmax(w)] for w in pred[i][:sents_lengths[i]]])
    pred_y.append([tag_classes[np.argmax(w)] for w in test_y[i][:sents_lengths[i]]])
    
print(classification_report(pred_y, pred_x))
print('f1_score: ')
print(f1_score(pred_y, pred_x))
