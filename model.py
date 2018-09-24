
# coding: utf-8

# In[21]:


import keras
from keras.layers import Dense, Input, GRU, Embedding, Dropout, Activation, Masking
from keras.layers import Bidirectional, GlobalMaxPool1D, TimeDistributed
from keras.models import Model, Sequential
from keras_contrib.layers import CRF



def build_model():
    crf_layer = CRF(9)
    input_layer = Input(shape=(None,300,))
#     embedding = Embedding(212, 20, input_length=None, mask_zero=True)(input_layer)
    mask_layer = Masking(mask_value=0., input_shape=(212, 300))(input_layer)
    bi_gru = Bidirectional(GRU(10, return_sequences=True))(mask_layer)
    bi_gru = TimeDistributed(Dense(10, activation="relu"))(bi_gru)
    output_layer = crf_layer(bi_gru)
    return Model(input_layer, output_layer), crf_layer


# In[22]:


# compile model
# def compile_model(model):
#     model.compile(optimizer="rmsprop", loss=crf_layer.loss_function, metrics=[crf_layer.accuracy], validation_split=0.25)
#     model.summary()
#     return model

# build_model().summary()

