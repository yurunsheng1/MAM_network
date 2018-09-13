"""
p instapip install
This file implements the convolutional neural network to train,
evaluate, and make inference prediction, generation.
"""

import os
import numpy as np
from gensim.models import Word2Vec
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Concatenate, Conv1D, Dense, Dropout, Embedding, Input, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from utils import read_data_set, AA_IDX, global_average_pooling

def train(params, dir_names):
    """
        Trains the HLA-CNN model,the input is [batch_size, peptide, vector]
    """
    data_set, peptide_n_mer = read_data_set(dir_names)
    # CNN parameters
    batch_size = int(
        np.ceil(len(data_set['X_train']) / 100.0))  # variable batch size depending on number of data points
    epochs = int(params['epochs'])
    nb_filter = int(params['filter_size'])
    filter_length = int(params['filter_length'])
    dropout = float(params['dropout'])
    lr = float(params['lr'])

    # load in learned distributed representation HLA-Vec
    hla_vec_obj = Word2Vec.load(os.path.join(dir_names['Vec_embedding'], 'Vec_Object'))
    hla_vec_embed = hla_vec_obj.wv
    embed_shape = hla_vec_embed.syn0.shape
    embedding_weights = np.random.rand(embed_shape[0] + 1, embed_shape[1])
    for key in AA_IDX.keys():
        embedding_weights[AA_IDX[key], :] = hla_vec_embed[key]
        embedded_dim = embed_shape[1]

    i = 0
    while True:
        input_data = Input(shape=(None,))
        embedded = Embedding(input_dim=len(AA_IDX) + 1, output_dim=embedded_dim, weights=[embedding_weights],
                             input_length=peptide_n_mer, trainable=True, name='embedded')(input_data)
        conv1 = Conv1D(int(nb_filter / 2), filter_length, padding='same', kernel_initializer='glorot_normal',
                       name="conv1", kernel_regularizer=regularizers.l1(0.01))(embedded)
        relu1 = LeakyReLU(.3)(conv1)
        dropout1 = Dropout(dropout)(relu1)

        conv_out = Conv1D(int(nb_filter), filter_length, padding='same',
                          kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l1(0.01),
                          name="conv_out")(dropout1)
        relu2 = LeakyReLU(.3)(conv_out)

        # modify
        gap1 = Lambda(global_average_pooling, name="gap1")(relu1)
        gap2 = Lambda(global_average_pooling, name="gap2")(relu2)
        weight1 = Dense(1, use_bias=False, kernel_constraint=None, name="weight1")(gap1)
        weight2 = Dense(1, use_bias=False, kernel_constraint=None, name="weight2")(gap2)
        merge = Concatenate(axis=1)([weight1, weight2])
        last_layer = Dense(1, use_bias=False, kernel_constraint=None, name="last_layer")(merge)
        output = Activation('sigmoid')(last_layer)
        model = Model(inputs=input_data, outputs=output)
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
        model.summary()
        early_stopping = EarlyStopping(monitor='loss', patience=2, verbose=1, mode='auto')

        mod = model.fit(data_set['X_train'], data_set['Y_train'], batch_size=batch_size, epochs=epochs, verbose=1,
                        callbacks=[early_stopping], shuffle=True,
                        validation_data=(data_set['X_test'], data_set['Y_test']))
        mod_loss = mod.history['loss']

        # check to make sure optimization didn't diverged
        if ~np.isnan(mod_loss[-1]):
            if not os.path.exists(dir_names['CNN_models']):
                os.makedirs(dir_names['CNN_models'])
            model.save(os.path.join(dir_names['CNN_models'], 'cnn_model_' + str(i) + '.hdf5'))

            i += 1
            if i > 4:
                break


