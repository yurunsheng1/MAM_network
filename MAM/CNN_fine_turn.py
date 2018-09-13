# -*- coding: utf-8 -*-
"""
This file is for fine tune the already trained MAM network and transfer them into MHC-I.
"""
import os
import numpy as np
from gensim.models import Word2Vec
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.optimizers import Adam
from utils import read_data_set, AA_IDX


def train(params, dir_names):
    """
    Fine-tune the model
    """
    data_set, peptide_n_mer = read_data_set(dir_names)

    # CNN parameters
    # variable batch size depending on number of data points
    batch_size = int(np.ceil(len(data_set['X_train']) / 100.0))
    epochs = int(params['epochs'])
    lr = float(params['lr'])

    # load in learned distributed representation MHC Vec
    hla_vec_obj = Word2Vec.load(os.path.join(dir_names['Vec_embedding'], 'Vec_Object'))
    hla_vec_embed = hla_vec_obj.wv
    embed_shape = hla_vec_embed.syn0.shape
    embedding_weights = np.random.rand(embed_shape[0] + 1, embed_shape[1])
    for key in AA_IDX.keys():
        embedding_weights[AA_IDX[key], :] = hla_vec_embed[key]

    i = 0
    while True:
        model = load_model(os.path.join(dir_names['CNN_models'], 'cnn_model_' + str(i) + '.hdf5'))
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=lr),
                      metrics=['accuracy'])
        model.summary()
        early_stopping = EarlyStopping(monitor='loss', patience=2, verbose=1, mode='auto')
        mod = model.fit(data_set['X_train'], data_set['Y_train'], batch_size=batch_size,
                        epochs=epochs, verbose=1, callbacks=[early_stopping], shuffle=True,
                        validation_data=(data_set['X_test'], data_set['Y_test']))
        mod_loss = mod.history['loss']

        # check to make sure optimization didn't diverged
        if ~np.isnan(mod_loss[-1]):
            if not os.path.exists(dir_names['CNN_models_fine']):
                os.makedirs(dir_names['CNN_models_fine'])
            model.save(os.path.join(dir_names['CNN_models_fine'], 'cnn_model_' + str(i) + '.hdf5'))

            i += 1
            if i > 4:
                break
