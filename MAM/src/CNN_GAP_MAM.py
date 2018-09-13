"""
This file implements the convolutional neural network to train,
evaluate, and make inference prediction, generation.
"""

import numpy as np
import pandas as pd
from collections import OrderedDict
import re
import os
import io
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.manifold import TSNE
from scipy import stats
from random import randint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from keras.layers import Conv1D
from keras.utils import np_utils
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.layers import Lambda
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from keras import regularizers

# Specify an ordering of amino acids for vectorizing peptide sequence
aa_idx = OrderedDict([
    ('A', 1),
    ('C', 2),
    ('E', 3),
    ('D', 4),
    ('G', 5),
    ('F', 6),
    ('I', 7),
    ('H', 8),
    ('K', 9),
    ('M', 10),
    ('L', 11),
    ('N', 12),
    ('Q', 13),
    ('P', 14),
    ('S', 15),
    ('R', 16),
    ('T', 17),
    ('W', 18),
    ('V', 19),
    ('Y', 20)
])


def build_training_matrix(fname, peptide, peptide_n_mer):
    """ Reads the training data file and returns the sequences of peptides
        and target values
    """
    df = pd.read_csv(fname, delim_whitespace=True, header=0)
    df.columns = ['sequence', 'HLA', 'target']

    # build training matrix
    df = df[df.HLA == peptide]
    df = df[df['sequence'].map(len) == peptide_n_mer]

    # remove any peptide with  unknown variables
    df = df[df.sequence.str.contains('X') == False]
    df = df[df.sequence.str.contains('B') == False]
    # remap target values to 1's and 0's
    df['target'] = np.where(df.target == 1, 1, 0)

    seqMatrix = df.sequence
    targetMatrix = df.target
    targetMatrix = targetMatrix.as_matrix()
    return seqMatrix, targetMatrix


def build_test_matrix(fname):
    """ Reads the test data file and extracts allele subtype,
        peptide length, and measurement type. Returns these information
        along with the peptide sequence and target values.
    """
    test_df = pd.read_csv(fname, delim_whitespace=True)
    peptide = re.search(r'[A-Z]\*\d{2}:\d{2}', test_df['Allele'][0]).group()
    peptide_length = len(test_df['Peptide_seq'][0])
    measurement_type = test_df['Measurement_type'][0]

    if measurement_type.lower() == 'binary':
        test_df['Measurement_value'] = np.where(test_df.Measurement_value == 1.0, 1, 0)
    else:
        test_df['Measurement_value'] = np.where(test_df.Measurement_value < 500.0, 1, 0)
    test_peptide = test_df.Peptide_seq
    test_target = test_df.Measurement_value
    test_target = test_target.as_matrix()
    return test_peptide, test_target, peptide_length, peptide


def aa_integerMapping(peptideSeq):
    """ maps amino acid to its numerical index
    """
    peptideArray = []
    for aa in peptideSeq:
        peptideArray.append(aa_idx[aa])
    return np.asarray(peptideArray)


def aa_inverse_Mapping(peptideArray):
    peptideSeq = []
    # print(peptideArray)
    for aa in peptideArray:
        peptideSeq.append(list(aa_idx.keys())[list(aa_idx.values()).index(int(aa))])
        # print(peptideSeq)
    return np.asarray(peptideSeq)


def read_in_datasets(dirnames):
    """ Reads the specified train and test files and return the
        relevant design and target matrix for the learning pipeline.
    """
    test_peptide, test_target, peptide_n_mer, peptide = build_test_matrix(dirnames['test_set'])
    seqMatrix, targetMatrix = build_training_matrix(dirnames['train_set'], peptide, peptide_n_mer)

    # map the training peptide sequences to their integer index
    featureMatrix = np.empty((0, peptide_n_mer), int)
    for num in range(len(seqMatrix)):
        featureMatrix = np.append(featureMatrix, [aa_integerMapping(seqMatrix.iloc[num])], axis=0)

    # map the test peptide sequences to their integer index
    testMatrix = np.empty((0, peptide_n_mer), int)
    for num in range(len(test_peptide)):
        testMatrix = np.append(testMatrix, [aa_integerMapping(test_peptide.iloc[num])], axis=0)

    # create training and test datasets
    datasets = {}
    datasets['X_train'] = featureMatrix
    datasets['Y_train'] = targetMatrix
    datasets['X_test'] = testMatrix
    datasets['Y_test'] = test_target
    return datasets, peptide_n_mer


def make_predictions(dirnames, datasets):
    """ Makes inference prediction
    """
    length = 9
    # width = 32
    X_test = datasets['X_test']
    predScores = np.zeros((5, len(X_test)))
    cams = np.zeros((5, len(X_test), length))
    for i in range(5):
        model = load_model(os.path.join(dirnames['CNN_models'], 'cnn_model_' + str(i) + '.hdf5'))
        predScores[i, :] = np.squeeze(model.predict(X_test))
        cams[i, :] = motif_map(model, X_test)
        # print(cams[i,:])
    predScoresAvg = np.average(predScores, axis=0)
    camspredScoresAvg = np.average(cams, axis=0)
    return predScoresAvg, camspredScoresAvg


def generate_peptide(dirnames, Y_pred, cam_matrix, datasets):
    """ write out predictions scores and labels to a new file
    """
    # testset = pd.read_csv(dirnames['test_set'], delim_whitespace=True, header=0)
    if not os.path.exists(os.path.join(dirnames['results_generation'])):
        os.makedirs(os.path.join(dirnames['results_generation']))

    # predicted_labels = ["binding" if score >= .5 else "non-binding" for score in Y_pred]

    # write prediction to new file
    with open(os.path.join(dirnames['results_generation'], "_generation.txt"), 'w') as f:
        f.write('new_peptide' + '\n')
        for i, score in enumerate(Y_pred):
            if score >= .5:
                batch_cam_matrix = cam_matrix[i, :]
                batch_X_test = datasets['X_test'][i, :]
                batch_X_test_mutated = mutation(batch_cam_matrix, batch_X_test)
                # cam_matrix[i,:] = batch_cam_matrix
                remap_peptide = aa_inverse_Mapping(batch_X_test_mutated)
                for single in remap_peptide:
                    print(single, end='')
                print("\n")
                # f.write(remap_peptide + '\n')
    f.close()

# This is for mutation
def mutation(batch_cam_matrix, batch_X_test):
    # print(batch_cam_matrix)
    batch_cam_matrix_list = batch_cam_matrix.tolist()
    min_index = batch_cam_matrix_list.index(min(batch_cam_matrix_list))
    useless_amino_acid_number = batch_X_test[min_index]
    # print(useless_amino_acid_number)
    replace_amino_acid_number = randint(1, 20)
    while replace_amino_acid_number == useless_amino_acid_number:
        replace_amino_acid_number = randint(1, 20)
    # print(replace_amino_acid_number)
    batch_X_test[min_index] = replace_amino_acid_number
    # print(batch_X_test)
    return batch_X_test


def write_predictions(dirnames, Y_pred):
    """
    generate new peptide automatically
    """
    testset = pd.read_csv(dirnames['test_set'], delim_whitespace=True, header=0)
    if not os.path.exists(os.path.join(dirnames['results'])):
        os.makedirs(os.path.join(dirnames['results']))

    predicted_labels = ["binding" if score >= .7 else "non-binding" for score in Y_pred]

    # write prediction to new file
    with open(os.path.join(dirnames['results'], "predictions.csv"), 'w') as f:
        f.write('Peptide_seq' + ',' + 'Predicted_Scores' + ',' + 'Predicted_Labels' + '\n')
        for i in range(len(Y_pred)):
            f.write(str(testset['Peptide_seq'].iloc[i]) + ',' + str(Y_pred[i]) + ',' + predicted_labels[i] + '\n')
    f.close()


def train(params, dirnames):
    """
	Trains the HLA-CNN model,the input is [batch_size, peptide, vector]

	"""
    datasets, peptide_n_mer = read_in_datasets(dirnames)

    # CNN parameters
    batch_size = int(
        np.ceil(len(datasets['X_train']) / 100.0))  # variable batch size depending on number of data points
    epochs = int(params['epochs'])
    nb_filter = int(params['filter_size'])
    filter_length = int(params['filter_length'])
    dropout = float(params['dropout'])
    lr = float(params['lr'])

    # load in learned distributed representation Vec
    hla_vec_obj = Word2Vec.load(os.path.join(dirnames['Vec_embedding'], 'Vec_Object'))
    hla_vec_embd = hla_vec_obj.wv
    embd_shape = hla_vec_embd.syn0.shape
    embedding_weights = np.random.rand(embd_shape[0] + 1, embd_shape[1])
    for key in aa_idx.keys():
        embedding_weights[aa_idx[key], :] = hla_vec_embd[key]
        embedded_dim = embd_shape[1]

    i = 0
    while True:
        #model = None
        model = Sequential()

        model.add(Embedding(input_dim=len(aa_idx) + 1, output_dim=embedded_dim, weights=[embedding_weights],
                            input_length=peptide_n_mer, trainable=True))
        model.add(
            Conv1D(int(nb_filter / 4), filter_length, padding='same', kernel_initializer='glorot_normal', name="conv0"))
        model.add(LeakyReLU(.3))
        model.add(Dropout(dropout))
        # model.add(Conv1D(int(nb_filter/2), filter_length, padding='same', kernel_initializer='glorot_normal',name="conv1"))
        # model.add(LeakyReLU(.3))
        # model.add(Dropout(dropout))

        model.add(Conv1D(int(nb_filter), filter_length, padding='same', kernel_initializer='glorot_normal',
                         kernel_regularizer=regularizers.l2(0.01)
                         , name="conv1"))
        model.add(LeakyReLU(.3))
        # model.add(Dropout(dropout))

        model.add(Conv1D(int(nb_filter), filter_length, padding='same', kernel_initializer='glorot_normal',
                         kernel_regularizer=regularizers.l2(0.01), name="conv_out"))
        model.add(LeakyReLU(.3))
        model.summary()
        # the MAM
        model.add(Lambda(global_average_pooling, name="gap"))

        model.add(Dense(1, use_bias=False, name="get_output1", kernel_constraint=None))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=lr),
                      metrics=['accuracy'])

        earlyStopping = EarlyStopping(monitor='loss', patience=2, verbose=1, mode='auto')
        mod = model.fit(datasets['X_train'], datasets['Y_train'], batch_size=batch_size, epochs=epochs, verbose=1,
                        callbacks=[earlyStopping], shuffle=True,
                        validation_data=(datasets['X_test'], datasets['Y_test']))
        modLoss = mod.history['loss']

        # check to make sure optimization didn't diverged
        if ~np.isnan(modLoss[-1]):
            if not os.path.exists(dirnames['CNN_models']):
                os.makedirs(dirnames['CNN_models'])
            model.save(os.path.join(dirnames['CNN_models'], 'cnn_model_' + str(i) + '.hdf5'))

            i += 1
            if i > 4:
                break


def evaluate(dirnames):
    """ Evaluates the test file and calculate SRCC and AUC score.
    """
    datasets, _ = read_in_datasets(dirnames)
    Y_test = datasets['Y_test']
    predMatrix = np.zeros((5, len(Y_test)))
    Y_pred, cam_matrix = make_predictions(dirnames, datasets)
    mean_fpr, mean_tpr, mean_thresholds = roc_curve(Y_test, Y_pred, pos_label=1)
    mean_auc = auc(mean_fpr, mean_tpr)
    rho, pValue = stats.spearmanr(Y_test, Y_pred)
    print('SRCC: ' + str(round(rho, 3)))
    print('AUC: ' + str(round(mean_auc, 3)))
    # print(cam_matrix.shape)
    tsne_peptide(cam_matrix)


def inference(dirnames):
    """ Makes inference prediction on the test file.
    """
    datasets, _ = read_in_datasets(dirnames)
    Y_pred, cam_matrix = make_predictions(dirnames, datasets)
    # print(cam_matrix)
    generate_peptide(dirnames, Y_pred, cam_matrix, datasets)


def global_average_pooling(x):
    return K.mean(x, axis=(1))


def global_average_pooling_shape(input_shape):
    return input_shape[0:2]


def motif_map(model, X_test):
    peptide_v = X_test
    # Get the 512 input weights to the softmax.
    class_weights = model.layers[-2].get_weights()[0]  # 32*1
    # print(class_weights)
    final_conv_layer = model.get_layer("conv_out")
    get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([peptide_v])
    # conv_outputs = conv_outputs[3, :, :] #9*32
    print("start count map value")
    cams = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])  # batch pep channel
    for i, single_conv_output in enumerate(conv_outputs):
        # de facto, this can be optimized.
        # Create the class activation map.
        single_cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[1])
        single_cam = np.dot(conv_outputs[i, :, :], class_weights)
        # print(single_cam.shape)
        cam_normal = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(single_cam)
        cams[i, :] = np.squeeze(cam_normal)
        # print(cams.shape)
    return cams

# calculate the TSNE
def tsne_peptide(cam_matrix):  # 77*9
    tsne = TSNE(n_components=2).fit_transform(cam_matrix)
    print(tsne.shape)
    for dot in tsne:
        plt.scatter(dot[0], dot[1])
# plt.show()
