# -*- coding: utf-8 -*-
import sys
from sklearn.manifold import TSNE
import seaborn as sns
from scipy import stats
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
from random import randint
from keras import backend as K
import os
import numpy as np
import pandas as pd
import re
from collections import OrderedDict
from keras.models import load_model
from sklearn.cross_validation import train_test_split

# Specify an ordering of amino acids for vectorizing peptide sequence
AA_IDX = OrderedDict([
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


def str2bool(word):
    """ It is from https://github.com/uci-cbcl/HLA-bind. """
    if word.lower() == 'true':
        return True
    else:
        return False


def build_training_matrix(file_name, peptide, peptide_n_mer):
    """ Reads the training data file and returns the sequences of peptides
        and target values
    """
    train_data = pd.read_csv(file_name, delim_whitespace=True, header=0)
    train_data.columns = ['sequence', 'HLA', 'target']

    # build training matrix
    peptide_data = train_data[train_data.HLA == peptide]
    n_mer_data = peptide_data[peptide_data['sequence'].map(len) == peptide_n_mer]

    # remove any peptide with  unknown variables
    filtered_x_data = n_mer_data[n_mer_data.sequence.str.contains('X') == False]
    filtered_xb_data = filtered_x_data[filtered_x_data.sequence.str.contains('B') == False]

    # remap target values to 1's and 0's
    filtered_xb_data['target'] = np.where(filtered_xb_data.target == 1, 1, 0)

    seq_matrix = filtered_xb_data.sequence
    target_matrix = filtered_xb_data.target
    target_matrix = target_matrix.as_matrix()
    return seq_matrix, target_matrix


def build_test_matrix(file_name):
    """ Reads the test data file and extracts allele subtype,
        peptide length, and measurement type. Returns these information
        along with the peptide sequence and target values.
    """
    test_data = pd.read_csv(file_name, delim_whitespace=True)
    print(test_data['Allele'][0])
    #peptide = re.search(r'[A-Z]\*\d{2}:\d{2}', test_data['Allele'][0]).group()
    peptide = test_data['Allele'][0] #animals
    peptide_length = len(test_data['Peptide_seq'][0])
    measurement_type = test_data['Measurement_type'][0]

    if measurement_type.lower() == 'binary':
        test_data['Measurement_value'] = np.where(test_data.Measurement_value == 1.0, 1, 0)
    else:
        test_data['Measurement_value'] = np.where(test_data.Measurement_value < 500.0, 1, 0)
    test_peptide = test_data.Peptide_seq
    test_target = test_data.Measurement_value
    test_target_matrix = test_target.as_matrix()
    return test_peptide, test_target_matrix, peptide_length, peptide


def aa_map_int(peptide_sequence):
    """ maps amino acid to its numerical index
    """
    peptide_array = []
    for amino_acid in peptide_sequence:
        peptide_array.append(AA_IDX[amino_acid])
    return np.asarray(peptide_array)


def aa_inverse_mapping(peptide_array):
    peptide_sequence = []
    for amino_acid_num in peptide_array:
        value_index = list(AA_IDX.values()).index(int(amino_acid_num))
        amino_acid = list(AA_IDX.keys())[value_index]
        peptide_sequence.append(amino_acid)
    return np.asarray(peptide_sequence)


def read_data_set(dir_names):
    """ Reads the specified train and test files and return the
        relevant design and target matrix for the learning pipeline.
    """
    test_peptide, test_target, peptide_n_mer, peptide = build_test_matrix(dir_names['test_set'])
    "This is the another segmentation methods"
    seq_matrix, target_matrix = build_training_matrix(dir_names['train_set'], peptide, peptide_n_mer)
    seq_matrix,test_peptide,target_matrix,test_target = train_test_split(seq_matrix,target_matrix,test_size=0.05,random_state=1)

    # map the training peptide sequences to their integer index
    feature_matrix = np.empty((0, peptide_n_mer), int)
    for num in range(len(seq_matrix)):
        feature_matrix = np.append(feature_matrix, [aa_map_int(seq_matrix.iloc[num])], axis=0)
    # map the test peptide sequences to their integer index
    test_matrix = np.empty((0, peptide_n_mer), int)
    for num in range(len(test_peptide)):
        test_matrix = np.append(test_matrix, [aa_map_int(test_peptide.iloc[num])], axis=0)

    # create training and test data_set
    data_set = dict()
    data_set['X_train'] = feature_matrix
    data_set['Y_train'] = target_matrix
    data_set['X_test'] = test_matrix
    data_set['Y_test'] = test_target
    return data_set, peptide_n_mer


def make_predictions(dir_names, data_sets, mode):
    """ Makes inference prediction
    """
    x_test = data_sets['X_test']
    predict_scores = np.zeros((5, len(x_test)))
    if mode == 'fine tune':
        cams = np.zeros((5, len(x_test), len(x_test[1])))
        model_name = 'CNN_models_fine'
    elif mode == 'normal':
        length = 9
        cams = np.zeros((5, len(x_test), length))
        model_name = 'CNN_models'
    else:
        sys.exit('mode does not support')
    for i in range(5):
        model = load_model(os.path.join(dir_names[model_name], 'cnn_model_' + str(i) + '.hdf5'))
        predict_scores[i, :] = np.squeeze(model.predict(x_test))
        cams[i, :] = motif_map(model, x_test, mode)

    predict_scores_avg = np.average(predict_scores, axis=0)
    cams_predict_scores_avg = np.average(cams, axis=0)

    return predict_scores_avg, cams_predict_scores_avg,x_test


def motif_map(model, x_test, mode):
    peptide_v = x_test
    # Get the 512 input weights to the softmax.

    class_weights1 = model.get_layer("weight1").get_weights()  # 32*1
    class_weights2 = model.get_layer("weight2").get_weights()  # 4*1
    final_conv_layer = model.get_layer("conv_out")
    first_conv_layer = model.get_layer("conv1")
    last_layer_w = model.get_layer("last_layer").get_weights()
    get_output = K.function([model.layers[0].input],
                                  [first_conv_layer.output, model.layers[-1].output,
                                   final_conv_layer.output])
    [conv1_outputs, _, conv2_outputs] = get_output([peptide_v])
    if mode == 'fine tune':
        print("start count map value")
    else:
        print("start count cam")
    cams = np.zeros(dtype=np.float32, shape=x_test.shape[0:2])  # batch pep channel
    for i in range(0, x_test.shape[0]):
        # Create the class activation map.
        single_cam = last_layer_w[0][0] * np.dot(conv1_outputs[i, :, :], class_weights1)
        single_cam = single_cam+last_layer_w[0][1] * np.dot(conv2_outputs[i, :, :], class_weights2)
        cams[i, :] = np.squeeze(single_cam)
    return cams


def generate_peptide(dir_names, y_pred, cam_matrix, data_set):
    """ write out predictions scores and labels to a new file
    """
    if not os.path.exists(os.path.join(dir_names['results_generation'])):
        os.makedirs(os.path.join(dir_names['results_generation']))

    # write prediction to new file
    with open(os.path.join(dir_names['results_generation'], "_generation.txt"), 'w') as f:
        f.write('new_peptide' + '\n')
        for i, score in enumerate(y_pred):
            if score >= .5:
                batch_cam_matrix = cam_matrix[i, :]
                batch_x_test = data_set['X_test'][i, :]
                batch_x_test_mutated = mutation(batch_cam_matrix, batch_x_test)
                remap_peptide = aa_inverse_mapping(batch_x_test_mutated)
                for single in remap_peptide:
                    print(single, end='')
                print("\n")


def mutation(batch_cam_matrix, batch_x_test):
    batch_cam_matrix_list = batch_cam_matrix.tolist()
    min_index = batch_cam_matrix_list.index(min(batch_cam_matrix_list))
    useless_amino_acid_number = batch_x_test[min_index]
    replace_amino_acid_number = randint(1, 20)
    while replace_amino_acid_number == useless_amino_acid_number:
        replace_amino_acid_number = randint(1, 20)
    batch_x_test[min_index] = replace_amino_acid_number
    return batch_x_test


def write_predictions(dir_names, y_predict, mode):
    """ generate new peptide automatically
    """
    test_set = pd.read_csv(dir_names['test_set'], delim_whitespace=True, header=0)
    if not os.path.exists(os.path.join(dir_names['results'])):
        os.makedirs(os.path.join(dir_names['results']))

    if mode == 'fine tune':
        score_level = .5
    elif mode == 'normal':
        score_level = .7
    else:
        sys.exit('mode does not support')
    predicted_labels = ["binding" if score >= score_level else "non-binding" for score in y_predict]

    # write prediction to new file
    with open(os.path.join(dir_names['results'], "predictions.csv"), 'w') as f:
        f.write('Peptide_seq' + ',' + 'Predicted_Scores' + ',' + 'Predicted_Labels' + '\n')
        for i in range(len(y_predict)):
            f.write(str(test_set['Peptide_seq'].iloc[i]) + ',' + str(y_predict[i]) + ',' + predicted_labels[i] + '\n')
    f.close()


def evaluate(dir_names, mode):
    """
    Evaluates the test file and calculate SRCC and AUC score.

    """
    data_set, _ = read_data_set(dir_names)
    y_test = data_set['Y_test']
    y_predict, cam_matrix,x_test = make_predictions(dir_names, data_set, mode)
    mean_fpr, mean_tpr, mean_thresholds = roc_curve(y_test, y_predict, pos_label=1)
    mean_auc = auc(mean_fpr, mean_tpr)
    rho, p_value = stats.spearmanr(y_test, y_predict)
    print('SRCC: ' + str(round(rho, 3)))
    print('AUC: ' + str(round(mean_auc, 3)))
    if mode == 'fine tune':
        print(rho)
        plt.plot(mean_fpr, mean_tpr)
        plt.show()
    elif mode == 'normal':
        heat_map(cam_matrix, mode)
    else:
        sys.exit('mode does not support')


def heat_map(cam_matrix, mode):
    #cam_matrix = cam_matrix[:20,:]
    shape = cam_matrix.shape
    print(shape[0])
    print("mode")
    sns.heatmap(cam_matrix, annot=False, xticklabels=range(1, shape[1]+1))
    plt.yticks(rotation=0)
    plt.ylabel('score of each generated peptide')
    plt.xlabel('site number')
    if mode == 'fine tune':
        plt.title('Mamu-A1*001:01 9mer')
        #plt.show()
    elif mode == 'normal':
        plt.title('HLA-A*0201 9mer')
        #plt.show()
    else:
        sys.exit('mode does not support')
    plt.savefig('heatmap_fine.pdf')


def inference_and_generation(dir_names, mode):
    """ Makes inference prediction and generation on the test file.
    """
    data_set, _ = read_data_set(dir_names)
    y_predict,cam_matrix,x_test = make_predictions(dir_names, data_set, mode)
    #tsne_peptide(cam_matrix,mode)
    heat_map(cam_matrix, mode)
    box(cam_matrix,mode)
    generate_peptide(dir_names, y_predict, cam_matrix, data_set)


def global_average_pooling(x):
    from keras import backend as K
    return K.mean(x, axis=1)


def global_average_pooling_shape(input_shape):
    return input_shape[0:2]


def tsne_peptide(cam_matrix, mode, mutate_encoding=None):
    tsne1 = TSNE(n_components=2).fit_transform(cam_matrix)
    for dot in tsne1:
        plt.scatter(dot[0], dot[1])
    if mode == 'fine tune':
        print(tsne1.shape)
    elif mode == 'normal':
        tsne2 = TSNE(n_components=2).fit_transform(mutate_encoding)
        for dot in tsne2:
            plt.scatter(dot[0], dot[1], c='r')
        plt.legend()
        plt.show()
    else:
        sys.exit('mode does not support')


def box(cam_matrix, mode):
    pf = pd.DataFrame(cam_matrix)
    pf.boxplot()
    plt.ylabel('score of each site')
    plt.xlabel('site number')
    if mode == 'fine tune':
        plt.title('HLA-B*2705 9mer')
    elif mode == 'normal':
        plt.title('HLA-A*0201 9mer')
    else:
        sys.exit('mode does not support')
    plt.savefig('box.pdf')


def draw_tsne(cam_matrix, mutate_after_test, dir_names, mode):
    x_test = mutate_after_test
    pred_scores = np.zeros((5, len(x_test)))
    cams = np.zeros((5, len(x_test), len(x_test[1])))
    for i in range(5):
        model = load_model(os.path.join(dir_names['HLA-CNN_models_new'], 'hla_cnn_model_' + str(i) + '.hdf5'))
        pred_scores[i, :] = np.squeeze(model.predict(x_test))
        cams[i, :] = motif_map(model, x_test, mode)
    cams_pred_scores_avg = np.average(cams, axis=0)
    mutate_encoding = cams_pred_scores_avg
    tsne_peptide(cam_matrix, mutate_encoding)


def tsne(cam_matrix):
    tsne1 = TSNE(n_components=2).fit_transform(cam_matrix)
    for i, dot in enumerate(tsne1):
        if i > 0.7:
            plt.scatter(dot[0], dot[1], c='r')
        else:
            plt.scatter(dot[0], dot[1], c='b')
    plt.legend()
    plt.show()

def heatmap(cam_matrix, data):
    data_num, data_length = data.shape
    new_matrix = np.zeros((20, data_length))
    for i in range(data_num):
        for j in range(data_length):
            tempt = data[i][j] - 1
            new_matrix[tempt][j] += cam_matrix[i][j]
    new_matrix /= data_num
    sns.heatmap(new_matrix, annot=False, xticklabels=range(1, data_length), yticklabels=['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y'])
    plt.yticks(rotation=0)
    plt.ylabel('score of each site')
    plt.xlabel('site number')
    plt.title('HLA-A*0201 9mer')
    plt.savefig('heatmap_fine.pdf')
    plt.close('all')
