"""
This file is for fine tune the already trained MAM network and transfer them into MHC-I.

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
from keras.models import Sequential,Model
#from keras.legacy.models import Graph
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding,Input,Add
from keras.layers import Conv1D,Concatenate
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
import seaborn as sns

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
    print(test_df['Allele'][0])
    #peptide = re.search(r'[A-Z]\*\d{2}:\d{2}', test_df['Allele'][0]).group()
    peptide = test_df['Allele'][0] #animals
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
 peptideSeq=[]
 #print(peptideArray)
 for aa in peptideArray:
  peptideSeq.append(list(aa_idx.keys())[list(aa_idx.values()).index(int(aa))])
  #print(peptideSeq)
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
    datasets={}
    datasets['X_train'] = featureMatrix
    datasets['Y_train'] = targetMatrix
    datasets['X_test'] = testMatrix
    datasets['Y_test'] = test_target
    return datasets, peptide_n_mer


def make_predictions(dirnames, datasets):    
    """ Makes inference prediction
    """
    #length  = 10 #modifable
    #width = 32
    X_test = datasets['X_test']
    predScores = np.zeros((5, len(X_test)))
    cams = np.zeros((5,len(X_test),len(X_test[1])))
    for i in range(5):
        model = load_model(os.path.join(dirnames['CNN_models_fine'], 'cnn_model_' + str(i) + '.hdf5'))
        predScores[i, :] = np.squeeze(model.predict(X_test))
        cams[i,:] = visulization(model,X_test)
        #print(cams[i,:])
    predScoresAvg = np.average(predScores, axis=0)
    camspredScoresAvg = np.average(cams, axis=0)
    return predScoresAvg,camspredScoresAvg


def generate_peptide(dirnames, Y_pred,cam_matrix,datasets):
    """ write out predictions scores and labels to a new file
    """
    #testset = pd.read_csv(dirnames['test_set'], delim_whitespace=True, header=0)
    if not os.path.exists(os.path.join(dirnames['results_generation'])):
        os.makedirs(os.path.join(dirnames['results_generation']))
        
    #predicted_labels = ["binding" if score >= .5 else "non-binding" for score in Y_pred]
    
    # write prediction to new file
    with open(os.path.join(dirnames['results_generation'], "_generation.txt"), 'w') as f:
        f.write('new_peptide' +'\n')
        for i,score in enumerate(Y_pred):
         if score >= .7 :
          batch_cam_matrix = cam_matrix[i,:]
          batch_X_test = datasets['X_test'][i,:]
          batch_X_test_mutated = mutation(batch_cam_matrix,batch_X_test)
          #cam_matrix[i,:] = batch_cam_matrix
          remap_peptide = aa_inverse_Mapping(batch_X_test_mutated)
          for single in remap_peptide: 
           print(single, end='')
          print("\n")
          #f.write(remap_peptide + '\n')
    f.close()
	
def mutation(batch_cam_matrix,batch_X_test):
    #print(batch_cam_matrix)
    batch_cam_matrix_list = batch_cam_matrix.tolist()  
    min_index = batch_cam_matrix_list.index(min(batch_cam_matrix_list))
    useless_amino_acid_number = batch_X_test[min_index]
    #print(useless_amino_acid_number)
    replace_amino_acid_number = randint(1,20)
    while replace_amino_acid_number == useless_amino_acid_number :
     replace_amino_acid_number = randint(1,20)
    #print(replace_amino_acid_number)
    batch_X_test[min_index] = replace_amino_acid_number
    #print(batch_X_test)
    return batch_X_test
	
	
def write_predictions(dirnames, Y_pred):
    """ generate new peptide automatically
    """
    testset = pd.read_csv(dirnames['test_set'], delim_whitespace=True, header=0)
    if not os.path.exists(os.path.join(dirnames['results'])):
        os.makedirs(os.path.join(dirnames['results']))
        
    predicted_labels = ["binding" if score >= .5 else "non-binding" for score in Y_pred]
    
    # write prediction to new file
    with open(os.path.join(dirnames['results'], "predictions.csv"), 'w') as f:
        f.write('Peptide_seq' + ',' + 'Predicted_Scores' + ',' + 'Predicted_Labels' + '\n')
        for i in range(len(Y_pred)):
            f.write(str(testset['Peptide_seq'].iloc[i]) + ',' + str(Y_pred[i]) + ',' + predicted_labels[i] + '\n')
    f.close()
	

#batch_size, peptide, dimension
def train(params, dirnames):
	"""
	Trains the HLA-CNN model
	"""
	datasets, peptide_n_mer = read_in_datasets(dirnames)

    # CNN parameters
	batch_size = int(np.ceil(len(datasets['X_train']) / 100.0))  # variable batch size depending on number of data points
	epochs = int(params['epochs'])
	nb_filter = int(params['filter_size'])
	filter_length = int(params['filter_length'])
	dropout = float(params['dropout'])
	lr = float(params['lr'])

    # load in learned distributed representation MHC Vec
	hla_vec_obj = Word2Vec.load(os.path.join(dirnames['Vec_embedding'], 'Vec_Object'))
	hla_vec_embd = hla_vec_obj.wv
	embd_shape = hla_vec_embd.syn0.shape
	embedding_weights = np.random.rand(embd_shape[0] + 1, embd_shape[1])
	for key in aa_idx.keys():
		embedding_weights[aa_idx[key],:] = hla_vec_embd[key]
		embedded_dim = embd_shape[1]

	i = 0
	while True:
		model = None
		#model = Graph()
		model = load_model(os.path.join(dirnames['CNN_models'], 'cnn_model_' + str(i) + '.hdf5'))
		#model = Model(inputs=input, outputs=output)
		model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=lr),
                      metrics=['accuracy'])
		model.summary()
		earlyStopping = EarlyStopping(monitor='loss', patience=2, verbose=1, mode='auto')
		mod = model.fit(datasets['X_train'], datasets['Y_train'], batch_size=batch_size, epochs=epochs, verbose=1,
                        callbacks=[earlyStopping], shuffle=True, validation_data=(datasets['X_test'], datasets['Y_test']))
		modLoss = mod.history['loss']

		# check to make sure optimization didn't diverged
		if ~np.isnan(modLoss[-1]):
			if not os.path.exists(dirnames['CNN_models_fine']):
				os.makedirs(dirnames['CNN_models_fine'])
			model.save(os.path.join(dirnames['CNN_models_fine'], 'cnn_model_' + str(i) + '.hdf5'))

			i += 1
			if i > 4:
				break


def evaluate(dirnames):
    """
    Evaluates the test file and calculate SRCC and AUC score.

    """
    datasets,_ = read_in_datasets(dirnames)
    Y_test = datasets['Y_test']
    predMatrix = np.zeros((5, len(Y_test)))
    Y_pred,cam_matrix = make_predictions(dirnames, datasets)
    mean_fpr, mean_tpr, mean_thresholds = roc_curve(Y_test, Y_pred, pos_label=1)
    mean_auc = auc(mean_fpr, mean_tpr)
    rho, pValue = stats.spearmanr(Y_test, Y_pred)
    print('SRCC: ' + str(round(rho, 3)))
    print('AUC: ' + str(round(mean_auc,3)))
    print(rho)
    #print(cam_matrix.shape)
    #tsne_peptide(cam_matrix)
    #heatmap(cam_matrix)
    #box(cam_matrix) 
    #for i,single_tpr in enumerate(mean_tpr):
    plt.plot(mean_fpr,mean_tpr)
    #plt.legend()
    plt.show()
	
def inference_and_generation(dirnames):
    """ Makes inference prediction and generation on the test file.
    """
    datasets,_ = read_in_datasets(dirnames)
    Y_pred,cam_matrix = make_predictions(dirnames, datasets)
    #print(cam_matrix)
    generate_peptide(dirnames, Y_pred,cam_matrix,datasets)
	
	
def global_average_pooling(x):
        return K.mean(x, axis = (1))
    
def global_average_pooling_shape(input_shape):
        return input_shape[0:2]
		
			
def visulization(model,X_test):
    peptide_v = X_test  
    #Get the 512 input weights to the softmax.
    
    class_weights1 = model.get_layer("weight1").get_weights()#32*1
    class_weights2 = model.get_layer("weight2").get_weights()#4*1
    #print(class_weights2.shape)
    final_conv_layer = model.get_layer("conv_out")
    first_conv_layer = model.get_layer("conv1")
    last_layer_w = model.get_layer("last_layer").get_weights()
    #print(last_layer_w[0])
    #print(last_layer_w.shape)
    get_output = K.function([model.layers[0].input],[first_conv_layer.output, model.layers[-1].output,
     final_conv_layer.output])
    [conv1_outputs, predictions,conv2_outputs] = get_output([peptide_v])
    #conv_outputs = conv_outputs[3, :, :] #9*32
    print("start count map value")
    cams = np.zeros(dtype = np.float32, shape = X_test.shape[0:2]) #batch pep channel
    for i in range(0,X_test.shape[0]):
		# de facto, this can be optimized.	
		#Create the class activation map.
     #single_cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[1])
     single_cam = last_layer_w[0][0]*np.dot(conv1_outputs[i,:,:],class_weights1) 
     single_cam = single_cam+last_layer_w[0][1]*np.dot(conv2_outputs[i,:,:],class_weights2)
     #print(single_cam.shape)
     #cam_normal = preprocessing.MinMaxScaler( feature_range=(-1, 1)).fit_transform(single_cam)
     cams[i,:]= np.squeeze(single_cam) 
    #print(cams.shape)
    return cams



def tsne_peptide(cam_matrix):
	tsne=TSNE(n_components=2).fit_transform(cam_matrix)
	print(tsne.shape)
	for dot in tsne:
		plt.scatter(dot[0],dot[1])
	#plt.show()


def box(cam_matrix):
	pf = pd.DataFrame(cam_matrix)
	pf.boxplot()
	plt.ylabel('score of each site')
	plt.xlabel('site number')
	plt.title('HLA-B*2705 9mer')
	#plt.show()	



def heatmap(cam_matrix):
        sns.heatmap(cam_matrix,annot=False)
        plt.ylabel('score of each site')
        plt.xlabel('site number')
        plt.title('HLA-B*2705 9mer')
		#plt.show()
		
		