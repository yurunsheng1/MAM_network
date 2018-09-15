# MAM_network
The Keras implement of MAM.

The Keras implement of In silico design of MHC class I high binding affinity peptides through motifs activation map.It is a framewrok of both the unknown MHC-I peptide prediction and novel peptide generation.  Our paper is still under reviewed.


Prerequisites:
  Python 3.5.3
  Tensorflow 1.4.0
  Scipy 0.19.1
  Numpy 1.13.3
  Pandas 0.20.3
  Sklearn 0.18.2
  Keras 2.1.3
  Gensim 3.2.0


Getting Started:
The pipeline is as follows:
1. Amino Acid to Vector: Convert the Amino Acid into n-dimemsion vector.
  I Find config.ini and Set  Vec == True   
  II Run python main.py config.ini

2. Train model.
  I Find config.ini and Set  train == True  
  II Run python main.py config.ini

3. Evaluate model. Do the prediction on  the test dataset.
  I Find config.ini and Set  evaluate == True  
  II Run python main.py config.ini

4. Inference model. Do the prediction on  the test dataset as well as generation.
  I Find config.ini and Set  Inference == True  
  II Run python main.py config.ini

5. Fine-tuning. The setting of train, evaluate and inference model are the same operation as we mentioned above.


Dataset
The training dataset is available in train_data file while the test dataset is in  test_data file. 
Moreover, the label name in drawing function (e.g., heatmap, tsne) need to be rewritten by the user.  

Acknowledge: 
Some of the code function refer from HLA-CNN https://github.com/uci-cbcl/HLA-bind.
We appreciate so much for their excellent code. 	


If you have any question you can email runshengyu@gmail.com for help.
