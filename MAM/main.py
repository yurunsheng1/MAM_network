"""
The main scripts. 
"""
import configparser
import sys

import CNN_fine_turn
import CNN_GAP_MAM
from utils import str2bool, evaluate, inference_and_generation
import Vec

if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    if str2bool(config['Pipeline']['Vec']):
        print("Starting to learn a distributed representation of amino acids...")
        Vec.run(config['Vec'], config['FilesDirectories'])

    if str2bool(config['Pipeline']['train']):
        print("Start training")
        CNN_GAP_MAM.train(config['Hyper-Parameter for MAM'], config['FilesDirectories'])
    if str2bool(config['Pipeline']['evaluate']):
        print("Performing prediction on the test set...")
        # CNN_GAP_MAM.evaluate(config['FilesDirectories'])
        evaluate(dir_names=config['FilesDirectories'], mode='normal')

    if str2bool(config['Pipeline']['inference']):
        print("Performing inference on the test set...")
        # CNN_GAP_MAM.inference_and_generation(config['FilesDirectories'])
        inference_and_generation(dir_names=config['FilesDirectories'], mode='normal')

    if str2bool(config['Pipeline']['fine_train']):
        print("Starting the fine turn training of HLA-CNN...")
        CNN_fine_turn.train(config['Hyper-Parameter for Fine-tune'], config['FilesDirectories'])

    if str2bool(config['Pipeline']['fine_evaluate']):
        print("Performing fine turn inference on the test set...")
        # CNN_fine_turn.evaluate(config['FilesDirectories'])
        evaluate(dir_names=config['FilesDirectories'], mode='fine tune')

    if str2bool(config['Pipeline']['fine_inference']):
        print("Performing fine turn test on the test set...")
        # CNN_fine_turn.inference_and_generation(config['FilesDirectories'])
        inference_and_generation(dir_names=config['FilesDirectories'], mode='fine tune')
