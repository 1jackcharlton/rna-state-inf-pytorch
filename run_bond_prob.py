import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import bond_bool_prob
import bond_partial_prob
import bond_bool_prob_optimize

def main():
    # command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default = 1, type = int)
    parser.add_argument("--hiddensizes", default = 64, nargs='+', type = int)
    parser.add_argument("--lr", default = 0.001, type=float)
    parser.add_argument("--newmodel", default = False, type = bool) # change to train or load
    parser.add_argument("--savepredictionstrain", default = True, type = bool)
    parser.add_argument("--savepredictionstest", default = True, type = bool)
    parser.add_argument("--optimizer", default = 'adam') # can use 'rmsprop'
    parser.add_argument("--modelfilename", default = 'trained_model_optimizing_2.pth')
    parser.add_argument("--trainfilename", default = 'data/train_file_0.9.txt')
    parser.add_argument("--validfilename", default = 'data/valid_file_0.1.txt')
    parser.add_argument("--testfilename", default = 'data/test_file.txt') 
    parser.add_argument("--predtrainfilename", default = 'predictions/predictions_training.txt') 
    parser.add_argument("--predvalidfilename", default = 'predictions/predictions_valid.txt') 
    parser.add_argument("--predtestfilename", default = 'predictions/predictions_testing.txt') 
    parser.add_argument("--resultsvalidfilename", default = 'results/results_valid_file.txt') 
    parser.add_argument("--resultstestfilename", default = 'results/results_test_file.txt') 
   
    args = parser.parse_args()

    print('These are the RNN arguments: ')
    print(args, '\n')
    bond_bool_prob.RNN_for_RNA(args)
    return(0)
    
main()
