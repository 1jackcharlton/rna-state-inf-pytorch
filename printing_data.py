import argparse
from sympy import hyper
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import ParameterGrid
from itertools import product

def RNN_data_results_for_RNA(arguments):
    class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, dropout_prob):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.dropout = nn.Dropout(dropout_prob)
                self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
                self.linear = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional

            def forward(self, input_seq):
                lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
                output_seq = self.linear(lstm_out.view(len(input_seq), -1))
                return output_seq.view(-1, 1)


    # Data loading and preprocessing
    def load_data(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        sequences = [list(map(int, line.strip().split())) for i, line in enumerate(lines) if i % 5 == 1]
        bond_states = [list(map(int, line.strip().split())) for i, line in enumerate(lines) if i % 5 == 3]
        return sequences, bond_states


    # Convert input and target sequences to tensors
    def sequences_to_tensor(sequences, bond_states):
        input_tensor = torch.tensor(sequences, dtype=torch.float).view(-1, 1)
        target_tensor = torch.tensor(bond_states, dtype=torch.float).view(-1, 1)
        return input_tensor, target_tensor

    def test_model(model, sequences, bond_states):
        model.eval()
        with torch.no_grad():
            # Initializing accuracy metrics (true and false pos and neg)
            tpnum = 0
            tnnum = 0
            fpnum = 0
            fnnum = 0
            accuracy = 0
            for i in range(len(sequences)):
                input_tensor, _ = sequences_to_tensor(sequences[i], [])
                output = model(input_tensor)
                # Apply threshold to convert probabilities to binary predictions
                binary_output = torch.round(output).int().squeeze().tolist()
            

                # True Positives
                tp = sum([x and y for x, y in zip(binary_output, bond_states[i])])
                tpnum += tp
                # True Negatives
                tn = sum([not x and not y for x, y in zip(binary_output, bond_states[i])])
                tnnum += tn
                # False Positives
                fp = sum([x and not y for x, y in zip(binary_output, bond_states[i])])
                fpnum += fp
                # False Negatives
                fn = sum([not x and y for x, y in zip(binary_output, bond_states[i])])
                fnnum += fn
            
                # # UNCOMMENT FOR PREDICTIONS
                # # Save predictions to a text file
                # with open(f'predictions/predictions_{i}.txt', 'w') as file:
                #     file.write(' '.join(map(str, binary_output)))
        accuracy = (tpnum + tnnum)/(tpnum + fpnum + tnnum + fnnum)
        return(accuracy, tpnum, tnnum, fpnum, fnnum)


    def metrics(model, sequences, bond_states, metrics_array): 
        model.eval()
        with torch.no_grad():
            for i in range(len(sequences)):
                input_tensor, _ = sequences_to_tensor(sequences[i], [])
                output = model(input_tensor)
                # Apply threshold to convert probabilities to binary predictions
                binary_output = torch.round(output).int().squeeze().tolist()
            

                # True Positives
                tp = sum([x and y for x, y in zip(binary_output, bond_states[i])])
                # Number of bonds in the reference
                senval = sum([y for y in bond_states[i]])
                # Sensitivty
                sen = tp/senval
                # Number of predicted states that bond
                ppvval = senval = sum([x for x in binary_output])
                # Positive Predictive Value
                ppv = tp/ppvval

                metrics_array.append([sen, ppv])

        return metrics_array


    hyperparam_num = 1 # denote which combination of params like trial name

    # Load and process data
    training_sequences, training_bond_states = load_data(train_file_name)
    valid_sequences, valid_bond_states = load_data(valid_file_name)
    test_sequences, test_bond_states = load_data(test_file_name)

    # Call the test function
    while hyperparam_num < 18:
        input_size = 1
        output_size = 1
        epochs = 1
        train_file_name = arguments.trainfilename
        valid_file_name = arguments.validfilename
        test_file_name = arguments.testfilename
        results_valid_file_name = arguments.resultsvalidfilename
        results_test_file_name = arguments.resultstestfilename

        # Loading the saved model
        model = LSTMModel(input_size, hidden_size, output_size)
        model.load_state_dict(torch.load(f'trained_model_optimize_{hyperparam_num}.pth'))
        model.eval()  # Set the model to evaluation mode



        accuracy_valid, tpnum_valid, tnnum_valid, fpnum_valid, fnnum_valid = test_model(model, valid_sequences, valid_bond_states)
        accuracy_test, tpnum_test, tnnum_test, fpnum_test, fnnum_test = test_model(model, test_sequences, test_bond_states) 
        metrics_valid = metrics_valid = metrics(model, valid_sequences, valid_bond_states, metrics_valid)
        metrics_test = metrics(model, test_sequences, test_bond_states, metrics_test)



        # Printing the values
        print('\nHyperparameter Run:', hyperparam_num)
        print('Hyperparameter Vals:', params, '\n')
        
        print("Validation Set:")
        print("Accuracy:", accuracy_valid)
        print("True Positives:", tpnum_valid)
        print("True Negatives:", tnnum_valid)
        print("False Positives:", fpnum_valid)
        print("False Negatives:", fnnum_valid, "\n")

        print("Test Set:")
        print("Accuracy:", accuracy_test)
        print("True Positives:", tpnum_test)
        print("True Negatives:", tnnum_test)
        print("False Positives:", fpnum_test)
        print("False Negatives:", fnnum_test, "\n")
        

        

        with open(results_valid_file_name, 'w') as file:
            file.write(f"Hyperparameter Run: {hyperparam_num}\n")
            file.write(f"Params: {params}\n")
            file.write(f"Valid Set:\nAccuracy: {accuracy_valid}\nTrue Positives: {tpnum_valid}\nTrue Negatives: {tnnum_valid}\nFalse Positives: {fpnum_valid}\nFalse Negatives: {fnnum_valid}\n")
            file.write(f"Valid Set: Sensitivity and PPV: {metrics_valid(hyperparam_num-1)}\n\n")

        with open(results_test_file_name, 'w') as file:
            file.write(f"Hyperparameter Run: {hyperparam_num}\n")
            file.write(f"Params: {params}\n")   
            file.write(f"Test Set:\nAccuracy: {accuracy_test}\nTrue Positives: {tpnum_test}\nTrue Negatives: {tnnum_test}\nFalse Positives: {fpnum_test}\nFalse Negatives: {fnnum_test}\n")
            file.write(f"Test Set: Sensitivity and PPV: {metrics_test(hyperparam_num-1)}\n\n")
        
        hyperparam_num += 1
             
