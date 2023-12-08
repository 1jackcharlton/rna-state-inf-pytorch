import argparse
from sympy import hyper
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import ParameterGrid


# MAIN FUNCTION
###############
def RNN_param_for_RNA(arguments):
    # LSTM MODEL
    ############
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

    # TRAINING FUNCTION
    ###################
    def train_model(model, criterion, optimizer, input_tensor, target_tensor):
        optimizer.zero_grad()
        output = model(input_tensor) # padding make loss 0
    
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()
        return output, loss.item()


    # EXTRACT RNA SEQUENCES AND BOND STATES
    #######################################
    def load_data(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        sequences = [list(map(int, line.strip().split())) for i, line in enumerate(lines) if i % 5 == 1]
        bond_states = [list(map(int, line.strip().split())) for i, line in enumerate(lines) if i % 5 == 3]
        return sequences, bond_states


    # INPUT AND TARGET TENSOR CONVERSION
    ####################################
    def sequences_to_tensor(sequences, bond_states):
        input_tensor = torch.tensor(sequences, dtype=torch.float).view(-1, 1)
        target_tensor = torch.tensor(bond_states, dtype=torch.float).view(-1, 1)
        return input_tensor, target_tensor


    # HYPERPARAMETER GRID
    #####################
    hyper_param_num = 0 # denote which combination of params like trial name
    input_size = 1
    output_size = 1
    epochs = 1
    train_file_name = arguments.trainfilename
    valid_file_name = arguments.validfilename
    test_file_name = arguments.testfilename
    results_valid_file_name = arguments.resultsvalidfilename
    results_test_file_name = arguments.resultstestfilename

    # Define hyperparameter grid for optimization
    param_grid = {
        'l1_regularization': [0.00001, 0.0001, 0.001],
        #'l2_regularization': [0.00001, 0.0001, 0.001],
        'dropout_prob': [0, 0.2, 0.5, 0.8],
        'hidden_size': [64, 128, 256, 512],
        'learning_rate': [0.0001, 0.001, 0.01]
    }
   

    # MODEL TESTING FUNCTION
    ########################
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



    # ITERATE THROUGH PARAMETER GRID TO OPTIMIZE
    ############################################
    for params in ParameterGrid(param_grid):
        hyper_param_num += 1
        
        hidden_size = params['hidden_size']
        dropout_prob = params['dropout_prob']
        learning_rate = params['learning_rate']
        l1_reg = params['l1_regularization']
        
        # Load and process data
        training_sequences, training_bond_states = load_data(train_file_name)
        valid_sequences, valid_bond_states = load_data(valid_file_name)
        test_sequences, test_bond_states = load_data(test_file_name)

        model = LSTMModel(input_size, hidden_size, output_size, dropout_prob)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l1_reg)

        # # Loading the saved model
        # model = LSTMModel(input_size, hidden_size, output_size)
        # model.load_state_dict(torch.load('trained_model.pth'))
        # model.eval()  # Set the model to evaluation mode

        # Training loop 
        for epoch in range(epochs):
            for i in range(len(training_sequences)):
                input_tensor, target_tensor = sequences_to_tensor(training_sequences[i], training_bond_states[i])
                output, loss = train_model(model, criterion, optimizer, input_tensor, target_tensor)
                if i % 100 == 0:
                    print(f'Epoch {epoch + 1}, Step {i}, Loss {loss}')


        # Saving the trained model
        torch.save(model.state_dict(), f'trained_model_optimizing_{hyper_param_num}_testing.pth')
        
        # Call the test function
        accuracy_valid, tpnum_valid, tnnum_valid, fpnum_valid, fnnum_valid = test_model(model, valid_sequences, valid_bond_states)
        accuracy_test, tpnum_test, tnnum_test, fpnum_test, fnnum_test = test_model(model, test_sequences, test_bond_states)        


        # Printing the values
        print('\nHyperparameter Run:', hyper_param_num)
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
        

        
        # Adding accuracy results to output files
        with open(results_valid_file_name, 'a') as file:
            file.write(f"Hyperparameter Run: {hyper_param_num}\n")
            file.write(f"Params: {params}\n")
            file.write(f"Valid Set:\nAccuracy: {accuracy_valid}\nTrue Positives: {tpnum_valid}\nTrue Negatives: {tnnum_valid}\nFalse Positives: {fpnum_valid}\nFalse Negatives: {fnnum_valid}\n\n")
            

        with open(results_test_file_name, 'a') as file:
            file.write(f"Hyperparameter Run: {hyper_param_num}\n")
            file.write(f"Params: {params}\n")   
            file.write(f"Test Set:\nAccuracy: {accuracy_test}\nTrue Positives: {tpnum_test}\nTrue Negatives: {tnnum_test}\nFalse Positives: {fpnum_test}\nFalse Negatives: {fnnum_test}\n\n")
             
        




                



