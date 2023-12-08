# no batch sizes which would require padding which means masking 
# writing, existing algorithms both end to end as well bond predicitons 5 papers on each
# github
# use mapping or normalization of bond predictions to match SHAPE range of values
import argparse
from re import A
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def RNN_for_RNA(arguments):
    # Define the LSTM model
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
            self.linear = nn.Linear(hidden_size * 2, output_size)

        def forward(self, input_seq):
            lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
            output_seq = self.linear(lstm_out.view(len(input_seq), -1))
            return output_seq.view(-1, 1)


    # Training function
    def train_model(model, criterion, optimizer, input_tensor, target_tensor):
        optimizer.zero_grad()
        output = model(input_tensor) # padding make loss 0
    
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()
        return output, loss.item()


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

    


    # Define hyperparameters
    input_size = 1
    hidden_size = arguments.hiddensizes
    output_size = 1
    learning_rate = arguments.lr
    epochs = arguments.epochs
    newmodel = arguments.newmodel 
    savepredictions_train = arguments.savepredictionstrain
    savepredictions_test = arguments.savepredictionstest 
    optimizer_name = arguments.optimizer 
    
    model_file_name = arguments.modelfilename
    train_file_name = arguments.trainfilename
    valid_file_name = arguments.validfilename
    test_file_name = arguments.testfilename
    pred_train_file_name = arguments.predtrainfilename
    pred_valid_file_name = arguments.predvalidfilename
    pred_test_file_name = arguments.predtestfilename
    results_valid_file_name = arguments.resultsvalidfilename
    results_test_file_name = arguments.resultstestfilename

    # Load and process data
    training_sequences, training_bond_states = load_data(train_file_name)
    valid_sequences, valid_bond_states = load_data(valid_file_name)
    test_sequences, test_bond_states = load_data(test_file_name)

    model = LSTMModel(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    if optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)




    # Training the model
    if newmodel == False:
        # Loading the saved model
        model = LSTMModel(input_size, hidden_size, output_size)
        model.load_state_dict(torch.load(model_file_name))
        model.eval()  # Set the model to evaluation mode
    else:
        # Training loop
        for epoch in range(epochs):
            for i in range(len(training_sequences)):
                input_tensor, target_tensor = sequences_to_tensor(training_sequences[i], training_bond_states[i])
                output, loss = train_model(model, criterion, optimizer, input_tensor, target_tensor)
                if i % 100 == 0:
                    print(f'Epoch {epoch + 1}, Step {i}, Loss {loss}')
        
        # Saving the trained model
        torch.save(model.state_dict(), model_file_name)

    if savepredictions_train == True:
        all_predictions = ""
        for i in range(len(training_sequences)):
            input_tensor, _ = sequences_to_tensor(training_sequences[i], [])
            output = model(input_tensor)
            # Apply threshold to convert probabilities to binary predictions
            float_output = torch.squeeze(output).tolist()
            
            all_predictions += ' '.join(map(str, float_output)) + "\n\n"

            
        # Save predictions to a text file
        with open(pred_train_file_name, 'w') as file:
             file.write(all_predictions) 



    # Testing the model
    def test_model(model, sequences, bond_states, pred_file_name):
        model.eval()
        with torch.no_grad():
            # Initializing accuracy metrics (true and false pos and neg)
            tpnum = 0
            tnnum = 0
            fpnum = 0
            fnnum = 0
            accuracy = 0
            all_predictions = ""
            for i in range(len(sequences)):
                input_tensor, _ = sequences_to_tensor(sequences[i], [])
                output = model(input_tensor)
                # Apply threshold to convert probabilities to binary predictions
                float_output = torch.squeeze(output).tolist()
            
                all_predictions += ' '.join(map(str, float_output)) + "\n\n"
                
                binary_output = torch.round(output).int().squeeze().tolist()

                # True Positives
                tp = [x and y for x, y in zip(binary_output, bond_states[i])]
                tpnum = tpnum + tp.count(1)
                # True Negatives
                tn = [not x and not y for x, y in zip(binary_output, bond_states[i])]
                tnnum = tnnum + tn.count(1)
                # False Positives
                fp = [x and not y for x, y in zip(binary_output, bond_states[i])]
                fpnum = fpnum + fp.count(1)
                # False Negatives
                fn = [not x and y for x, y in zip(binary_output, bond_states[i])]
                fnnum = fnnum + fn.count(1)
            
            # Save predictions to a text file
            if savepredictions_test == True:
                with open(pred_file_name, 'w') as file:
                    file.write(all_predictions)  


        accuracy = (tpnum + tnnum)/(tpnum + fpnum + tnnum + fnnum)
        return(accuracy, tpnum, tnnum, fpnum, fnnum)
                

    # Call the test function
    accuracy_valid, tpnum_valid, tnnum_valid, fpnum_valid, fnnum_valid = test_model(model, valid_sequences, valid_bond_states, pred_valid_file_name)
    accuracy_test, tpnum_test, tnnum_test, fpnum_test, fnnum_test = test_model(model, test_sequences, test_bond_states, pred_test_file_name)

    
    # Printing the values
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

    with open(results_valid_file_name, 'a') as file:
        file.write(f"Valid Set:\nAccuracy: {accuracy_valid}\nTrue Positives: {tpnum_valid}\nTrue Negatives: {tnnum_valid}\nFalse Positives: {fpnum_valid}\nFalse Negatives: {fnnum_valid}\n\n")
        file.write(f"Arguments: {arguments}\n\n")

    with open(results_test_file_name, 'a') as file:
        file.write(f"Test Set:\nAccuracy: {accuracy_test}\nTrue Positives: {tpnum_test}\nTrue Negatives: {tnnum_test}\nFalse Positives: {fpnum_test}\nFalse Negatives: {fnnum_test}\n\n")
        file.write(f"Arguments: {arguments}\n\n")    
    





'''# no batch sizes which would require padding which means masking
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        output_seq = self.linear(lstm_out.view(len(input_seq), -1))
        return output_seq.view(-1, 1)


# Training function
def train_model(model, criterion, optimizer, input_tensor, target_tensor):
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = criterion(output, target_tensor)
    loss.backward()
    optimizer.step()
    return output, loss.item()


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


# Define hyperparameters
input_size = 1
hidden_size = 128
output_size = 1
learning_rate = 0.001
epochs = 1

# Load and process data
training_sequences, training_bond_states = load_data('data/crw16s-shortened.txt')
test_sequences, test_bond_states = load_data('data/testset.txt')

model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    for i in range(len(training_sequences)):
        input_tensor, target_tensor = sequences_to_tensor(training_sequences[i], training_bond_states[i])
        output, loss = train_model(model, criterion, optimizer, input_tensor, target_tensor)
        if i % 100 == 0:
            print(f'Epoch {epoch + 1}, Step {i}, Loss {loss}')

# Testing the model
def test_model(model, test_sequences):
    model.eval()
    with torch.no_grad():
        for i in range(len(test_sequences)):
            input_tensor, _ = sequences_to_tensor(test_sequences[i], [])
            output = model(input_tensor)
            prob_out = output.squeeze().tolist()
            # Save predictions to a text file
            with open(f'predictions_{i}.txt', 'w') as file:
                file.write(' '.join(map(str, prob_out)))
                
# Call the test function
test_model(model, test_sequences)
'''
