import os
import sys
import subprocess
import numpy as np



# BOND PREDICTION STATES
########################
# Getting labels of bond predicitions
def load_data(file_path, testing_file):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Specify if this is testing file format
    if testing_file == 0:
        prefix = 'raw/crw16s/SB.'
        suffix = '.nopct'
    if testing_file == 1:
        prefix = 'raw/zs/'
        suffix = '-native-nop.ct'
    else:
        print('Error. Specify if this is the testing file data (1) or not (0)')    

    titles = [line.strip().replace(prefix, '').replace(suffix, '') for i, line in enumerate(lines) if i % 5 == 0]  
    return titles

def find_file(directory_path, file_name):
    for foldername, subfolders, filenames in os.walk(directory_path):
        if file_name in filenames:
            # File found
            return os.path.join(foldername, file_name)

    # File not found
    return None





# SHAPE REPLACEMENT FILES - BOND PREDICTIONS
############################################
def makeshape(bond_predictions, titles, set_type, a = 0.214, b = 0.6624, outfile = None):
    s = 0.3603

    outfolder = f'shape/{set_type}'

    os.makedirs(outfolder, exist_ok=True)
    
    for bond_state_prediction, title in zip(bond_state_predictions, titles):
        # open output shape file
        outfile = title + '.SHAPE'
        output_file = open(outfolder+'/'+outfile, 'w')

        # for each position, write a SHAPE value depending on the probability
        for i, prediction in enumerate(bond_state_prediction):  
            # New prediction equation
            if (prediction < 0):
                shapevalue = 0.2042 # create threshold for negative values
            else:
                shapevalue = 10**((8/5)*(1-prediction)-2.29)
            shapevalue = shapevalue/0.2042 # normalize reactivities
            output_file.write('%d %.3f' % (i+1, shapevalue))
            
            if i != len(bond_state_prediction) - 1:
            # Add a newline character if it's not the last iteration
                output_file.write('\n')
            
        output_file.close()
    return





# RUNNING RNASTRUCTURE USING SHAPE AND SEQUENCE INPUTS
######################################################
def makestructures(shapetype, testnames, set_type, fastapath):
    iteration_count = 0

    filepath = f'sectructpred/{set_type}'
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        
    for testname in testnames:
        # shape file
        shapefilename = f"{testname}.SHAPE"
        shapefilepath = f"D:/1jack/Source/Repos/rna-state-inf-pytorch/shape/{set_type}/{shapefilename}" 
        shapefilepathcmd = shapefilepath.replace("/", "\\") # backslash for command
        shapefilepathcmd = f"\"{shapefilepath}\"" # adding quotes for command

        # fold command file
        foldfilename = "Fold.exe"
        foldfilepath = f"D:/Program Files/RNAstructure6.4/RNACommandLineInterface/RNAstructure6.4/exe/{foldfilename}"
        foldfilepathcmd = foldfilepath.replace("/", "\\") # backslash for command
        foldfilepathcmd = f"\"{foldfilepath}\"" # adding quotes for command

        # fasta file
        fastafilename = f"{testname}.fasta"
        fastafilepath = find_file(fastapath, fastafilename)
        fastafilepathcmd = fastafilepath.replace("/", "\\") # backslash for command
        fastafilepathcmd = f"\"{fastafilepath}\"" # adding quotes for command

        # ct file
        ctfilename = f"{testname}.ct"
        ctfilepath = f"D:/1jack/Source/Repos/rna-state-inf-pytorch/sectructpred/{set_type}/{shapetype}"
        # Ensure the output directory exists
        os.makedirs(ctfilepath, exist_ok=True)
        ctfilepath = f"{ctfilepath}/{ctfilename}"
        with open(ctfilepath, 'w') as file:
            file.close()
        ctfilepathcmd = ctfilepath.replace("/", "\\") # backslash for command
        ctfilepathcmd = f"\"{ctfilepathcmd}\"" # adding quotes for command
        

        
        # Path to RNAstructure data_tables directory
        thermodatapath = "D:/Program Files/RNAstructure6.4/RNAstructureSource/RNAstructure/data_tables"

        # Set the DATAPATH environment variable
        os.environ["DATAPATH"] = thermodatapath
        

        print(f"\nFolding {testname} from the {set_type} set with {shapetype} data . . .\n" )
        print(f"\n{iteration_count}/{len(testnames)} complete\n" )
        iteration_count += 1


        if shapetype == "noshape":
            # command with executable and arguments
            command = f"{foldfilepathcmd} {fastafilepathcmd} {ctfilepathcmd}"
        else:
            # command with executable and arguments
            command = f"{foldfilepathcmd} {fastafilepathcmd} {ctfilepathcmd}"
            shapestring = " --SHAPE  %s" % (shapefilepathcmd)
            command = command + shapestring
        result = subprocess.run(command)
        print("Return Code:", result.returncode)
    
    return

        

# OBTAINING NNTM CONNECTIVITY TABLE BOND PARTNERS
#################################################
def process_ct_file(testname, set_type, shapetype):
    # ct file
    ctfilename = f"{testname}.ct"
    ctfilepath = f"D:/1jack/Source/Repos/rna-state-inf-pytorch/sectructpred/{set_type}/{shapetype}"
    ctfilepath = f"{ctfilepath}/{ctfilename}"
    
    # Reading the file
    with open(ctfilepath, 'r') as file:
        lines = file.readlines()

    # Extracting sequence length from the first line
    sequence_length = int(lines[0].split()[0])

    ct_base_pairs = []

    # Processing subsequent lines
    for line in lines[1:sequence_length+1]: # ignore header in file
        elements = line.split()
        ct_base_pairs.append(int(elements[-2])) # grab only relevant second to last column

    return ct_base_pairs


# METRICS AND STATISTICS
########################
def metrics(seq_filepath, ct_base_pairs): 
    with open(seq_filepath, 'r') as file:
        lines = file.readlines()
    base_partners = [list(map(int, line.strip().split())) for i, line in enumerate(lines) if i % 5 == 2]
    metrics = []
    correct_pred_pair = 0
    for seq_pred, seq_true in zip(ct_base_pairs, base_partners):
        for i in range(1, len(seq_pred) + 1):
            for j in range(1, len(seq_true) + 1):
                # Check if the conditions are met
                if seq_pred[i-1] == j and seq_pred[j-1] == i and seq_pred[i-1] != 0 and seq_pred[i-1] == seq_true[i-1]:
                    # Increment the counter variable
                    correct_pred_pair += 0.5 # correct number of predicted base pairs
        reference_base_pairs = sum(0.5 for element in seq_true if element != 0) # number of base pairs in true sequence
        pred_base_pairs = sum(0.5 for element in seq_pred if element != 0) # number of base pairs in pred sequence
        sen = correct_pred_pair/ reference_base_pairs
        ppv = correct_pred_pair/pred_base_pairs
        metrics.append([sen, ppv])
        correct_pred_pair = 0

    return metrics

    
    
 
        
        


if __name__ == '__main__':
    fasta_path = 'data/test_file_RNAstruct' # where are fasta files stored
    shapetype = "shape" # specify if shape data is to be included
    bond_state_predictions = [] # LSTM predicted bond states
    pred_path = 'predictions/predictions_testing.txt'
    testing_file = 1 # denote that processing testing file = True
    seq_names = []; # ordered list of RNA seq specimen names
    seq_filepath = 'data/test_file.txt' # file with reference data
    ct_base_pairs = [] # predicted base pair partner sequences
    
    set_type = pred_path.split('.')[0] # get rid of extension
    set_type = set_type.split('_')[-1] # get rid of prefix, denotes if test, training or validation
    
    results_dir = f'results/{set_type}'
    os.makedirs(results_dir, exist_ok=True)
    results_file_path = f'{results_dir}/{shapetype}.txt' # metrics data
    
    new_SHAPE = True # toggle for new SHAPE files
    new_ct = True # toggle for new folded connectivity tables
    


    # record all the bond prediction sequences
    with open(pred_path, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2): # every other line
            line = lines[i].strip()
            bond_state_predictions.append([float(bit) for bit in line.split()])
            


    # Finding matching titles to each sequence of bond predictions
    titles = load_data(seq_filepath, testing_file)
    if testing_file == 0: # different formatting for non test set titles
        for title in titles:
            file_path = f'{title}'
            filename = title.rsplit('/', 1)[1] 
            seq_names.append(str(filename))
        
    if testing_file == 1: # different formatting for non test set titles
        for title in titles:   
            filename = f'{title}'
            filename = filename.rsplit('/', 1)[1] 
            seq_names.append(str(filename))
    
    # Create files if not already present        
    if new_SHAPE:
        makeshape(bond_state_predictions, seq_names, set_type) # SHAPE files
    if new_ct:
        makestructures(shapetype, seq_names, set_type, fasta_path) # ct files
    
    # Record all RNA predicted base partner sequences
    for seq_name in seq_names: # for each ct file
        ct_base_pairs.append(process_ct_file(seq_name, set_type, shapetype)) # append ct predicted seq of base pairings
    metrics = metrics(seq_filepath, ct_base_pairs) # collect stats
    
    # Display statistics and save to file for each CT sequence
    for metric, seq_name in zip(metrics, seq_names):
        sen = metric[0] # sensitivity
        ppv = metric[1] # positive predicted value
        fscore = 2*ppv*sen/(ppv+sen)
        print(f'Sensitivity: {sen}')
        print(f'PPV: {ppv}')
        print(f'Fscore: {fscore}')
        
        with open(results_file_path, 'a') as file:
            file.write(f"RNA Sequence: {seq_name}\nSensitivity: {sen}\nPPV: {ppv}\nFscore: {fscore}\n\n")
        
        
    
