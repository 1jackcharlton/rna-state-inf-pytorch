import os

# Data loading and preprocessing
def load_data(file_path, testing_file):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Specify if this is testing file format
    if testing_file == False:
        prefix = 'raw/crw16s/SB.'
        suffix = '.nopct'
    if testing_file == True:
        prefix = 'raw/zs/'
        suffix = '-native-nop.ct'
    else:
        print('Error. Specify if this is the testing file data (1True) or not (False)')    

    titles = [line.strip().replace(prefix, '').replace(suffix, '') for i, line in enumerate(lines) if i % 5 == 0]  
    sequences = [list(map(int, line.strip().split())) for i, line in enumerate(lines) if i % 5 == 1]
    bond_states = [list(map(int, line.strip().split())) for i, line in enumerate(lines) if i % 5 == 3]
    return titles, sequences, bond_states

# Convert 0 -> 'A', 1 -> 'C', 2 -> 'G', 3 -> 'U'
def convert_sequence(sequence):
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'U', 4: 'X'}
    if any(i not in mapping for i in sequence):        
        raise ValueError("Invalid element in sequence. Must be 0, 1, 2, or 3.")
    
    return ''.join(mapping[i] for i in sequence)

# Create a FASTA file with the given title and converted sequence.
def create_fasta_file(sequence, directory):
    with open(directory, 'w') as file:
        file.write(f'>{directory}\n{convert_sequence(sequence)}')
        
def main():
    print('Running . . .')
    testing_file = True # denote that processing the testing data set or not
    data_set_file_path = 'data/test_file.txt' # which data set to make into fasta files

    # Data extraction from master files
    titles, sequences, bond_states = load_data(data_set_file_path, testing_file)
    if testing_file == False:
        for title, sequence in zip(titles, sequences):
            file_path = f'{title}.fasta'
            filename = title.rsplit('/', 1)[1] 
            filename = f'{filename}.fasta'
            directory = title.split('/', 2)[1]
            directory = 'data/' + directory
            os.makedirs(directory, exist_ok=True)
        
            create_fasta_file(sequence, file_path)
    if testing_file == True:
        for title, sequence in zip(titles, sequences):   
            filename = f'{title}.fasta'
            filename = filename.rsplit('/', 1)[1] 
            directory = f'{title}'
            os.makedirs(directory, exist_ok=True)
            file_path = f'{directory}/{filename}'
            
            create_fasta_file(sequence, file_path)

if __name__ == "__main__":
    main()

    

