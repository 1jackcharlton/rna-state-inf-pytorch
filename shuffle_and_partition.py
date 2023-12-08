# Read the input file
import random
input_file = "crw16s-filtered.txt"
output_file = "crw16s-randomized.txt"
sets = []
current_set = []

with open(input_file, 'r') as file:
    for line in file:
        if line.strip():
            current_set.append(line)
        else:
            if current_set:
                sets.append(current_set)
                current_set = []

# Shuffle the sets
random.shuffle(sets)

# Write the shuffled sets to a new file
with open(output_file, 'w') as file:
    for set_ in sets:
        for line in set_:
            file.write(line)
        file.write('\n')





# Partition the training set into training and validation sets
def partition_data(file_path, train_file_path, valid_file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    total_sets = len(lines) // 5
    train_sets = int(total_sets * 0.9)

    indices = list(range(total_sets))
    random.shuffle(indices)

    train_indices = indices[:train_sets]
    valid_indices = indices[train_sets:]

    with open(train_file_path, 'w') as train_file, open(valid_file_path, 'w') as valid_file:
        for i in range(total_sets):
            start_line = i * 5
            end_line = (i + 1) * 5
            lines_set = lines[start_line:end_line]
            if i in train_indices:
                for line in lines_set:
                    train_file.write(line)
            else:
                for line in lines_set:
                    valid_file.write(line)

input_file_path = 'data/crw16s-randomized.txt'
train_file_path = 'data/train_file_0.9.txt'
valid_file_path = 'data/valid_file_0.1.txt'
partition_data(input_file_path, train_file_path, valid_file_path)