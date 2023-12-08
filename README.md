# rna-state-inf-pytorch

The following scripts were designed by Jack Charlton with guidance provided by Dr. Luis Sanchez.

The general objectives for this repository are to 1. Create predictions of bond states in an RNA sequence using ML and 2. Use these predictions to inform/bias a program with secondary structure modeling and estimation.

The training, validation, and testing text files can be found at the Google Drive link below:
https://drive.google.com/drive/folders/1j6y0GpvP6QEME6ob5R2x66cOmtbfg-of?usp=drive_link

fasta files have not been included but can be recreated using the text files and the script Fasta_File_Creator.py

This project was designed for Windows operation using Python 3 to create RNA bond state predictions using an LSTM and to implement these predictions into a secondary structure predictor program–RNAstructure. 
All rights relating to RNAstructure belong to their respective creators. The download for RNAstructure can be found here: 
https://rna.urmc.rochester.edu/RNAstructureDownload.html



# script synopsis
bond_bool_prob.py is a function capable of training a bi-directional LSTM to predict whether a RNA base is likely to be bonded or not based on an input sequence of bases in a given strand. It can be called by run_bond_prob.py

bond_partial_prob.py behaves the same was as bond_bool_prob.py but predicts decimal probabilities of bond likelihoods rather than "not bonded" 0's and "bonded" 1's in the previously mentioned script

bond_bool_prob_optimize.py is a function created to perform a hyperparameter grid search to find the optimal settings for the LSTM, and it can be invoked by run_bond_prob.py

Fasta_File_Creator.py takes the various RNA .txt files and generates correlating .fasta (sequence) files for each RNA strand for use in fasta_to_ct_and_SHAPE(foldRNA).py and the RNAstructure program

fasta_to_ct_and_SHAPE(foldRNA).py is a script that will estimate the secondary structure of RNA sequences contingent on reactivity (SHAPE) data and the incoming sequence of bases (fasta), outputting connectivity table files by invoking RNAstructure

run_bond_prob.py can call any of the first three scripts and offers a parameter setup for custom settings and intentions

printing_data.py and shuffle_and_partition.py are supplemental scripts used for recording and initialization
