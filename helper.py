#!/usr/bin/env python3
# -*- coding: utf-8 -*-


##################################################################
# PROGRAMMER: Pierre-Antoine Ksinant                             #
# DATE CREATED: 14/11/2018                                       #
# REVISED DATE: -                                                #
# PURPOSE: This file consists of a library of functions used the #
#          associated Jupyter Notebook, core of the project. Its #
#          main purpose is to manage the data and the generated  #
#          model.                                                #
##################################################################


##################
# Needed imports #
##################

import os
import pickle
import torch


#######################
# Constant dictionary #
#######################

SPECIAL_WORDS = {'PADDING': '<PAD>'}


######################
# Function load_data #
######################

def load_data(path):
    """
    Load dataset from file
    """
    
    input_file = os.path.join(path)
    
    with open(input_file, "r") as f:
        data = f.read()

    return data


#####################################
# Function preprocess_and_save_data #
#####################################

def preprocess_and_save_data(dataset_path, token_lookup, create_lookup_tables):
    """
    Preprocess text data
    """
    
    text = load_data(dataset_path)
    
    # Ignore notice (we don't use it for analysing the data):
    text = text[81:]

    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))

    text = text.lower()
    text = text.split()

    vocab_to_int, int_to_vocab = create_lookup_tables(text + list(SPECIAL_WORDS.values()))
    int_text = [vocab_to_int[word] for word in text]
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('data/preprocessed_data.pickle', 'wb'))


############################
# Function load_preprocess #
############################

def load_preprocess():
    """
    Load the preprocessed training data and return them in batches of <batch_size> or less
    """
    
    return pickle.load(open('data/preprocessed_data.pickle', mode='rb'))


#######################
# Function save_model #
#######################

def save_model(filename, decoder):
    """
    Save the ENTIRE PyTorh model (not the recommended way to act)
    """
    
    torch.save(decoder, filename)


#######################
# Function load_model #
#######################

def load_model(filename):
    """
    Load an ENTIRE PyTorch model
    """
    
    return torch.load(filename)
