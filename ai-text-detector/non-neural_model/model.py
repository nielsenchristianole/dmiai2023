import pandas as pd
import numpy as np
import re
from collections import defaultdict

def grab_word_dict():
    # Step 1: Open the .txt file
    with open('non-neural_model/danske_ord.txt', 'r') as f:
        # Step 2: Initialize an empty dictionary
        word_dict = {}

        # Step 3: For each line in the file
        for line in f:
            # Remove all spaces, dots, numbers and linebreaks
            line = re.sub(r'[\s\.0-9\n]', '', line)

            # Split the line by the semicolon to get the word and word_type
            word, word_type = line.split(';')
            
            # Add the word as a key and the word_type as a value to the dictionary
            word_dict[word] = word_type
            
        return word_dict

# Load data.csv from data/ folder
data = pd.read_csv('data/data.csv')

word_dict = grab_word_dict()

word_types = set(word_dict.values())

word_types_dict = {word_type: i for i, word_type in enumerate(word_types)}

for text, label in zip(data['text'],data['is_generated']):
    
    # Remove '\n' and '\r'
    text = re.sub(r'[\n\r]', '', text)
    
    char_to_keep = r'[^a-zA-Z0-9æøåÆØÅ ]' # Char to keep
    special_char = re.findall(char_to_keep, text) # Count special char
    
    # Count number of comma, dots, dashes, exclamation marks and question marks seperately
    dots_and_dashes = [',', '.', '-', '!', '?']
    special_char = [special_char.count(char) for char in dots_and_dashes]
    
    text = re.sub(char_to_keep, '', text) # Remove special char
    
    # Convert to list
    words = text.lower().split()
    
    # word type count
    word_type_count = [0 for _ in range(len(word_types))]
    
    # Average word length
    avg_word_length = sum([len(word) for word in words])/len(words)
    
    # Lix score
    n_words = len(words)
    n_long_words = len([word for word in words if len(word) > 6])
    n_special_char = sum(special_char)
    lix_score = n_words/n_long_words + 100*n_special_char/n_words
    
    # Count word types
    for word in text:
        if word in word_dict:
            word_type = word_dict[word]
            word_type_count[word_types_dict[word_type]] += 1
    
    # print(list(word_type_count.values()))
    print(word_type_count, n_special_char, round(lix_score,2), label)
    
    
    # for word in text:
    #     if word in word_dict:
    #         print(word, word_dict[word])

