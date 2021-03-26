from text_to_data import Doc2Data
from text_to_data import CalculatePair
from model import train_model, get_data
from progress.bar import Bar
import os
from corpy.udpipe import Model
import stopwords
from random import randint
import numpy as np

m = Model("russian-syntagrus-ud-2.5-191206.udpipe")
stop = stopwords.get_stopwords('ru')
# postfix can be _sm for demo corpus and _med for other texts
postfix = '40k'

text_folder_name = 'texts' + postfix + '/'
data_folder_name = 'data' + postfix + '/'


def make_data_from_texts():
    # goes through folder and process all texts in json
    all_texts = os.listdir(text_folder_name)
    for text in Bar(' text parsing...').iter(all_texts):
        Doc2Data(text_folder_name + text, m, stop, data_folder_name)


def make_pairs(authors):
    all_texts = os.listdir(text_folder_name)
    texts = open('db' + postfix + '.csv', 'r').read().split('\n')[:authors]
    text = []
    for i in texts:
        i = i.split(',')[1]
        text.append(i.split('|')[1:])
    aut_texts = text
    train_data = []
    labels = []
    for author in Bar('Author: ').iter(aut_texts):
        for i, text in enumerate(author[:-1]):
            try:
                open(text_folder_name + text + '.txt', encoding='utf8')
            except FileNotFoundError:
                continue
            for right_text in author[i + 1:]:
                if right_text == text:
                    continue
                try:
                    open(text_folder_name + right_text + '.txt', encoding='utf8')
                except FileNotFoundError:
                    continue
                while True:
                    false_text = '/' + all_texts[randint(0, len(all_texts) - 1)][:-4]
                    false_text_2 = '/' + all_texts[randint(0, len(all_texts) - 1)][:-4]
                    if false_text not in author and false_text_2 not in author and false_text != false_text_2:
                        break
                valid = CalculatePair(text, right_text, data_folder_name).get_pair_data()
                invalid = CalculatePair(text, false_text, data_folder_name).get_pair_data()
                invalid_2 = CalculatePair(text, false_text_2, data_folder_name).get_pair_data()
                train_data.extend([valid, invalid, invalid_2])
                labels.extend([[1, 0], [0, 1], [0, 1]])
    train_data = np.array(list(train_data))
    train_labels = np.array(list(labels))
    with open('data' + postfix + '.npy', 'wb') as fl:
        np.save(fl, train_data)
        np.save(fl, train_labels)


def load_train_data():
    with open('data' + postfix + '.npy', 'rb') as fl:
        a = np.load(fl)
        b = np.load(fl)
    return a, b


def calculate(start=3): # function to process every step of preparing dataset and training model
    if start < 2:
        make_data_from_texts()
    if start < 3:
        make_pairs(-1)
    a, b = load_train_data()
    train_model(a, b)



