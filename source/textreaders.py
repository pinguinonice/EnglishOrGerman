
import numpy as np
import time as time
import mmap
import csv
import keras as kr






def mapcount(filename): #returns the number of lines in a txt file
    f = open(filename, "r+")
    buf = mmap.mmap(f.fileno(), 0)
    lines = 0
    readline = buf.readline
    while readline():
        lines += 1
    return lines


def readTxt(filename,cls):
    input_file.readline()  # skip first line
    N = mapcount(filename)  # lines in foo.txt
    inputs = np.zeros((N, 16))
    outputs = np.ones((len(words)-1, 1))*cls
    outputs = kr.utils.to_categorical(outputs)

    N = 7
    print N

    i = 0
    for line in input_file:
        word = line.strip()
        for j in range(0, len(word)):
            print word[j], ord(word[j])
            if j < 15:
                inputs[i, j] = ord(word[j])
            else:
                break
        # time.sleep(1)
        print '\n'
        i = i+1
        if i == N:
            break

    input_file.close()

    print inputs
    print inputs[:7, 0:15]
    return inputs, outputs


def loadData(csvpath):
    print('Importing...')
    words = list(csv.reader(open(csvpath)))
    # defining input word length max =16
    inputs_ger = np.zeros((len(words)-1, 16))
    inputs_eng = np.zeros((len(words)-1, 16))
    outputs_ger = np.ones((len(words)-1, 1))
    outputs_eng = np.zeros((len(words)-1, 1))
    for i in range(0, 999):  # len(words)-1
        temp_word_ger = list(words[i][1])
        temp_word_eng = list(words[i][2])
        # decompose words into numbers and add to two lists
        for j in range(0, len(temp_word_ger)):
            inputs_ger[i, j] = ord(temp_word_ger[j])
        for j in range(0, len(temp_word_eng)):
            inputs_eng[i, j] = ord(temp_word_eng[j])
    inputs = np.concatenate((inputs_ger, inputs_eng), axis=0)
    outputs = np.concatenate((outputs_ger, outputs_eng), axis=0)
    # Encode the category integers as binary categorical vairables.
    outputs = kr.utils.to_categorical(outputs)
    # Split the input and output data sets into training and test subsets.
    inds = np.random.RandomState(seed=42).permutation(len(inputs))
    train_inds, test_inds = np.array_split(inds, 2)
    inputs_train, outputs_train = inputs[train_inds], outputs[train_inds]
    inputs_test,  outputs_test = inputs[test_inds],  outputs[test_inds]
    print 'inputs_train: \n', inputs_train
    print 'outputs_train: \n', outputs_train
    return [inputs_train, outputs_train, inputs_test, outputs_test]