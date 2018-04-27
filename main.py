import csv
import numpy as np
import keras as kr
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.optimizers import RMSprop
from keras.utils import plot_model
from source.textreaders import mapcount, loadData, readTxt

# define load wordlist as csv





# import csv
[inputs_train, outputs_train, inputs_test, outputs_test] = loadData('data_input/words.csv')

[inputs_ger, outputs_ger] = readTxt('data_input/deutsch.txt',1)



# build the model:
print('Build model...')
model = Sequential()
model.add(Dense(50, input_shape=(16,)))
kr.layers.LSTMCell(16, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True,
                   kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1)
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

# solving algorithem
optimizer = RMSprop(lr=0.1)
optimizer = "adam"

# compile model
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# draw structure
plot_model(model, to_file='data_output/model.png', show_shapes=True)

# Fit the model using our training data.
model.fit(inputs_train, outputs_train, epochs=10000, batch_size=1000, verbose=1)
loss, accuracy = model.evaluate(inputs_test, outputs_test, verbose=1)

# Output the accuracy of the model.
print("\n\nLoss: %6.4f\tAccuracy: %6.4f" % (loss, accuracy))

# Predict the class of a single word.
n = 2
print "".join([chr(int(letter)) for letter in inputs_test[n]])
print inputs_test[n].astype('string')
prediction = np.around(model.predict(np.expand_dims(inputs_test[n], axis=0))).astype(np.int)[0]
print("Actual: %s\tEstimated: %s" % (outputs_test[n].astype(np.int), prediction))
#print("That means it's a %s" % outputs_vals[prediction.astype(np.bool)][10])

# Save the model to a file for later use.
model.save("data_output/iris_nn.h5")
