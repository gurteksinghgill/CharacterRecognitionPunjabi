import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Softmax
import numpy
from training-data import X_train, Y_train

# Set up neural network architecture
# model
model = Sequential([
  Dense(100, activation = 'tanh', input_shape = (100,)),
  Dense(100, activation = 'tanh'),
  Dense(5, activation = 'softmax')
])

model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, Y_train)

# Testing the model on the input examples provided in the the README file
test_sasa = np.array([
[1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
[0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
[0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
[1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
[1, 0, 1, 1, 0, 0, 1, 0, 0, 0],
[1, 1, 1, 1, 0, 0, 1, 0, 0, 0],
[0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
])

test_haha = np.array([
[0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
[0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
[0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
[0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
[0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
])

# Test the model
model.predict(test_sasa.flatten().reshape((1, 100)))
model.predict(test_haha.flatten().reshape((1, 100)))
