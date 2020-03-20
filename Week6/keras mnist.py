import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import plot_model
import numpy as np


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

image_index = 3
print(y_train[image_index])
#plt.imshow(x_train[image_index], cmap='Greys')
#plt.show()
print(x_train.shape)        # dang cua tap du lieu (60000,28,28) 60000 du lieu, kich thuoc 28*28

x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28,28,1)
input_shape = (28,28,1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

print('x_train shape: ',x_train.shape)
print('Number of images in x_train: ', x_train.shape[0])
print('Number of images in x_test: ', x_test.shape[0])

# xay dung mang convolutional nerual

model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))    #dau ra co 128 neuron
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))   # dau ra co 10 reuron tuong ung voi 0->9

# bien dich model

optimizer = Adam(lr=0.02)
model.compile(optimizer = optimizer, loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])   # do chinh xac

plot_model(model=model,  to_file='model.png', show_layer_names=True, show_shapes=True)

history = model.fit(x=x_train, y=y_train,validation_split=0.25, epochs=50, batch_size=16, verbose=1)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy   Learning rate = 0.01')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss   Learning rate = 0.01')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

result = model.evaluate(x_test, y_test)         # result[0] = loss, result[1] = accuracy

output = model.predict(x_train[0:1])
#print(result)
        

plt.imshow(x_test[0].reshape(28,28), cmap='Greys')
print(output.argmax())