import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Input
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras import optimizers

left_input = Input((32,128,1))
right_input = Input((32,128,1))
# Convolutional Neural Network
CNN = Sequential()
CNN.add(Conv2D(64, (3,3), activation='relu', input_shape=(32,128,1)))
CNN.add(BatchNormalization())
CNN.add(MaxPooling2D(pool_size=(2, 2)))

CNN.add(Conv2D(128, (3,3), activation='relu'))
CNN.add(BatchNormalization())
CNN.add(MaxPooling2D(pool_size=(2, 2)))
CNN.add(Dropout(0.25))

CNN.add(Flatten())
CNN.add(Dense(512, activation='sigmoid',
                   kernel_regularizer=l2(1e-3)))
    
# Generate the encodings (feature vectors) for the two images
encoded_l = CNN(left_input)
encoded_r = CNN(right_input)
    
# Add a customized layer to compute the absolute difference between the encodings
L1_layer = Lambda(lambda tensors:K.abs((tensors[0] - tensors[1])))
L1_distance = L1_layer([encoded_l, encoded_r])
    
# Add a dense layer with a sigmoid unit to generate the similarity score
prediction = Dense(1,activation='sigmoid')(L1_distance)
    
# Connect the inputs with the outputs
siamese = Model(inputs=[left_input,right_input],outputs=prediction)
sgd = optimizers.SGD(lr=0.01, decay=0.001, momentum=0.99, nesterov=True)
siamese.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

CNN.summary()
history_siamese = siamese.fit(X_train_prime, y_train_prime,
          batch_size=64,
          epochs=35,
          verbose=1)



# serialize model to JSON
model_json = CNN.to_json()
with open("model_final_baseCNN.json", "w") as json_file:
    json_file.write(model_json)
    

CNN.save_weights("model_final_baseCNN.h5")

accuracy = history_siamese.history['accuracy']
loss = history_siamese.history['loss']
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.title('Training accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.title('Training loss')
plt.legend()
plt.show()
