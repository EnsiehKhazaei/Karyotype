## Define Res-CRANN architecture 
import numpy as np
import pandas as pd
import seaborn as sn
import keras
import matplotlib.pyplot as plt


from keras.layers import Dense, Dropout, Flatten
from keras.layers import BatchNormalization
from sklearn.metrics import confusion_matrix
from keras.models import model_from_json
from keras.layers import Input
from keras.layers import Activation, Reshape, LSTM, TimeDistributed, Dot, Concatenate
from keras.models import Model


## Load pretrained ResNet50 model on our dataset to create ResCRANN model
json_file         = open('model_final_ResNet50_straighten.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model      = model_from_json(loaded_model_json)

## Load pretrained ResNet50 weights
loaded_model.load_weights("model_final_ResNet50_straighten.h5")
print("Loaded model from disk")
loaded_model.layers.pop() # Get rid of the classification layer
loaded_model.layers.pop() # Get rid of the dropout layer
loaded_model.layers.pop()

input_tensor = Input(shape=(32, 128, 1))  
input_length = Input((1,))
input_area  = Input((1,))

base_model = Model(inputs=[loaded_model.input, input_length, input_area],outputs=loaded_model.get_layer('activation_48').output)
for layer in loaded_model.layers[:-3]:
        layer.trainable=False

X = base_model([input_tensor, input_length, input_area])     
print(X.shape)
X = Reshape((2*5,2048))(X)

activations = LSTM(128, return_sequences=True, dropout=0.25)(X)

## Attention layer
attention = TimeDistributed(Dense(1, activation='tanh'))(activations) 
attention = Flatten()(attention)
attention = Activation('softmax')(attention)

activations = Dot(axes=1, normalize=True)([activations, attention])

activations  = Concatenate()([activations, input_length, input_area])
X = BatchNormalization()(activations)
X = Dropout(0.5)(X)
outputs = Dense(24,
                    activation='softmax',
                    kernel_initializer='he_normal')(X)

ResCRANN = Model([input_tensor, input_length, input_area], outputs)


ResCRANN.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001, epsilon=1e-8),
              metrics=['accuracy'])


ResCRANN.summary()

history_ResCRANN = ResCRANN.fit([X_train, X_length_train, X_area_train], y_train,
          batch_size=64,
          epochs=20,
          verbose=1,
          validation_data=([X_valid, X_length_valid, X_area_valid], y_valid))



## Serialize model to JSON
model_json = ResCRANN.to_json()
with open("model_final_ResCRNN.json", "w") as json_file:
    json_file.write(model_json)
    
## Serialize weights to HDF5
ResCRANN.save_weights("model_final_ResCRNN.h5")
print("Saved model to disk") 

score_ResCRANN = ResCRANN.evaluate([X_test, X_length_test, X_area_test], y_test, verbose=0)
print('Test loss:', score_ResCRANN[0])
print('Test accuracy:', score_ResCRANN[1])

accuracy     = history_ResCRANN.history['accuracy']
val_accuracy = history_ResCRANN.history['val_accuracy']
loss         = history_ResCRANN.history['loss']
val_loss     = history_ResCRANN.history['val_loss']
epochs       = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training loss')
plt.legend()
plt.show()

y_pred = ResCRANN.predict([X_test, X_length_test, X_area_test])
y_pred = np.argmax(y_pred,1)
cm = confusion_matrix(np.argmax(y_test,1), y_pred)

df_cm = pd.DataFrame(cm, range(24),
                  range(24))
plt.figure(figsize = (24,24))
sn.set(font_scale=1)
sn.heatmap(df_cm, annot=True,annot_kws={"size": 10})
