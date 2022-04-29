## Define CRANN architecture
import numpy as np
import pandas as pd
import seaborn as sn
import keras
import matplotlib.pyplot as plt


from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from sklearn.metrics import confusion_matrix
from keras.layers import Input
from keras.layers import Activation, Reshape, LSTM, TimeDistributed, multiply, Dot, Concatenate
from keras.models import Model

def squeeze_excite_block(input,ch, ratio=16):

    init = input
    filters = ch

    se = GlobalAveragePooling2D()(init)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)

    x = multiply([init, se])
    return x

inputs = Input(shape=(32,128,1))
X = Conv2D(64, kernel_size=(3, 3), activation='relu')(inputs)
X =  squeeze_excite_block(X, ch=64, ratio=4)
X = BatchNormalization()(X)

X = Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same')(X)
X =  squeeze_excite_block(X, ch=64, ratio=4)
X = BatchNormalization()(X)
X = MaxPooling2D(pool_size=(2, 2), padding='same')(X)
X = Dropout(0.25)(X)

X = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(X)
X =  squeeze_excite_block(X, ch=32, ratio=4)
X = BatchNormalization()(X)

X = Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same')(X)
X =  squeeze_excite_block(X, ch=32, ratio=4)
X = BatchNormalization()(X)
X = MaxPooling2D(pool_size=(2, 2), padding='same')(X)
X = Dropout(0.25)(X)

X = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(X)
X =  squeeze_excite_block(X, ch=16, ratio=4)
X = BatchNormalization()(X)

X = Conv2D(16, kernel_size=(5, 5), activation='relu', padding='same')(X)
X =  squeeze_excite_block(X, ch=16, ratio=4)
X = BatchNormalization()(X)
X = Dropout(0.25)(X)

X = Reshape((8, 32*16))(X)

activations = LSTM(128, return_sequences=True, dropout=0.25)(X)

# attention
attention = TimeDistributed(Dense(1, activation='tanh'))(activations) 
attention = Flatten()(attention)
attention = Activation('softmax')(attention)

activations = Dot(axes=1, normalize=True)([activations, attention])
input_length = Input((1,))
input_area   = Input((1,))
X = Concatenate()([activations, input_length, input_area])
X = BatchNormalization()(X)
X = Dropout(0.5)(X)

outputs = Dense(24,
                    activation='softmax',
                    kernel_initializer='he_normal')(X)


model = Model(inputs=[inputs, input_length, input_area], outputs=outputs)

  
CRANN = model

CRANN.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001, epsilon=1e-8),
              metrics=['accuracy'])


CRANN.summary()
history4 = CRANN.fit([X_train, X_length_train, X_area_train], y_train,
          batch_size=64,
          epochs=60,
          verbose=1,
          validation_data=([X_valid, X_length_valid, X_area_valid], y_valid))

# serialize model to JSON
model_json = CRANN.to_json()
with open("model_final_CRANN_straighten.json", "w") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
CRANN.save_weights("model_final_CRANN_straighten.h5")
print("Saved model to disk") 

score4 = CRANN.evaluate([X_test, X_length_test, X_area_test], y_test, verbose=0)
print('Test loss:', score4[0])
print('Test accuracy:', score4[1])

accuracy = history4.history['acc']
val_accuracy = history4.history['val_acc']
loss = history4.history['loss']
val_loss = history4.history['val_loss']
epochs = range(len(accuracy))

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

y_pred = CRANN.predict([X_test, X_length_test, X_area_test])
y_pred = np.argmax(y_pred,1)
cm = confusion_matrix(np.argmax(y_test,1), y_pred)

df_cm = pd.DataFrame(cm, range(24),
                  range(24))
plt.figure(figsize = (24,24))
sn.set(font_scale=1)
sn.heatmap(df_cm, annot=True,annot_kws={"size": 10})
