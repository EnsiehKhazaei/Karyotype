## Define SECNN architecture 
import numpy as np
import pandas as pd
import seaborn as sn
import keras
import matplotlib.pyplot as plt


from sklearn.metrics import confusion_matrix
from keras.layers import Input, Flatten
from keras.layers import Dense, Conv2D, BatchNormalization, Dropout, multiply, Concatenate
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model

def squeeze_excite_block(input,ch, ratio=16):
    init = input
    filters = ch

    se = GlobalAveragePooling2D()(init)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)

    x = multiply([init, se])
    return x

  
input_main = Input(shape=(32,128,1))

X = Conv2D(64, kernel_size=(3, 3), activation='relu')(input_main)
X =  squeeze_excite_block(X, ch=64, ratio=4)
X = BatchNormalization()(X)

X = Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same')(X)
X = squeeze_excite_block(X, ch=64, ratio=4)
X = BatchNormalization()(X)
X = MaxPooling2D(pool_size=(2, 2), padding='same')(X)
X = Dropout(0.25)(X)

X = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(X)
X = squeeze_excite_block(X, ch=32, ratio=4)
X = BatchNormalization()(X)

X = Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same')(X)
X = squeeze_excite_block(X, ch=32, ratio=4)
X = BatchNormalization()(X)
X = MaxPooling2D(pool_size=(2, 2), padding='same')(X)
X = Dropout(0.25)(X)

X = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(X)
X = squeeze_excite_block(X, ch=16, ratio=4)
X = BatchNormalization()(X)

X = Conv2D(16, kernel_size=(5, 5), activation='relu', padding='same')(X)
X = squeeze_excite_block(X, ch=16, ratio=4)
X = BatchNormalization()(X)
X = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(X)
X = Dropout(0.25)(X)

X = Flatten()(X)

input_length = Input(shape=(1,))
input_area = Input(shape=(1,))
 
X = Concatenate()([X, input_length, input_area])

X = Dense(256, activation='relu')(X)

X = BatchNormalization()(X)
X = Dropout(0.5)(X)

output = Dense(24, activation='softmax')(X)
SECNN  = Model(inputs=[input_main, input_length, input_area], outputs=[output])

SECNN.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

SECNN.summary()

history_SECNN = SECNN.fit([X_train, X_length_train, X_area_train], y_train,
          batch_size=64,
          epochs=60,
          verbose=1,
          validation_data=([X_valid, X_length_valid, X_area_valid], y_valid))

model_json = SECNN.to_json()
with open("model_final_SECNN.json", "w") as json_file:
    json_file.write(model_json)
    
SECNN.save_weights("model_final_SECNN.h5")
print("Saved model to disk") 

score_SECNN = SECNN.evaluate([X_test, X_length_test, X_area_test], y_test, verbose=0)
print('Test loss:', score_SECNN[0])
print('Test accuracy:', score_SECNN[1])

accuracy     = history_SECNN.history['accuracy']
val_accuracy = history_SECNN.history['val_accuracy']
loss         = history_SECNN.history['loss']
val_loss     = history_SECNN.history['val_loss']
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

y_pred = SECNN.predict([X_test, X_length_test, X_area_test])
y_pred = np.argmax(y_pred,1)
cm = confusion_matrix(np.argmax(y_test,1), y_pred)

df_cm = pd.DataFrame(cm, range(24),
                  range(24))
plt.figure(figsize = (24,24))
sn.set(font_scale=1)
sn.heatmap(df_cm, annot=True,annot_kws={"size": 10})
