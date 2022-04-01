## Define ResNet50
import numpy as np
import pandas as pd
import seaborn as sn
import keras
import matplotlib.pyplot as plt

from keras.layers import Dense, Dropout, Flatten, Activation, Add, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from sklearn.metrics import confusion_matrix
from keras.layers import AveragePooling2D, Input
from keras.regularizers import l2
from keras.layers import Concatenate
from keras.models import Model
from keras import optimizers
from keras.initializers import glorot_uniform


def identity_block(X, f, filters, stage, block):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'same', name = conv_name_base + '2a', kernel_initializer = 'he_normal',kernel_regularizer=l2(0.0002))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = 'he_normal')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'same', name = conv_name_base + '2c', kernel_initializer = 'he_normal')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X
  

def convolutional_block(X, f, filters, stage, block, s = 2):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), padding='same', name = conv_name_base + '2a', kernel_initializer = 'he_normal', kernel_regularizer=l2(0.0002))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer= 'he_normal')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='same', name=conv_name_base + '2c', kernel_initializer= 'he_normal')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    
    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), padding='same', name = conv_name_base + '1', kernel_initializer = 'he_normal')(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X
  
  
  
def ResNet50(input_shape_1 = (32, 128, 1), input_shape_2 =(1,),input_shape_3 =(1,), classes = 24):   
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape_1)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), padding='same', name = 'conv1', kernel_initializer = 'he_normal')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), padding='same', strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL.
    X = AveragePooling2D((2, 2), name='avg_pool')(X)#, padding='same'

    # output layer
    X = Flatten()(X)
    input_length = Input(input_shape_2)
    input_area = Input(input_shape_3)
 
    X = Concatenate()([X, input_length, input_area])
    X = BatchNormalization()(X)
    X = Dropout(0.5)(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    # Create model
    model = Model(inputs = [X_input, input_length, input_area], outputs = X, name='ResNet50')

    return model
  
model = ResNet50(input_shape_1 = (32, 128, 1), input_shape_2 =(1,),input_shape_3 =(1,), classes = 24)
sgd   = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
checkpoint_filepath = 'model_final_ResNet50_straighten_length_area.h5'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history_ResNet50 = model.fit([X_train, X_length_train, X_area_train], y_train,
          batch_size=64,
          epochs=30,
          verbose=1,
          validation_data=([X_valid, X_length_valid, X_area_valid], y_valid),
          callbacks=[model_checkpoint_callback])



# serialize model to JSON
model_json = model.to_json()
with open("model_final_ResNet50_straighten_length_area.json", "w") as json_file:
    json_file.write(model_json)
     

score_ResNet50 = model.evaluate([X_test, X_length_test, X_area_test], y_test, verbose=0)
print('Test loss:', score_ResNet50[0])
print('Test accuracy:', score_ResNet50[1])

accuracy     = history_ResNet50.history['acc']
val_accuracy = history_ResNet50.history['val_acc']
loss         = history_ResNet50.history['loss']
val_loss     = history_ResNet50.history['val_loss']
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

y_pred = model.predict([X_test, X_length_test, X_area_test])
y_pred = np.argmax(y_pred,1)
cm     = confusion_matrix(np.argmax(y_test,1), y_pred)

df_cm = pd.DataFrame(cm, range(24),
                  range(24))
plt.figure(figsize = (24,24))
## Adjust label size
sn.set(font_scale=1)
## Adjust font size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 10})
