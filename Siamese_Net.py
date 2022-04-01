## Define the second part of Siamese architecture
import numpy as np
import pandas as pd
import keras
import seaborn as sn
import matplotlib.pyplot as plt

from keras.layers import Dense
from keras.models import model_from_json
from keras.layers import Dropout, Input
from keras.layers import BatchNormalization, Concatenate

from keras.models import Model
from keras import regularizers
from keras import optimizers
from sklearn.metrics import confusion_matrix


json_file         = open('model_final_baseCNN_straighten.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model      = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_final_baseCNN_straighten.h5")
print("Loaded model from disk")

for layer in loaded_model.layers:
        layer.trainable = False

x = loaded_model.output
x = BatchNormalization()(x)
input_length = Input((1,))
input_area = Input((1,))
x = Concatenate()([x, input_length, input_area])

x = Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
predictions = Dense(24, activation='sigmoid')(x) 
finetune_model = Model(inputs=[loaded_model.input, input_length, input_area], outputs=predictions)
    
sgd = optimizers.SGD(lr=0.01, decay=0.01, momentum=0.99, nesterov=True)
finetune_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
finetune_model.summary()
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history5 = finetune_model.fit([X_train, X_length_train, X_area_train], y_train,
          batch_size=64,
          epochs=60,
          verbose=1, 
          validation_data=([X_valid, X_length_valid, X_area_valid], y_valid),
          callbacks=[model_checkpoint_callback])

# serialize model to JSON
model_json = finetune_model.to_json()
with open("model_final_siamese_straighten.json", "w") as json_file:
    json_file.write(model_json)
    
score5 = finetune_model.evaluate([X_test, X_length_test, X_area_test], y_test, verbose=0)
print('Test loss:', score5[0])
print('Test accuracy:', score5[1])

accuracy = history5.history['acc']
val_accuracy = history5.history['val_acc']
loss = history5.history['loss']
val_loss = history5.history['val_loss']
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

y_pred = finetune_model.predict([X_test, X_length_test, X_area_test])
y_pred = np.argmax(y_pred,1)
cm = confusion_matrix(np.argmax(y_test,1), y_pred)

df_cm = pd.DataFrame(cm, range(24),range(24))
plt.figure(figsize = (24,24))
sn.set(font_scale=1)
sn.heatmap(df_cm, annot=True,annot_kws={"size": 10})
