## Define the second part of Siamese architecture
import numpy as np
import pandas as pd
import seaborn as sn
import keras
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from keras.layers import Dense, BatchNormalization, Dropout
from keras.models import model_from_json
from keras.models import Model
from keras import optimizers

# load json and create model
json_file         = open('model_final_baseCNN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model      = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_final_baseCNN.h5")
print("Siamese_base is loaded from disk")

for layer in loaded_model.layers:
        layer.trainable = False

x = loaded_model.output
x = BatchNormalization()(x)

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
predictions = Dense(24, activation='sigmoid')(x) 
finetune_model = Model(inputs=loaded_model.input, outputs=predictions)
    
sgd = optimizers.SGD(lr=0.1, decay=0.01, momentum=0.99, nesterov=True)
finetune_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
finetune_model.summary()

checkpoint_filepath="model_final_siamese.h5"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)


history5 = finetune_model.fit(X_train, y_train,
          batch_size=64,
          epochs=60,
          verbose=1, 
          validation_data=(X_valid, y_valid),
          callbacks=[model_checkpoint_callback])


model_json = finetune_model.to_json()
with open("model_final_siamese.json", "w") as json_file:
    json_file.write(model_json)
    
score5 = finetune_model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score5[0])
print('Test accuracy:', score5[1])

accuracy = history5.history['accuracy']
val_accuracy = history5.history['val_accuracy']
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

y_pred = finetune_model.predict(X_test)
y_pred = np.argmax(y_pred,1)
cm = confusion_matrix(np.argmax(y_test,1), y_pred)

df_cm = pd.DataFrame(cm, range(24),range(24))
plt.figure(figsize = (24,24))
sn.set(font_scale=1)
sn.heatmap(df_cm, annot=True,annot_kws={"size": 10})
