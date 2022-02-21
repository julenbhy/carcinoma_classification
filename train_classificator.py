import sys
sys.path.append('../tools')
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import cv2



NUM_CLASSES = 2
TRAIN_PATH = './train_dataset'   # dataset path
MAX_SIZE = 512  # the maximum size for images, if grater --> downsample
BATCH_SIZE = 32  # size of the readed batches from generator, must fit on memory
VAL_SPLIT = 0.2  # fraction of the images used for validation


##########           Image generators           ##########
   
PREPROCESS = None #funcion de preprocesado de imagenes
    
train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range = 5,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    width_shift_range=0.0,
                                    height_shift_range=0.0,
                                    fill_mode='wrap',
                                    brightness_range=None,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    validation_split=0.2,# set validation split
                                    preprocessing_function=PREPROCESS) 


train_generator = train_datagen.flow_from_directory(TRAIN_PATH,
                                                    subset='training',
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categprocal',
                                                    shuffle=True,) # set as training data

validation_generator = train_datagen.flow_from_directory(TRAIN_PATH, # same directory as training data
                                                         subset='validation',
                                                         batch_size=BATCH_SIZE,
                                                         class_mode='categorical',
                                                         shuffle=True,) # set as validation data



##########          CNN Construction           ##########

inputs = Input(shape=(None,None,3))
net = MobileNetV2(include_top=False, alpha=0.35, weights='imagenet', input_tensor=inputs, classes=NUM_CLASSES = 2)
net = GlobalMaxPooling2D()(net.output)
outputs = Dense(n_classes,activation='softmax')(net)
model = Model(inputs=inputs,outputs=outputs)
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])


##########           Training           ##########

history = model.fit(train_generator,
                    validation_data=validation_generator,
                    epochs=10,
                    verbose=1,
                    workers=2,
                    max_queue_size=20)

model_name = './trained_models/classification_model.hdf5'
print('Saving model to:', model_name)
model.save(model_name)



##########           Model evaluation           ##########

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['iou_score']
val_acc = history.history['val_iou_score']
plt.plot(epochs, acc, 'y', label='Training IOU')
plt.plot(epochs, val_acc, 'r', label='Validation IOU')
plt.title('Training and validation IOU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend()
plt.show()

max_iou = max(val_acc)
print ("Maximum accuracy reached: ", max_iou)
max_index = val_acc.index(max_iou)
print("Maximum accuracy reached at epoch: ",max_index+1)

