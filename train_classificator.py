import sys
sys.path.append('../tools')
import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Input,GlobalMaxPooling2D,Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import cv2
from imageGenerator import ImageGenerator



NUM_CLASSES = 2
TRAIN_PATH = './dataset'   # dataset path
SIZE = (640,304)  # original size (1920x912) -> (640x304)
BATCH_SIZE = 32  # size of the readed batches from generator, must fit on memory
VAL_SPLIT = 0.2  # fraction of the images used for validation



##########           Image generators           ##########


   
PREPROCESS = preprocess_input()  #funcion de preprocesado de imagenes
  
train_datagen = ImageDataGenerator(rescale=1./255,
                                    #rotation_range = 5,
                                    #shear_range=0.2,
                                    #zoom_range=0.2,
                                    #width_shift_range=0.0,
                                    #height_shift_range=0.0,
                                    #fill_mode='wrap',
                                    #brightness_range=None,
                                    #horizontal_flip=True,
                                    #vertical_flip=True,
                                    validation_split=0.2,# set validation split
                                    preprocessing_function=PREPROCESS) 


train_generator = train_datagen.flow_from_directory(TRAIN_PATH,
                                                    subset='training',
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    shuffle=True,
                                                    target_size=(512,None, 3)
                                                    )

validation_generator = train_datagen.flow_from_directory(TRAIN_PATH, # same directory as training data
                                                         subset='validation',
                                                         batch_size=BATCH_SIZE,
                                                         class_mode='categorical',
                                                         shuffle=True,
                                                         target_size=(None,None,3)
                                                    )


"""

train_generator = ImageGenerator(TRAIN_PATH, batch_size=BATCH_SIZE, shuffle=True, max_dimension=MAX_SIZE)
validation_generator = ImageGenerator(TRAIN_PATH, batch_size=BATCH_SIZE, max_dimension=MAX_SIZE)

train_dataset = tf.data.Dataset.from_generator(train_generator,
     (tf.float32, tf.int32),
    (tf.TensorShape([None,None,None,3]), tf.TensorShape([None])))

validation_dataset = tf.data.Dataset.from_generator(validation_generator,
     (tf.float32, tf.int32),
    (tf.TensorShape([None,None,None,3]), tf.TensorShape([None])))

"""

##########          CNN Construction           ##########

inputs = Input(shape=(None,None,3))
net = MobileNetV2(include_top=False, alpha=0.35, weights='imagenet', input_tensor=inputs, classes=NUM_CLASSES)
net = GlobalMaxPooling2D()(net.output)
outputs = Dense(NUM_CLASSES,activation='softmax')(net)
model = Model(inputs=inputs,outputs=outputs)
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])

model_config = model.get_config()
print(model.summary())

##########           Training           ##########

history = model.fit(train_generator,
                    validation_data = validation_generator,
                    #validation_split = 0.2,
                    epochs=10,
                    verbose=1)

save_path = './trained_models/classification_model.hdf5'
print('Saving model to:', save_path)
model.save(save_path)



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

