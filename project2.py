import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras import optimizers
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import models
from keras import layers
from keras.models import Model
from keras.models import load_model
import tensorflow as tf
import numpy as np
from keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95

learning_rate = 1e-3
lr_decay = 1e-6
epochs =200
num_classes = 2
train_batchsize = 64
val_batchsize = 64
nb_train_samples=296
nb_validation_samples=74
epochs_to_wait_for_improve=100
data_path = "/home/CAP5627-3/test_folder/"
train_dir = data_path + "Training/"
validation_dir = data_path + 'Validation/'

fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(160,160,1)))
fashion_model.add(BatchNormalization())
fashion_model.add(ReLU())
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(BatchNormalization())
fashion_model.add(ReLU())
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(BatchNormalization())
fashion_model.add(ReLU())                
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.4))
fashion_model.add(Flatten())
fashion_model.add(Dense(16, activation='linear'))
fashion_model.add(BatchNormalization())
fashion_model.add(ReLU())         
fashion_model.add(Dropout(0.3))
fashion_model.add(Dense(num_classes, activation='softmax'))

fashion_model.compile(loss=keras.losses.binary_crossentropy, 
                      optimizer=keras.optimizers.RMSprop(lr=learning_rate, decay=lr_decay),metrics=['accuracy'])
fashion_model.summary()

train_datagen = image.ImageDataGenerator(rescale=1./255,featurewise_std_normalization=True, featurewise_center=True,
      samplewise_center=True,
      shear_range=30,
      zoom_range=30,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      zca_whitening=True,
      horizontal_flip=True,vertical_flip=True)
 
validation_datagen = image.ImageDataGenerator(rescale=1./255, featurewise_std_normalization=True, featurewise_center=True,
      samplewise_center=True,
      shear_range=30,
      zoom_range=30,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      zca_whitening=True,
      horizontal_flip=True,vertical_flip=True)

train_generator = train_datagen.flow_from_directory(
        train_dir,batch_size=train_batchsize,class_mode='categorical', shuffle=True, target_size=(160, 160), color_mode='grayscale')

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,batch_size=val_batchsize,class_mode='categorical',shuffle=False, target_size=(160, 160), color_mode='grayscale')

#callbacks
early_stopping_callback = EarlyStopping(
        monitor='val_loss', patience=epochs_to_wait_for_improve)
checkpoint_callback = ModelCheckpoint(
       "weights_best5.h5", monitor='val_acc', mode='max', verbose=1, save_best_only=True, save_weights_only=False)

# fine-tune the model
fashion_model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples, verbose=2, callbacks=[early_stopping_callback, checkpoint_callback])

model = load_model('weights_best5.h5')    

test_images=3495
test_dir="/home/CAP5627-3/test_folder/Testing/"
val_batchsize = 233
nb_test_samples=test_images//val_batchsize

test_datagen = image.ImageDataGenerator(rescale=1./255,featurewise_center=True,featurewise_std_normalization=True,samplewise_center=True,zca_whitening=True)

test_generator = test_datagen.flow_from_directory(
        test_dir,target_size=(160, 160),batch_size=val_batchsize,class_mode='categorical',shuffle=False, color_mode='grayscale')


scores = fashion_model.evaluate_generator(test_generator, steps=nb_test_samples) # testing images
print("Accuracy = ", scores[1])
print("loss = ", scores[0])

#Confution Matrix and Classification Report
Y_pred = fashion_model.predict_generator(test_generator, test_images// val_batchsize)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
target_names = ['Pain', 'No Pain']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))