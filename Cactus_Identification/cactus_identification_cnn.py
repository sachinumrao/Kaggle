import os
import numpy as np
import pandas as pd

from scipy import ndimage
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# read training file ids and labels
train_label = pd.read_csv('~/Data/Cactus/train.csv', index_col=None)
train_label.head()

# Load training data filenames
train_path = '~/Data/Cactus/train/'
train_file_names = os.listdir('../../../Data/Cactus/train')


# load test data filenames
test_path = '~/Data/Cactus/test/'
test_file_names = os.system('../../../Data/Cactus/test/')

# load submission file
subm = pd.read_csv('~/Data/Cactus/sample_submission.csv', index_col=None)


# define data params
img_width = 32
img_height = 32
n = len(train_file_names)
channels = 3

# matrix to store train images in numpy
dataset = np.ndarray(shape=(n, img_height, img_width, channels), dtype=np.float32)
dataset.shape

y = np.ndarray(shape=(n,1), dtype=np.float32)
print(y.shape)

# load images into numpy array dataset
i = 0
for index,filename in enumerate(train_file_names):
    # read the image from disk
    img = load_img(train_path+filename)
    x = img_to_array(img)
    x = x.reshape(img_height, img_width, channels)
    x = (x-128.0) / 128.0
    
    dataset[index] = x
    
    y[index] = train_label[train_label['id']==filename]['has_cactus']
    
    
    i = i+1
    
    if i%500 == 0:
        print('Loaded image count: ',i)



#train test splitting 
X_train, X_val, y_train, y_val = train_test_split(dataset, y, test_size=0.25, random_state=33)

# define function for error metric measurement
def roc_auc(y_true, y_pred):
    
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


# Build a Keras convolutional model

#create model
model = Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', padding ='same', input_shape=(32,32,3)))
model.add(Conv2D(64, kernel_size=3, activation='relu', padding ='same', input_shape=(32,32,3)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=3, activation='relu', padding ='same'))
model.add(Conv2D(32, kernel_size=3, activation='relu', padding ='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(1, activation='sigmoid'))

#compile model using accuracy to measure model performance

nadam_optim = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
model.compile(optimizer=nadam_optim, loss='binary_crossentropy', metrics=['accuracy'])

batch_size = 32
epochs = 30

# train the model
model.fit(dataset, y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          )

# create test dataset numpy array
dataset_test = np.ndarray(shape=(len(test_file_names), img_height, img_width, channels), dtype=np.float32)

# load test dataset into numpy array
i = 0
for index,filename in enumerate(test_file_names):
    # read the image from disk
    img = load_img(test_path+filename)
    x = img_to_array(img)
    x = x.reshape(img_height, img_width, channels)
    #x = (x-128.0) / 128.0
    
    dataset_test[index] = x
    
    # y[index] = train_label[train_label['id']==filename]['has_cactus']
    
    
    i = i+1
    
    if i%500 == 0:
        print('Loaded image count: ',i)


# score the model on test dataset
y_hat = model.predict(dataset_test)

res = pd.DataFrame({'id': test_file_names, 'has_cactus' : y_hat[:,0]})
res.head()

# cretae submission dataframe
subm = subm.drop(['has_cactus'], axis=1)
subm = pd.merge(subm, res, on=['id'])

# stroe the submission dataframe
subm.to_csv('~/Data/Cactus/submission_1.csv', index=False)


