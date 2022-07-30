# Cell 2
#You can use a gpu to speed up training time. In colab go to runtime > "Change runtime type" > and select gpu
import tensorflow as tf
tf.test.gpu_device_name()

# Cell 4
# from google.colab import drive
import os
import shutil 
# drive.mount('/content/drive')
!unzip "MerchData.zip" -d "./"

# feel free to edit this cell to work with your local or colab directory
WORKING_DIR = '.'
TRAIN_DIR = './train'
VAL_DIR = './val'

shutil.move(os.path.join(WORKING_DIR,'MerchData','MathWorks Cap'), os.path.join(TRAIN_DIR, 'Cap'))
shutil.move(os.path.join(WORKING_DIR,'MerchData','MathWorks Cube'), os.path.join(TRAIN_DIR, 'Cube'))
shutil.move(os.path.join(WORKING_DIR,'MerchData','MathWorks Playing Cards'), os.path.join(TRAIN_DIR, 'PlayingCard'))
shutil.move(os.path.join(WORKING_DIR,'MerchData','MathWorks Screwdriver'), os.path.join(TRAIN_DIR, 'Screwdriver'))
shutil.move(os.path.join(WORKING_DIR,'MerchData','MathWorks Torch'), os.path.join(TRAIN_DIR, 'Torch'))
os.rmdir(os.path.join(WORKING_DIR,'MerchData'))

# Cell 6
from glob import glob
from sklearn.model_selection import train_test_split

## Randomly Split data into training and validation
# This code randomly chooses 75% and 25% of the MerchData to be set 
# as training and validation data, creates a train and a val folder.
caps = glob(TRAIN_DIR + '/Cap/*.jpg')
cubes = glob(TRAIN_DIR + '/Cube/*.jpg')
cards = glob(TRAIN_DIR + '/PlayingCard/*.jpg')
screwdrivers = glob(TRAIN_DIR + '/Screwdriver/*.jpg')
torches = glob(TRAIN_DIR + '/Torch/*.jpg')

# The *_val variables contain the filenames of the files chosen for validation. 
caps_train, caps_val = train_test_split(caps, test_size=0.25)
cubes_train, cubes_val = train_test_split(cubes, test_size=0.25)
cards_train, cards_val = train_test_split(cards, test_size=0.25)
sd_train, sd_val = train_test_split(screwdrivers, test_size=0.25)
torches_train, torches_val = train_test_split(torches, test_size=0.25)


# This code moves the validation files out of the train directory and over to 
# the val directory.
os.makedirs(os.path.join(VAL_DIR,'Cap'))
for file in caps_val:
  os.rename(file, file.replace('train','val'))

os.makedirs(os.path.join(VAL_DIR,'Cube'))
for file in cubes_val:
  os.rename(file, file.replace('train','val'))

os.makedirs(os.path.join(VAL_DIR,'PlayingCard'))
for file in cards_val:
  os.rename(file, file.replace('train','val'))

os.makedirs(os.path.join(VAL_DIR,'Screwdriver'))
for file in sd_val:
  os.rename(file, file.replace('train','val'))

os.makedirs(os.path.join(VAL_DIR,'Torch'))
for file in torches_val:
  os.rename(file, file.replace('train','val'))

# Cell 7
# Jupyter keeps checkpoint files that may screw up the ImageDataGenerator class
# So we delete those here. 
# !rm -rf train/.ipynb_checkpoints
# !rm -rf val/.ipynb_checkpoints

# Cell 10
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input

WIDTH = 299
HEIGHT = 299
BATCH_SIZE = 11

# data prep
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=90,
    zoom_range=[1,2],
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=90,
    zoom_range=[1,2],
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(HEIGHT, WIDTH),
		batch_size=BATCH_SIZE,
		class_mode='categorical')
    
validation_generator = validation_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# Cell 13
from keras.models import Model
from keras.applications.resnet import ResNet50, preprocess_input

base_model = ResNet50(weights='imagenet', include_top=False)

# Cell 15
from keras.utils.vis_utils import plot_model
plot_model(base_model, to_file=os.path.join(WORKING_DIR,'model_plot.png'), show_shapes=True, show_layer_names=True)

# Cell 18
from keras.layers import Dense, GlobalAveragePooling2D, Dropout


CLASSES = 5
# These layers were removed by setting include_top=false, but we still want them
x = base_model.output                          # The output layer of the googleNet
x = GlobalAveragePooling2D(name='avg_pool')(x) # This layer is used to downsample the features
x = Dropout(0.4)(x)                            # This layer is used to enforce feature redundancy 

# This layer specifies our classifier output
predictions = Dense(CLASSES, activation='softmax')(x)
#
model = Model(inputs=base_model.input, outputs=predictions)



# Cell 20
for layer in base_model.layers:
    layer.trainable = False # Set this to True if you want to re-train the entire network.

# Cell 22
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Cell 24
EPOCHS = 30
BATCH_SIZE = 11
STEPS_PER_EPOCH = 5
VALIDATION_STEPS = 1
MODEL_FILE = os.path.join(WORKING_DIR,'model')

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS)
  
model.save(MODEL_FILE)

# Cell 26
import matplotlib.pyplot as plt
acc = history.history['accuracy']
loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)

plt.plot(epochs, val_acc,'r')
plt.plot(epochs, acc,'r.')
plt.title("Training Accuracy")

plt.figure()
plt.plot(epochs, val_loss,'r')
plt.plot(epochs,loss,'r.')
plt.title("Training Loss")
plt.show()

# Cell 29
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

# Predict pre-processes the input so that it matches the format expected by the 
# network, then runs model.predict which provides the class predictions
def predict(model, img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]

# We plot the original image
plt.figure()
img = image.load_img('MerchDataTest.jpg', target_size=(HEIGHT, WIDTH))
preds = predict(load_model(MODEL_FILE), img)
plt.imshow(img)

# We can visualize the predicted likelihood of each class
plt.figure()
labels = ["Cap", "Cube","Cards","Screwdriver","Torch"]
plt.barh(range(5),preds,alpha=0.5)
plt.yticks(range(5),labels)
plt.xlabel('Probability')
plt.xlim(0,1)
plt.tight_layout()
plt.show()

