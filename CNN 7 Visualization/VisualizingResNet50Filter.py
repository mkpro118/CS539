# Cell 1
# This program is from the blog post:  
# How to Visualize Filters and Feature Maps in Convolutional Neural Networks
# by Jason Brownlee on May 6, 2019 in Deep Learning for Computer Vision
# https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
# Modified to use Resnet50
#
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims

# load the model
model = VGG16(weights='imagenet')
# summarize the model
model.summary()

# Cell 3
# summarize feature map shapes
for i in range(len(model.layers)):
    layer = model.layers[i]
    # check for convolutional layer
    if '_conv' not in layer.name:
        continue
    # summarize output shape
    print(i, layer.name, layer.output.shape)

# Cell 4
# retrieve weights from the second hidden layer
filters, biases = model.layers[2].get_weights()

# Cell 5
# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# Cell 6
print(filters[:,:,0,0])

# Cell 7
# plot first six filters
n_filters, ix = 6, 1
for i in range(n_filters):
    # get the filter
    f = filters[:, :, :, i]
    # plot each channel separately
    for j in range(3):
        # specify subplot and turn of axis
        ax = pyplot.subplot(n_filters, 3, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        pyplot.imshow(f[:, :, j], cmap='gray')
        ix += 1
# show the figure
pyplot.show()

# Cell 8
print(f_min, f_max)

# Cell 9
# redefine model to output right after the second convolutional layer
# this is layer #7
model = Model(inputs=model.inputs, outputs=model.layers[7].output)
# this model is much smaller than the VGG16 model, but still uses 
# the same weights (filters) in the first convolutional layer as 
# the VGG16 model.
model.summary()

# Cell 10
# load the image with the required shape
img = load_img('StopSign.jpg', target_size=(224, 224))
# convert the image to an array
# the image PIL object needs to be converted to a NumPy array of 
# pixel data and expanded from a 3D array to a 4D array with the 
# dimensions of [samples, rows, cols, channels], where we only 
# have one sample.
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
# See https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/preprocess_input
img = preprocess_input(img)

# Cell 11
# get feature map for first hidden layer
feature_maps = model.predict(img)

# Cell 12
# The result will be a feature map with 224x224x64. 
# We can plot all 64 two-dimensional images as an 8×8 square of images.
# plot the output from each block
# plot all 64 maps in an 8x8 squares
square = 8
ix = 1
for _ in range(square):
	for _ in range(square):
		# specify subplot and turn of axis
		ax = pyplot.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
		ix += 1
# show the figure
pyplot.show()

