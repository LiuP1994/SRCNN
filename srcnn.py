import time
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import scipy
import pdb
import skimage.measure

'''To	 read	the image named butterfly_GT.bmp'''
butterfly_GT=scipy.misc.imread('/homes/pl304/Downloads/CW1_for_students/CW1_Handout_Template_code/tf-SRCNN/image/butterfly_GT.bmp')

'''To 	show	the	size	of	this	image'''
print(butterfly_GT.shape)

'''To	 convert	the	image	 from	 the	RGB	colour	space 	into	the	gray-scale	space'''
butterfly_GT1=scipy.misc.imread('/homes/pl304/Downloads/CW1_for_students/CW1_Handout_Template_code/tf-SRCNN/image/butterfly_GT.bmp',flatten=True)
print(butterfly_GT1.shape)

'''To	 shrink	the	current	image	by	3	times	with	bicubic	interpolation	algorithm'''
butterfly_GT1=scipy.misc.imresize(butterfly_GT1, 1/3)
'''To	 enlarge	the	current	image	by	3	times	with	bicubic	interpolation	algorithm'''
butterfly_GT1=scipy.misc.imresize(butterfly_GT1, 300)



def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 3: 
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def preprocess(path, scale=3):
  """
  Preprocess single image file 
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize 
    (3) Apply image file with bicubicinterpolation
  Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
  """
  image = imread(path, is_grayscale=True)
  label_ = modcrop(image, scale)

  # Must be normalized
  image = image / 255.
  label_ = label_ / 255.

  input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
  input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)

  return input_, label_

"""Set the image hyper parameters
"""
c_dim = 1
input_size = 255

"""Define the model weights and biases 
"""

# define the placeholders for inputs and outputs
inputs = tf.placeholder(tf.float32, [None, input_size, input_size, c_dim], name='inputs')

## ------ Add your code here: set the weight of three conv layers
# replace '0' with your hyper parameter numbers 
# conv1 layer with biases: 64 filters with size 9 x 9
# conv2 layer with biases and relu: 32 filters with size 1 x 1
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
weights = {
    'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1'),
    'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
    'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')
    }

biases = {
      'b1': tf.Variable(tf.zeros([64]), name='b1'),
      'b2': tf.Variable(tf.zeros([32]), name='b2'),
      'b3': tf.Variable(tf.zeros([1]), name='b3')
    }

"""Define the model layers with three convolutional layers
"""
## ------ Add your code here: to compute feature maps of input low-resolution images
# replace 'None' with your layers: use the tf.nn.conv2d() and tf.nn.relu()
# conv1 layer with biases and relu : 64 filters with size 9 x 9

conv1 = tf.nn.relu(tf.nn.conv2d(inputs, weights['w1'], strides=[1,1,1,1], padding='VALID') + biases['b1'])
##------ Add your code here: to compute non-linear mapping
# conv2 layer with biases and relu: 32 filters with size 1 x 1

conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['w2'], strides=[1,1,1,1], padding='VALID') + biases['b2'])
##------ Add your code here: compute the reconstruction of high-resolution image
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
conv3 = tf.nn.conv2d(conv2, weights['w3'], strides=[1,1,1,1], padding='VALID') + biases['b3']


"""Load the pre-trained model file
"""
model_path='/homes/pl304/Downloads/CW1_for_students/CW1_Handout_Template_code/tf-SRCNN/model/model.npy'
model = np.load(model_path, encoding='latin1').item()

##------ Add your code here: show the weights of model and try to visualisa
# variabiles (w1, w2, w3)
'''The first convolutional layer:To show the value of the 1 st filter in command window'''
print(model['w1'][:,:,:,0])
'''The first convolutional layer:To	show the bias of the 10 st filter in command window'''
print(model['b1'][9])
'''The second convolutional layer:To show the value of the 5 st filter in command window'''
print(model['w2'][:,:,:,4])
'''The second convolutional layer:To show the bias of the 6 st filter in command window'''
print(model['b2'][5])
'''The third convolutional layer:To show the value of the 1 st filter in command window'''
print(model['w3'][:,:,:,0])
'''The third convolutional layer:To show the bias of the 1 st filter in command window'''
print(model['b3'][0])

"""Initialize the model variabiles (w1, w2, w3, b1, b2, b3) with the pre-trained model file
"""
# launch a session
sess = tf.Session()

for key in weights.keys():
  sess.run(weights[key].assign(model[key]))

for key in biases.keys():
  sess.run(biases[key].assign(model[key]))

"""Read the test image
"""
blurred_image, groudtruth_image = preprocess('/homes/pl304/Downloads/CW1_for_students/CW1_Handout_Template_code/tf-SRCNN/image/butterfly_GT.bmp')


"""Show the test image"""
plt.imshow(groudtruth_image)

"""Show the low-resolution image"""
plt.imshow(blurred_image)
"""Run the model and get the SR image
"""
# transform the input to 4-D tensor
input_ = np.expand_dims(np.expand_dims(blurred_image, axis =0), axis=-1)

# run the session
# here you can also run to get feature map like 'conv1' and 'conv2'
output_ = sess.run(conv3, feed_dict={inputs: input_})

"""Show the output high-resolution image by SRCNN"""
plt.imshow(output_.reshape(255,255))
##------ Add your code here: save the blurred and SR images and compute the psnr
# hints: use the 'scipy.misc.imsave()'  and ' skimage.measure.compare_psnr()'
scipy.misc.imsave('SRCNN.bmp',output_.reshape(255,255))
scipy.misc.imsave('blurred_image.bmp',blurred_image)

print(skimage.measure.compare_psnr(groudtruth_image.astype('float32'),output_.reshape(255,255)))
print(skimage.measure.compare_psnr(groudtruth_image,blurred_image))






