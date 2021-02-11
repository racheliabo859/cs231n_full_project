# Run some setup code for this notebook.

import random
import numpy as np
from cs231n.classifiers import KNearestNeighbor
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

print('here we go..')
# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
# %matplotlib inline
# plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'
#
# # Some more magic so that the notebook will reload external python modules;
# # see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# %load_ext autoreload
# %autoreload 2# -*- coding: utf-8 -*-

# Load the raw CIFAR-10 data.
cifar10_dir = 'C:/virtualenvs/assignment1/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
# classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# num_classes = len(classes)
# samples_per_class = 7
# for y, cls in enumerate(classes):
#     idxs = np.flatnonzero(y_train == y)
#     idxs = np.random.choice(idxs, samples_per_class, replace=False)
#     for i, idx in enumerate(idxs):
#         plt_idx = i * num_classes + y + 1
#         plt.subplot(samples_per_class, num_classes, plt_idx)
#         plt.imshow(X_train[idx].astype('uint8'))
#         plt.axis('off')
#         if i == 0:
#             plt.title(cls)
# plt.show()

# from scipy.misc import imread, imsave, imresize
# import matplotlib.pyplot as plt
# img = (X_train[0,:,:,]).astype('uint8')
# print (img.dtype, img.shape)
# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.show()


# Subsample the data for more efficient code execution in this exercise
num_training = 500
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 50
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

# remove unused variables
img = None
cifar10_dir = None
cls = None
idx = None
idxs = None
mask = None
# num_test = None
num_classes = None
# num_training = None
plt_idx = None
y = None
i = None
samples_per_class = None

# from cs231n.classifiers import KNearestNeighbor
# Create a kNN classifier instance. 
# Remember that training a kNN classifier is a noop: 
# the Classifier simply remembers the data and does no further processing 
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
# dists = classifier.compute_distances_two_loops(X_test)
dists2 = classifier.compute_distances_one_loop(X_test)


# dists = classifier.compute_distances_two_loops(X_test)
dists = classifier.compute_distances_one_loop(X_test)
classification = {'real': y_test, 'pred': classifier.predict_labels(dists, 5)}
print(dists.shape)

# accuracy
sum(classification.get('real') == classification.get('pred')) / y_test.shape[0]

(X_test[0, :] - X_train[:, :]).sum(axis=1)
temp = np.tile(X_test[:, :], (5000, 1))

temp = X_test[:, :].dot(np.ones((3072, 500)))
