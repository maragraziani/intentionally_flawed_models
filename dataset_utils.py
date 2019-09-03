import keras.backend as K
import numpy as np
import keras.datasets

import matplotlib.pyplot as plt
import keras
#from memorization_utils import *
import os
from sklearn.linear_model import Ridge
import sys
sys.path.append('../rcvs_fexps/')
sys.path.append('../rcvs_fexps/iMIMIC-RCVs/')
sys.path.append('../rcvs_fexps/iMIMIC-RCVs/scripts/')
sys.path.append('../rcvs_fexps/iMIMIC-RCVs/scripts/keras_vis_rcv/')
#from rcv_utils import *
from mnist_utils import *
import rcv_utils
import PIL

from scipy import misc
import numpy as np
#import tensorflow as tf
import argparse

import cv2
import h5py

"""
dataset_utils contains the list of classes and functions used to
create the datasets and to add the label perturbations
"""

class Dataset():

    def __init__(self, train_data, val_data, test_data,
                label_corrupt_p = 0.0, gaussian_noise_f = 0.0, num_classes=46, random_seed = 1):

        self.x_train = train_data[0]
        self.x_val = val_data[0]
        self.x_test = test_data[0]

        self.y_train = train_data[1]
        self.y_val = val_data[1]
        self.y_test = test_data[1]

        self.num_classes = num_classes

        self.train_mask = np.zeros(len(self.y_train))

        self.seed = random_seed

        if label_corrupt_p > 0.0:
            self.label_corrupt(label_corrupt_p)
        if gaussian_noise_f > 0.0:
            self.gaussian_noise(gaussian_noise_f)

    def label_corrupt(self, corrupted):
        ## NEW VERSION
        # Corrupts the labels in the training set according to
        # the specified corruption probability
        print 'NEW VERS'
        labels=np.array(self.y_train)
        #labels = np.reshape(len(labels),1)
        np.random.seed(self.seed)
        mask = np.random.rand(len(labels)) <= corrupted
        #rnd_labels = np.random.choice(self.num_classes, mask.sum())
        true_labels = labels[mask]
        #rnd_labels = np.reshape(rnd_labels, (len(rnd_labels),1))
        #labels[mask] = rnd_labels
        print true_labels
        print np.shape(true_labels)
        np.random.shuffle(true_labels)
        print true_labels
        print np.shape(true_labels)
        labels[mask] = true_labels
        #labels = [int(x) for x in labels]
        # corruption
        self.y_train = labels
        self.train_mask = mask

    def gaussian_noise(self, gaussian_noise_f):
        # Adds Gaussian Noise to the images,
        # matching the real dataset's mean and variance
        data = np.array(self.x_train)
        mean = np.mean(data)
        var = np.std(data)
        sigma = var**0.5
        #import pdb; pdb.set_trace()
        n_samples, row, col = data.shape
        mask = np.random.rand(n_samples) <= gaussian_noise_f
        gaussian = np.random.normal(mean, sigma, (row, col))
        gaussian = gaussian.reshape(row, col)
        noisy_imgs = [x+gaussian for x in data[mask]]
        data[mask] = noisy_imgs
        self.x_train = data


def check_max_img_size(img_list, source):
        max_img_width = 0
        max_img_height = 0
        for img_name in img_list:
            img_shape = np.asarray(PIL.Image.open('{}/dtd/images/{}'.format(source, img_name))).shape
            #import pdb; pdb.set_trace()
            img_height, img_width = img_shape[0], img_shape[1]
            if img_height>max_img_height:
                max_img_height = img_height
            if img_width>max_img_width:
                max_img_width = img_width
        return max_img_height, max_img_width

def load_split(no, source, textures):

    print "Loading split no. {}".format(no)

    train_split_file = '{}/dtd/labels/train{}.txt'.format(source, no)
    f = open(train_split_file, "r")
    training_img_names = [line.strip('\n') for line in f.readlines()]

    #import pdb; pdb.set_trace()

    val_split_file = '{}/dtd/labels/val{}.txt'.format(source, no)
    fv = open(val_split_file, "r")
    validation_img_names = [line.strip('\n') for line in fv.readlines()]

    test_split_file = '{}/dtd/labels/test{}.txt'.format(source, no)
    fte = open(test_split_file, "r")
    testing_img_names = [line.strip('\n') for line in fte.readlines()]

    all_img_names = training_img_names+validation_img_names+testing_img_names
    max_height, max_width  = check_max_img_size(all_img_names, source)
    print max_height, max_width

    '''
    training_images = np.zeros((len(training_img_names), 299,299,3), dtype='uint8')
    validation_images = np.zeros((len(validation_img_names), 299,299,3), dtype='uint8')
    testing_images = np.zeros((len(testing_img_names), 299,299,3), dtype='uint8')
    '''
    training_images = np.zeros((len(training_img_names), max_height,max_width,4), dtype='uint8')
    validation_images = np.zeros((len(validation_img_names), 299,299,3), dtype='uint8')
    testing_images = np.zeros((len(testing_img_names), 299,299,3), dtype='uint8')
    '''
    training_images[:,:,0] +=1
    validation_images[:,:,0] +=1
    testing_images[:,:,0] +=1
    '''

    '''
    training_images_mask = training_images[:,:,0]
    validation_images_mask = validation_images[:,:,0]
    testing_images_mask = testing_images[:,:,0]
    '''

    training_labels = []
    validation_labels = []
    testing_labels = []

    i=0
    for img_name in training_img_names[:]:
        img = PIL.Image.open('{}/dtd/images/{}'.format(source, img_name))

        '''
        width, height = img.size   # Get dimensions
        left = (width - 299)/2
        top = (height - 299)/2
        right = (width + 299)/2
        bottom = (height + 299)/2

        img = img.crop((left, top, right, bottom))
        '''
        nimg = np.asarray(img)
        nimg_shape = nimg.shape
        #import pdb; pdb.set_trace()
        training_images[i, 0:nimg_shape[0], 0:nimg_shape[1], :3] = nimg
        training_images[i, 0:nimg_shape[0]:, 0:nimg_shape[1], -1] = 1
        #import pdb; pdb.set_trace()
        #training_images_mask[i, 0:nimg_shape[0], 0:nimg_shape[1], :] = 1
        i+=1
        training_labels.append(np.argwhere(np.asarray(textures)==img_name.split('/')[0])[0][0])
        #print nimg.shape

    i=0
    for img_name in validation_img_names[:]:
        img = PIL.Image.open('{}/dtd/images/{}'.format(source, img_name))

        width, height = img.size   # Get dimensions
        left = (width - 299)/2
        top = (height - 299)/2
        right = (width + 299)/2
        bottom = (height + 299)/2

        img = img.crop((left, top, right, bottom))

        nimg = np.asarray(img)
        nimg_shape = nimg.shape
        validation_images[i, 0:nimg_shape[0], 0:nimg_shape[1], :3]=nimg
        #validation_images[i, 0:nimg_shape[0], 0:nimg_shape[1], -1]=1

        #validation_images_mask[i, 0:nimg_shape[0], 0:nimg_shape[1], :] = 1

        i+=1
        #validation_images.append(np.argwhere(np.asarray(textures)==img_name.split('/')[0])[0][0])
        #validation_images[i]=nimg
        #i+=1
        validation_labels.append(np.argwhere(np.asarray(textures)==img_name.split('/')[0])[0][0])
    i=0
    for img_name in testing_img_names[:]:
        img = PIL.Image.open('{}/dtd/images/{}'.format(source, img_name))

        width, height = img.size   # Get dimensions
        left = (width - 299)/2
        top = (height - 299)/2
        right = (width + 299)/2
        bottom = (height + 299)/2

        img = img.crop((left, top, right, bottom))

        nimg=  np.asarray(img)
        nimg_shape = nimg.shape
        testing_images[i, 0:nimg_shape[0], 0:nimg_shape[1], :3]=nimg
        #testing_images[i, 0:nimg_shape[0], 0:nimg_shape[1], -1]=1
        #testing_images_mask[i, 0:nimg_shape[0], 0:nimg_shape[1], :] = 1
        #testing_/\/images[i]=nimg
        i+=1
        testing_labels.append(np.argwhere(np.asarray(textures)==img_name.split('/')[0])[0][0])
    return (np.asarray(training_images, dtype='uint8'), training_labels), \
            (np.asarray(validation_images, dtype='uint8'), validation_labels), \
            (np.asarray(testing_images, dtype='uint8'), testing_labels)
            #(training_images_mask, validation_images_mask, testing_images_mask)

def load_val_split(no, source, textures):

    print "Loading split no. {}".format(no)

    val_split_file = '{}/dtd/labels/val{}.txt'.format(source, no)
    fv = open(val_split_file, "r")
    validation_img_names = [line.strip('\n') for line in fv.readlines()]

    validation_images = np.zeros((len(validation_img_names), 299,299,3), dtype='uint8')

    validation_labels = []

    i=0
    for img_name in validation_img_names[:]:
        img = PIL.Image.open('{}/dtd/images/{}'.format(source, img_name))

        width, height = img.size   # Get dimensions
        left = (width - 299)/2
        top = (height - 299)/2
        right = (width + 299)/2
        bottom = (height + 299)/2

        img = img.crop((left, top, right, bottom))

        nimg = np.asarray(img)
        nimg_shape = nimg.shape
        validation_images[i, 0:nimg_shape[0], 0:nimg_shape[1], :3]=nimg
        #validation_images[i, 0:nimg_shape[0], 0:nimg_shape[1], -1]=1

        #validation_images_mask[i, 0:nimg_shape[0], 0:nimg_shape[1], :] = 1

        i+=1
        #validation_images.append(np.argwhere(np.asarray(textures)==img_name.split('/')[0])[0][0])
        #validation_images[i]=nimg
        #i+=1
        validation_labels.append(np.argwhere(np.asarray(textures)==img_name.split('/')[0])[0][0])

    return (np.asarray(validation_images, dtype='uint8'), validation_labels)


'''
Datasets
'''

class ImageNet10Random():
    '''
    Params
    corrupted: float
      Default 0.0
    num_classes: int
      Default 10.
    '''

    def __init__(self, label_corrupt_p=0.0,
                 gaussian_noise_f = 0.0, 
                 classes=[], 
                 path_to_train='', 
                 path_to_val='',
                 random_seed=1,
                 **kwargs):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = get_imgnt_datasets(classes, path_to_train, path_to_val)
        self.num_classes = len(classes)
        
        self._train_mask = np.zeros(len(self.y_train))
        
        self.seed = random_seed

        if label_corrupt_p > 0.0:
            self.label_corrupt(label_corrupt_p)
        if gaussian_noise_f > 0.0:
            self.gaussian_noise(gaussian_noise_f)

    def label_corrupt(self, corrupted):
        # Corrupts the labels in the training set according to
        # the specified corruption probability
        labels=np.array(self.y_train)
        #labels = np.reshape(len(labels),1)
        np.random.seed(self.seed)
        mask = np.random.rand(len(labels)) <= corrupted
        true_labels = labels[mask]
        print true_labels, np.shape(true_labels)
        np.random.shuffle(true_labels)
        print true_labels, np.shape(true_labels)
                                    
        #rnd_labels = np.random.choice(self.num_classes, mask.sum())
        #rnd_labels = np.reshape(rnd_labels, (len(rnd_labels),1))
        #labels[mask] = rnd_labels
        labels[mask] = true_labels
        #labels = [int(x) for x in labels]
        # corruption
        self.y_train = labels
        self.train_mask = mask #saving which labels were actually perturbed
        
    def gaussian_noise(self, gaussian_noise_f):
        # Adds Gaussian Noise to the images,
        # matching the real dataset's mean and variance
        data = np.array(self.x_train)
        mean = np.mean(data)
        var = np.std(data)
        sigma = var**0.5
        n_samples, row, col, ch = data.shape
        mask = np.random.rand(n_samples) <= gaussian_noise_f
        gaussian = np.random.normal(mean, sigma, (row, col, ch))
        gaussian = gaussian.reshape(row, col, ch)
        noisy_imgs = [x+gaussian for x in data[mask]]
        data[mask] = noisy_imgs
        self.x_train = data

class CIFAR10Random():
    '''
    Params
    corrupted: float
      Default 0.0
    num_classes: int
      Default 10.
    '''

    def __init__(self, label_corrupt_p=0.0, gaussian_noise_f = 0.0, num_classes=10, random_seed=1,
                 **kwargs):
        #super(CIFAR10Random, self).__init__(**kwargs)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.cifar10.load_data()
        self.num_classes = num_classes
        #import pdb; pdb.set_trace()
        self.y_train = self.y_train.T[0]
        self.y_test = self.y_test.T[0]
        self._train_mask = np.zeros(len(self.y_train)) #to save which examples were corrupted, if any
        self.seed = random_seed
        # note: corruption is performed on the training set.
        # you test on real data to check generalization
        if label_corrupt_p > 0.0:
            self.label_corrupt(label_corrupt_p)
        if gaussian_noise_f > 0.0:
            self.gaussian_noise(gaussian_noise_f)

    def label_corrupt(self, corrupted):
        ## NEW VERSION
        # Corrupts the labels in the training set according to
        # the specified corruption probability
        print 'NEW VERS'
        labels=np.array(self.y_train)
        #labels = np.reshape(len(labels),1)
        np.random.seed(self.seed)
        mask = np.random.rand(len(labels)) <= corrupted
        #rnd_labels = np.random.choice(self.num_classes, mask.sum())
        true_labels = labels[mask]
        #rnd_labels = np.reshape(rnd_labels, (len(rnd_labels),1))
        #labels[mask] = rnd_labels
        print true_labels
        print np.shape(true_labels)
        np.random.shuffle(true_labels)
        print true_labels
        print np.shape(true_labels)
        labels[mask] = true_labels
        #labels = [int(x) for x in labels]
        # corruption
        self.y_train = labels
        self.train_mask = mask
        '''       
        
        # Corrupts the labels in the training set according to
        # the specified corruption probability
        labels=np.array(self.y_train)
        #labels = np.reshape(len(labels),1)
        np.random.seed(1)
        mask = np.random.rand(len(labels)) <= corrupted
        rnd_labels = np.random.choice(self.num_classes, mask.sum())
        rnd_labels = np.reshape(rnd_labels, (len(rnd_labels),1))
        labels[mask] = rnd_labels
        labels = [int(x) for x in labels]
        # corruption
        self.y_train = labels
        self.train_mask = mask #saving which labels were actually perturbed
        '''
        
    def gaussian_noise(self, gaussian_noise_f):
        # Adds Gaussian Noise to the images,
        # matching the real dataset's mean and variance
        data = np.array(self.x_train)
        mean = np.mean(data)
        var = np.std(data)
        sigma = var**0.5
        n_samples, row, col, ch = data.shape
        mask = np.random.rand(n_samples) <= gaussian_noise_f
        gaussian = np.random.normal(mean, sigma, (row, col, ch))
        gaussian = gaussian.reshape(row, col, ch)
        noisy_imgs = [x+gaussian for x in data[mask]]
        data[mask] = noisy_imgs
        self.x_train = data

class MNISTRandom():
    '''
    Params
    corrupted: float
      Default 0.0
    num_classes: int
      Default 10.
    '''

    def __init__(self, label_corrupt_p=0.0, gaussian_noise_f = 0.0, num_classes=10, **kwargs):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()
        self.num_classes = num_classes
        # note: corruption is performed on the training set.
        # you test on real data to check generalization
        if label_corrupt_p > 0.0:
            self.label_corrupt(label_corrupt_p)
        if gaussian_noise_f > 0.0:
            self.gaussian_noise(gaussian_noise_f)
    '''
    def label_corrupt(self, corrupted):
        # Corrupts the labels in the training set according to
        # the specified corruption probability
        labels=np.array(self.y_train)
        #labels = np.reshape(len(labels),1)
        np.random.seed(1)
        mask = np.random.rand(len(labels)) <= corrupted
        rnd_labels = np.random.choice(self.num_classes, mask.sum())
        #rnd_labels = np.reshape(rnd_labels, (len(labels),1))
        labels[mask] = rnd_labels
        labels = [int(x) for x in labels]
        # corruption
        self.y_train = labels
        '''
    def label_corrupt(self, corrupted):
        ## NEW VERSION
        # Corrupts the labels in the training set according to
        # the specified corruption probability
        print 'NEW VERS'
        labels=np.array(self.y_train)
        #labels = np.reshape(len(labels),1)
        np.random.seed(1)
        mask = np.random.rand(len(labels)) <= corrupted
        #rnd_labels = np.random.choice(self.num_classes, mask.sum())
        true_labels = labels[mask]
        #rnd_labels = np.reshape(rnd_labels, (len(rnd_labels),1))
        #labels[mask] = rnd_labels
        #print true_labels
        #print np.shape(true_labels)
        np.random.shuffle(true_labels)
        #print true_labels
        #print np.shape(true_labels)
        labels[mask] = true_labels
        #labels = [int(x) for x in labels]
        # corruption
        self.y_train = labels

    def gaussian_noise(self, gaussian_noise_f):
        # Adds Gaussian Noise to the images,
        # matching the real dataset's mean and variance
        data = np.array(self.x_train)
        mean = np.mean(data)
        var = np.std(data)
        sigma = var**0.5
        #import pdb; pdb.set_trace()
        n_samples, row, col = data.shape
        mask = np.random.rand(n_samples) <= gaussian_noise_f
        gaussian = np.random.normal(mean, sigma, (row, col))
        gaussian = gaussian.reshape(row, col)
        noisy_imgs = [x+gaussian for x in data[mask]]
        data[mask] = noisy_imgs
        self.x_train = data


'''
Extra functions
'''

## Adding ImageNet10 functions
def get_imgnt_datasets(classes, path_to_train, path_to_val):
    print path_to_train, path_to_val
    dataset = h5py.File(path_to_train, 'r')
    dataset_val = h5py.File(path_to_val, 'r')
    #for c in classes:
        #print dataset[c], dataset_val[c]
    x_train, y_train = get_data(dataset, classes)
    x_val, y_val = get_data(dataset_val, classes)
    return (x_train, y_train), (x_val, y_val)
def get_data(dataset, classes):
    data=[]
    labels=[]
    l=0
    for c in classes:
        data.append(np.asarray(dataset[c][:]))
        labels.append(np.asarray([l]* len(dataset[c])))
        #import pdb; pdb.set_trace()
        l+=1
    data = np.concatenate([d for d in data])
    labels = np.concatenate([l for l in labels])
    return data, labels

def print_info(name, obj):
    print name 
    
def get_data(dataset, classes):
    data=[]
    labels=[]
    l=0
    for c in classes:
        data.append(np.asarray(dataset[c][:]))
        labels.append(np.asarray([l]* len(dataset[c])))
        #import pdb; pdb.set_trace()
        l+=1
    data = np.concatenate([d for d in data])
    labels = np.concatenate([l for l in labels])
    return data, labels

def get_imgnt_datasets(classes, path_to_train, path_to_val):
    print path_to_train, path_to_val
    dataset = h5py.File(path_to_train, 'r')
    dataset_val = h5py.File(path_to_val, 'r')
    x_train, y_train = get_data(dataset, classes)
    x_val, y_val = get_data(dataset_val, classes)
    return (x_train, y_train), (x_val, y_val)