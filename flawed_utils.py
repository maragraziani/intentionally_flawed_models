from __future__ import division #needed
import keras.backend as K
import numpy as np
import keras.datasets
import matplotlib.pyplot as plt
import keras
from memorization_utils import *
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
from skimage.feature import greycomatrix, greycoprops
import skimage.color

import cv2

import tensorflow as tf

"""
Here we gather all the functions used to compute the concept measures,
the measures of colorfulness, color-ness, and texuture descriptors of
the images.
"""

def compute_concept_measures(x_test):
    concept_measures={}
    concept_measures['dissimilarity']=[]
    concept_measures['contrast']=[]
    concept_measures['homogeneity']=[]
    concept_measures['ASM']=[]
    concept_measures['energy']=[]
    concept_measures['correlation']=[]
    for i in range(len(x_test)):
        glcm = greycomatrix(skimage.img_as_ubyte(skimage.color.rgb2gray(x_test[i])), [1], [0] , symmetric=True, normed=True)
        concept_measures['dissimilarity'].append(greycoprops(glcm, 'dissimilarity'))
        concept_measures['contrast'].append(greycoprops(glcm, 'contrast'))
        concept_measures['homogeneity'].append(greycoprops(glcm, 'homogeneity'))
        concept_measures['ASM'].append( greycoprops(glcm, 'ASM'))
        concept_measures['energy'].append( greycoprops(glcm, 'energy'))
        concept_measures['correlation'].append(greycoprops(glcm, 'correlation'))
    for k in concept_measures.keys():
        concept_measures[k] = np.asarray(concept_measures[k]).T[0][0]
    return concept_measures

def colorfulness(img):
    # split the image into its respective RGB components
    (B, G, R) = cv2.split(img.astype("float"))

    # compute rg = R - G
    rg = np.absolute(R - G)

    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)

    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))

    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)

"""
modules for
      RGB to LAB color-space conversion
      LAB to RGB color-space conversion
"""


def preprocess(image):
    with tf.name_scope('preprocess'):
        # [0, 1] => [-1, 1]
        return image * 2 - 1

def deprocess(image):
    with tf.name_scope('deprocess'):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

def preprocess_lab(lab):
    with tf.name_scope('preprocess_lab'):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]

def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope('deprocess_lab'):
        #TODO This is axis=3 instead of axis=2 when deprocessing batch of images
               # ( we process individual images but deprocess batches)
        #return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=2)

def augment(image, brightness):
    # (a, b) color channels, combine with L channel and convert to rgb
    a_chan, b_chan = tf.unstack(image, axis=3)
    L_chan = tf.squeeze(brightness, axis=3)
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    rgb = lab_to_rgb(lab)
    return rgb

def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message='image must have 3 color channels')
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError('image must be either 3 or 4 dimensions')

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image

def rgb_to_lab(srgb):
    # based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
    with tf.name_scope('rgb_to_lab'):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])
        with tf.name_scope('srgb_to_xyz'):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope('xyz_to_cielab'):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope('lab_to_rgb'):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])
        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope('cielab_to_xyz'):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope('xyz_to_srgb'):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

    return tf.reshape(srgb_pixels, tf.shape(lab))

def rgb_to_hsv(rgb):
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv

def hsv_to_rgb(hsv):
    # Translated from source of colorsys.hsv_to_rgb
    # h,s should be a numpy arrays with values between 0.0 and 1.0
    # v should be a numpy array with values between 0.0 and 255.0
    # hsv_to_rgb returns an array of uints between 0 and 255.
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')

from skimage.feature import greycomatrix, greycoprops
import skimage.color
# loading concept measures (only for validation split)
def compute_concept_measures(x_test):
    concept_measures={}
    concept_measures['dissimilarity']=[]
    concept_measures['contrast']=[]
    concept_measures['homogeneity']=[]
    concept_measures['ASM']=[]
    concept_measures['energy']=[]
    concept_measures['correlation']=[]

    concept_measures['colorfulness']=[]
    concept_measures['black']=[]
    concept_measures['white']=[]
    concept_measures['red']=[]
    concept_measures['orange']=[]
    concept_measures['yellow']=[]
    concept_measures['green']=[]
    concept_measures['cyano']=[]
    concept_measures['blue']=[]
    concept_measures['purple']=[]
    concept_measures['magenta']=[]

    colors_list = ['black', 'white', 'red', 'orange','yellow', 'green', 'cyano', 'blue', 'purple', 'magenta']


    for i in range(len(x_test)):
        glcm = greycomatrix(skimage.img_as_ubyte(skimage.color.rgb2gray(x_test[i])), [1], [0] , symmetric=True, normed=True)
        concept_measures['dissimilarity'].append(greycoprops(glcm, 'dissimilarity'))
        concept_measures['contrast'].append(greycoprops(glcm, 'contrast'))
        concept_measures['homogeneity'].append(greycoprops(glcm, 'homogeneity'))
        concept_measures['ASM'].append( greycoprops(glcm, 'ASM'))
        concept_measures['energy'].append( greycoprops(glcm, 'energy'))
        concept_measures['correlation'].append(greycoprops(glcm, 'correlation'))
        concept_measures['colorfulness'].append(colorfulness(x_test[i]))
        x_test[i] = cv2.cvtColor(x_test[i],cv2.COLOR_RGB2BGR)
        for color in colors_list:
            concept_measures[color].append(colorness(x_test[i], color))
    for k in ['dissimilarity', 'contrast', 'homogeneity','ASM', 'energy', 'correlation']:
        concept_measures[k] = np.asarray(concept_measures[k]).T[0][0]
    return concept_measures

def colorfulness(img):
    # split the image into its respective RGB components
    (B, G, R) = cv2.split(img.astype("float"))

    # compute rg = R - G
    rg = np.absolute(R - G)

    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)

    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))

    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)
def hsv_histograms(image):
    hist_hue = cv2.calcHist([image], [0], None, [180], [0, 180])
    hist_sat = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_val = cv2.calcHist([image], [2], None, [256], [0, 256])
    print np.mean(image[:,:,0])
    print np.min(image[:,:,0])
    print np.max(image[:,:,0])

    return hist_hue, hist_sat, hist_val

def color_picker(color_name):
    brg_colors={}
    brg_colors['red']= np.uint8([[[0,0,255 ]]])
    brg_colors['orange'] = np.uint8([[[0,128,255 ]]])
    brg_colors['yellow'] = np.uint8([[[0,255,255 ]]])
    brg_colors['green'] = np.uint8([[[0,255,0 ]]])
    brg_colors['cyano'] = np.uint8([[[255,255,0 ]]])
    brg_colors['blue'] = np.uint8([[[255,0,0]]])
    brg_colors['purple'] = np.uint8([[[255,0,128]]])
    brg_colors['magenta'] = np.uint8([[[255,0,255 ]]])
    brg_colors['white'] = np.uint8([[[255,255,255 ]]])
    brg_colors['black'] = np.uint8([[[0,0,0 ]]])

    rgb_color_code = brg_colors[color_name]
    return cv2.cvtColor(rgb_color_code,cv2.COLOR_BGR2HSV)

def round_hue(hue_val):
    hues = np.arange(0,180)
    if hue_val<180:
        hue_def = hues[hue_val]
    else:
        hue_def = hues[(hue_val)%179]
    return hue_def

def quantize_hue_ranges(image, color_name):
    if color_name == 'red':
        hue_min = 165
        hue_max = 10

    elif color_name == 'orange':
        hue_min = 10
        hue_max = 25
    elif color_name == 'yellow':
        hue_min = 25
        hue_max = 40
    elif color_name == 'green':
        hue_min = 40
        hue_max = 75
    elif color_name == 'cyano':
        hue_min = 75
        hue_max = 100
    elif color_name == 'blue':
        hue_min = 100
        hue_max = 125
    elif color_name == 'purple':
        hue_min = 125
        hue_max = 145
    elif color_name == 'magenta':
        hue_min = 145
        hue_max = 165
    elif (color_name == 'white' or color_name == 'black'):
        hue_min = 0
        hue_max = 255

    return hue_min, hue_max

def colorness(image, color_name, threshold = 0, verbose=False):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    #plt.imshow(image)
    if verbose:
        h,s,v = hsv_histograms(image)
        plt.figure()
        plt.plot(h)
        plt.figure()
        plt.plot(s)
        plt.figure()
        plt.plot(v)

    if threshold == 0:
        hue_min, hue_max = quantize_hue_ranges(image, color_name)
        if verbose:
            print 'hue min, hue max: ', hue_min, hue_max
    else:
        h_point =color_picker(color_name)
        hue_min = round_hue(h_point[0][0][0]-threshold)
        hue_max = round_hue(h_point[0][0][0]+threshold)
        if verbose:
            print 'hue min, hue max: ', hue_min, hue_max
    '''
    if verbose:
        print 'red', color_picker(brg_colors['red'])
        print 'orange', color_picker(brg_colors['orange'])
        print 'yellow', color_picker(brg_colors['yellow'])
        print 'green', color_picker(brg_colors['green'])
        print 'cyano', color_picker(brg_colors['cyano'])
        print 'blue', color_picker(brg_colors['blue'])
        print 'purple', color_picker(brg_colors['purple'])
        print  'magenta', color_picker(brg_colors['magenta'])
    '''
    if (hue_min == hue_max == 0) or (hue_min == 0 and hue_max == 255):
        #it is either black or white
        if color_name=='black':
            low_c = np.array([0,
                              0,
                              0])
            upp_c = np.array([hue_max,
                              100,
                              100])
        if color_name=='white':
            low_c = np.array([0,
                              0,
                              190])
            upp_c = np.array([hue_max,
                              50,
                              255])

        if verbose:
            print 'low_c', low_c, 'upp_c', upp_c

        mask = cv2.inRange(image, low_c, upp_c)

    elif hue_min>hue_max:
        low_c = np.array([0,
                      50,
                      77])
        upp_c = np.array([hue_max,
                      255,
                      255])
        mask1 = cv2.inRange(image, low_c, upp_c)

        low_c = np.array([hue_min,
                      50,
                      77])
        upp_c = np.array([180,
                      255,
                      255])
        mask2 = cv2.inRange(image, low_c, upp_c)

        mask = cv2.bitwise_or(mask1, mask1, mask2)

    else:
        low_c = np.array([hue_min,
                          50,
                          77])
        upp_c = np.array([hue_max,
                          255,
                          255])
        if verbose:
            print 'low_c', low_c, 'upp_c', upp_c

        mask = cv2.inRange(image, low_c, upp_c)
    if verbose:
        print mask

    res = cv2.bitwise_and(image, image, mask = mask)
    if verbose:
        plt.figure()
        plt.imshow(mask, cmap='Greys')
        plt.colorbar()

        plt.figure()
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB))
        plt.figure()
        plt.imshow(cv2.cvtColor(res, cv2.COLOR_HSV2RGB))

    #print np.sum(mask==255)
    x,y,z = image.shape
    print x, y , z
    if verbose:
        print np.sum(mask==255)/(float(x)*float(y))

    return float(np.sum(mask==255))/(float(x)*float(y))

def get_color_mask(image, color_name, threshold=0, verbose=False):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    if threshold == 0:
        hue_min, hue_max = quantize_hue_ranges(image, color_name)
        if verbose:
            print 'hue min, hue max: ', hue_min, hue_max
    else:
        h_point =color_picker(color_name)
        hue_min = round_hue(h_point[0][0][0]-threshold)
        hue_max = round_hue(h_point[0][0][0]+threshold)
        if verbose:
            print 'hue min, hue max: ', hue_min, hue_max
    '''
    if verbose:
        print 'red', color_picker(brg_colors['red'])
        print 'orange', color_picker(brg_colors['orange'])
        print 'yellow', color_picker(brg_colors['yellow'])
        print 'green', color_picker(brg_colors['green'])
        print 'cyano', color_picker(brg_colors['cyano'])
        print 'blue', color_picker(brg_colors['blue'])
        print 'purple', color_picker(brg_colors['purple'])
        print  'magenta', color_picker(brg_colors['magenta'])
    '''
    if (hue_min == hue_max == 0) or (hue_min == 0 and hue_max == 255):
        print 'black or white', color_name
        #it is either black or white
        if color_name=='black':
            print 'in black'
            low_c = np.array([0,
                              0,
                              0])
            upp_c = np.array([hue_max,
                              100,
                              100])
        if color_name=='white':
            print 'in white'
            low_c = np.array([0,
                              0,
                              190])
            upp_c = np.array([hue_max,
                              50,
                              255])

        mask = cv2.inRange(image, low_c, upp_c)
        return mask
    if hue_min>hue_max:
        low_c = np.array([0,
                      50,
                      77])
        upp_c = np.array([hue_max,
                      255,
                      255])
        mask1 = cv2.inRange(image, low_c, upp_c)

        low_c = np.array([hue_min,
                      50,
                      77])
        upp_c = np.array([180,
                      255,
                      255])
        mask2 = cv2.inRange(image, low_c, upp_c)

        mask = cv2.bitwise_or(mask1, mask1, mask2)

    else:
        low_c = np.array([hue_min,
                          50,
                          77])
        upp_c = np.array([hue_max,
                          255,
                          255])
        if verbose:
            print 'low_c', low_c, 'upp_c', upp_c

        mask = cv2.inRange(image, low_c, upp_c)
    if verbose:
        print mask
    return mask

def image_colors(image, threshold=0):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    x,y,z = image.shape
    new_image = np.ones((x,y,3), dtype='uint8')*255
    #print new_image
    #new_image = cv2.cvtColor(new_image,cv2.COLOR_BGR2RGB)
    #plt.imshow(new_image)
    #new_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    colors = ['black', 'white', 'red', 'orange','yellow', 'green', 'cyano', 'blue', 'purple', 'magenta']
    #plt.imshow(image)

    for c in colors:
        #print c
        color_mask = get_color_mask(image, c, threshold=threshold)
        #plt.figure()
        #plt.imshow(color_mask)
        #print color_mask
        #plt.figure()
        #plt.title(c)
        #new_image =  cv2.cvtColor(new_image,cv2.COLOR_RGB2HSV)
        new_image = recolor(new_image, color_mask, c)
        #new_image =  cv2.cvtColor(new_image,cv2.COLOR_HSV2RGB)
        #plt.imshow(new_image)
        #plt.figure()
        #plt.imshow(color_mask)

        #
    new_image =  cv2.cvtColor(new_image,cv2.COLOR_HSV2RGB)
    plt.figure()
    plt.imshow(new_image)
    #print new_image
    return new_image

def recolor(image, color_mask, color_name):
    color_hues = {}
    color_hues['red']=5
    color_hues['orange']=15
    color_hues['yellow']=30
    color_hues['green']=50
    color_hues['cyano']=90
    color_hues['blue']=120
    color_hues['purple']=140
    color_hues['magenta']=160
    color_hues['black']=0
    color_hues['white']=0

    if color_name == 'black':
        image[color_mask==255] = (0,0,0)

    elif color_name == 'white':
        image[color_mask==255] = (0,0,255)
    else:
        #print image[color_mask==255].shape
        image[color_mask==255] = (color_hues[color_name],255,255)
        #print image.shape

        #plt.imshow(image)
    return image
