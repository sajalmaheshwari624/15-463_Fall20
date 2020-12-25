import cp_hw5
import numpy as np
import skimage
from skimage import io
from skimage.color import rgb2gray, rgb2xyz
from skimage.transform import resize
from scipy.interpolate import interp2d
import skimage.filters
from os import walk
import matplotlib.pyplot as plt
import os	
import imageio
import copy
import sys
import argparse
import scipy
from scipy.ndimage import uniform_filter
from PIL import Image
from matplotlib.colors import LightSource 
from mpl_toolkits.mplot3d import Axes3D

import pdb
st = pdb.set_trace

def get_gradients(im) :
    filter_x = np.array([[1,-1]])
    filter_y = np.array([[1],[-1]])
    im_x = scipy.signal.convolve2d(im, filter_x, mode = 'same')
    im_y = scipy.signal.convolve2d(im, filter_y, mode = 'same')
    return im_x, im_y

def initials(path, size = 1) :
    count = 0
    for i,data_file in enumerate(sorted(os.listdir(path))):
        break_file = data_file.split('.')
        if break_file[1] == 'tiff' or break_file[1] == 'tif':
            count += 1
            im = io.imread(path + data_file)
            #im = im[1462:3063, 2710:3953,:] #- Laundry bag
            im = im[3003:3700, 3177:3960,:]
            im = uniform_filter(im, size = size)
            H,W,_ = im.shape
            im_xyz = rgb2xyz(im)
            im_y_channel = im_xyz[:,:,1]
            flatten_im = np.reshape(im_y_channel, (1,-1))
            if count == 1 :
                stacked_im = flatten_im
            else :
                stacked_im = np.vstack((stacked_im, flatten_im))
    return stacked_im,H,W

def uncalibrated(stacked_im, QMatrix, rows = 431, cols = 369) :
    I = stacked_im
    u,s,vh = np.linalg.svd(I, full_matrices = False)
    u_truncated = u[:,0:3]
    s_truncated = s[0:3]
    vh_truncated = vh[0:3,:]
    vh_truncated_sigma_corrected = np.matmul(np.diag(np.sqrt(s_truncated)), vh_truncated)
    u_truncated_sigma_corrected = np.matmul(u_truncated, np.diag(np.sqrt(s_truncated)))
    vh_truncated_sigma_corrected = np.matmul(np.transpose(np.linalg.inv(QMatrix)), vh_truncated_sigma_corrected)
    for i in range(3) :
        current_channel_image = np.reshape(vh_truncated_sigma_corrected[i,:],(rows, cols))
        if i == 0 :
            magnitude_image = current_channel_image ** 2
            final_image_unnormalized = current_channel_image
        else :
            magnitude_image += current_channel_image ** 2
            final_image_unnormalized = np.dstack((final_image_unnormalized, current_channel_image))

    magnitude_image = np.sqrt(magnitude_image)
    for i in range(3) :
        if i == 0 :
            final_image_normalized = final_image_unnormalized[:,:,i] / magnitude_image
        else :
            final_image_normalized = np.dstack((final_image_normalized, final_image_unnormalized[:,:,i]/magnitude_image))

    final_image_visualize = (final_image_normalized + 1)/2
    plt.imshow(final_image_visualize)
    plt.show()
    plt.imshow(magnitude_image, cmap = 'gray')
    plt.show()

    normal_image = final_image_normalized
    albedo_image = magnitude_image
    return u_truncated_sigma_corrected, normal_image, albedo_image, final_image_unnormalized

def render_image(normal_dir, albedo_image, light_direction, H = 431, W = 369) :
        light_dir = np.array(light_direction)
        light_dir = np.reshape(light_dir, (1,3))
        for i in range(3) :
            current_channel = normal_dir[:,:,i]
            current_channel_flatten = np.reshape(current_channel, (1,-1))
            if i == 0 :
                vectorized_image = current_channel_flatten
            else :
                vectorized_image = np.vstack((vectorized_image, current_channel_flatten))
        rendered_image_vectorized = np.matmul(light_dir, vectorized_image)
        rendered_image = np.reshape(rendered_image_vectorized, (H, W))
        rendered_image = albedo_image * rendered_image
        rendered_image[rendered_image < 0] = 0
        plt.imshow(rendered_image, cmap = 'gray')
        plt.show()

def integrability(pseudo_normal, sigma = 18, H = 431, W = 369, mu = 1, nu = 1, lamda = 1, vis = False) :
    filtered_pseudo_normal = skimage.filters.gaussian(pseudo_normal, sigma = sigma)
    pseudo_normal_x = np.gradient(filtered_pseudo_normal, axis = 1)
    pseudo_normal_y = np.gradient(filtered_pseudo_normal, axis = 0)

    A1 = pseudo_normal[:,:,0] * pseudo_normal_x[:,:,1] - pseudo_normal[:,:,1] * pseudo_normal_x[:,:,0]
    A2 = pseudo_normal[:,:,0] * pseudo_normal_x[:,:,2] - pseudo_normal[:,:,2] * pseudo_normal_x[:,:,0]
    A3 = pseudo_normal[:,:,1] * pseudo_normal_x[:,:,2] - pseudo_normal[:,:,2] * pseudo_normal_x[:,:,1]
    A4 = -pseudo_normal[:,:,0] * pseudo_normal_y[:,:,1] + pseudo_normal[:,:,1] * pseudo_normal_y[:,:,0]
    A5 = -pseudo_normal[:,:,0] * pseudo_normal_y[:,:,2] + pseudo_normal[:,:,2] * pseudo_normal_y[:,:,0]
    A6 = -pseudo_normal[:,:,1] * pseudo_normal_y[:,:,2] + pseudo_normal[:,:,2] * pseudo_normal_y[:,:,1]

    A1 = np.reshape(A1, (-1,1))
    A2 = np.reshape(A2, (-1,1))
    A3 = np.reshape(A3, (-1,1))
    A4 = np.reshape(A4, (-1,1))
    A5 = np.reshape(A5, (-1,1))
    A6 = np.reshape(A6, (-1,1))

    A_matrix = np.hstack((A1, A2, A3, A4, A5, A6))
    U,D,Vt = np.linalg.svd(A_matrix, full_matrices = False)
    V = np.transpose(Vt)
    Xvec = V[:,-1]
    delta = np.array([[-Xvec[2], Xvec[5], 1],
        [Xvec[1], -Xvec[4], 0],
        [-Xvec[0], Xvec[3], 0]])

    correction_mat = np.array([[1,0,0],[0,1,0],[mu,nu,lamda]])
    for i in range(3) :
        current_channel = pseudo_normal[:,:,i]
        current_channel_flatten = np.reshape(current_channel, (1,-1))
        if i == 0 :
            vectorized_image = current_channel_flatten
        else :
            vectorized_image = np.vstack((vectorized_image, current_channel_flatten))

    integrable_vectorized_image = np.matmul(np.matmul(correction_mat, np.linalg.inv(delta)), vectorized_image)

    for i in range(3) :
        current_channel_image = np.reshape(integrable_vectorized_image[i,:],(H, W))
        if i == 0 :
            magnitude_image = current_channel_image ** 2
            final_image_unnormalized = current_channel_image
        else :
            magnitude_image += current_channel_image ** 2
            final_image_unnormalized = np.dstack((final_image_unnormalized, current_channel_image))

    magnitude_image = np.sqrt(magnitude_image)
    for i in range(3) :
        if i == 0 :
            final_image_normalized = final_image_unnormalized[:,:,i] / magnitude_image
        else :
            final_image_normalized = np.dstack((final_image_normalized, final_image_unnormalized[:,:,i]/magnitude_image))

    integrable_albedo_image = magnitude_image
    integrable_normal_image = final_image_normalized

    final_image_visualize = (integrable_normal_image+ 1)/2
    if vis :
        plt.imshow(final_image_visualize)
        plt.show()
        plt.imshow(integrable_albedo_image * 10, cmap = 'gray')
        plt.show()

    return integrable_normal_image, integrable_albedo_image

def integrate_normal(integrable_normal_dir, isPos = True) :
    integrable_normal_image = integrable_normal_dir
    integrable_normal_image_x = integrable_normal_image[:,:,0] 
    integrable_normal_image_y = integrable_normal_image[:,:,1]
    if isPos :
        depth = cp_hw5.integrate_poisson(integrable_normal_image_x, integrable_normal_image_y)
    else :
        depth = cp_hw5.integrate_frankot(integrable_normal_image_x, integrable_normal_image_y)
    plt.imshow(depth, cmap = 'gray')
    plt.show()

    H, W = depth.shape
    x, y = np.meshgrid(np.arange(0,W), np.arange(0,H))

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ls = LightSource()
    color_shade = ls.shade(-depth, plt.cm.gray)
    surf = ax.plot_surface(x, y, -depth, facecolors=color_shade, rstride=4, cstride=4)
    plt.axis('off')
    plt.show()

def calibrated_stereo_and_rest(stacked_im, H = 431, W = 369) :
    S = cp_hw5.load_sources()
    S_t = np.transpose(S)
    A = np.matmul(np.linalg.inv(np.matmul(S_t,S)),S_t)
    correction_mat = np.array([[1,0,0],[0,1,0],[0,0,-1]])

    unnormalized_normal_image = np.matmul(correction_mat,(np.matmul(A,stacked_im)))
    for i in range(3) :
        current_channel_image = np.reshape(unnormalized_normal_image[i,:],(H, W))
        if i == 0 :
            magnitude_image = current_channel_image ** 2
            final_image_unnormalized = current_channel_image
        else :
            magnitude_image += current_channel_image ** 2
            final_image_unnormalized = np.dstack((final_image_unnormalized, current_channel_image))

    magnitude_image = np.sqrt(magnitude_image)
    for i in range(3) :
        if i == 0 :
            final_image_normalized = final_image_unnormalized[:,:,i] / magnitude_image
        else :
            final_image_normalized = np.dstack((final_image_normalized, final_image_unnormalized[:,:,i]/magnitude_image))

    normal_image = final_image_normalized
    albedo_image = magnitude_image
    plt.imshow((normal_image+1)/2)
    plt.show()

    plt.imshow(albedo_image, cmap = 'gray')
    plt.show()

    integrable_normal_image = normal_image
    integrable_normal_image_x = integrable_normal_image[:,:,0] / integrable_normal_image[:,:,2]
    integrable_normal_image_y = integrable_normal_image[:,:,1] / integrable_normal_image[:,:,2]
    depth = cp_hw5.integrate_poisson(integrable_normal_image_x, integrable_normal_image_y)
    plt.imshow(depth, cmap = 'gray')
    plt.show()

    H, W = depth.shape
    x, y = np.meshgrid(np.arange(0,W), np.arange(0,H))

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ls = LightSource()
    color_shade = ls.shade(-depth, plt.cm.gray)
    surf = ax.plot_surface(x, y, -depth, facecolors=color_shade, rstride=4, cstride=4)
    plt.axis('off')
    plt.show()

def find_min_entropy_gbr(unnormalized_normal_output, sigma = 18, H = 431,W = 369) :
    min_entropy = 10000
    final_i = 0
    final_j = 0
    final_k = 0
    for i in range(-5,5,1) :
        for j in range(-5, 5, 1) :
            for k in range(-10,10,2) :
                if k == 0 :
                    pass
                else :
                    _,integrable_albedo = integrability(unnormalized_normal_output, 18, H, W, i,j,k/10)
                    histogram,_ = np.histogram(integrable_albedo, bins=256, range=(0, 1))
                    histogram = histogram/np.sum(histogram)
                    entropy = scipy.special.entr(histogram)
                    entropy = np.sum(entropy)
                    if entropy < min_entropy :
                        min_entropy = entropy
                        final_i = i
                        final_j = j
                        final_k = k

    return final_i, final_j, final_k




if __name__ == "__main__":
    path="../coffee/"
    stacked_im,H,W = initials(path, 10)
    #QMatrix = np.random.random(size=(3,3))
    #QMatrix = np.dot(QMatrix,QMatrix.T) #invertible matrix 
    # QMatrix = np.array([[0.78817474, 0.4618897 , 0.67526141],
    #    [0.4618897 , 0.84887858, 0.61281791],
    #    [0.67526141, 0.61281791, 0.96396453]])
    QMatrix = np.eye(3)

    light_dir, normal_image, albedo_image, unnormalized_normal_output = uncalibrated(stacked_im, QMatrix, H, W)
    light_direction = [-0.1, -0.58, -0.58]
    rendered_image = render_image(normal_image, albedo_image, light_direction,H,W)


    integrable_normal_dir, integrable_albedo = integrability(unnormalized_normal_output, 18, H, W, 1,0,-1)
    integrable_normal_dir[:,:,2] = integrable_normal_dir[:,:,2] + 0.01

    integrate_normal(integrable_normal_dir)

    final_i, final_j, final_k = find_min_entropy_gbr(unnormalized_normal_output, 18, H,W)
    integrable_normal_dir, integrable_albedo = integrability(unnormalized_normal_output, 18, H, W, final_i,final_j,final_k, True)
    # integrable_normal_dir[:,:,2] = integrable_normal_dir[:,:,2] + 0.01

    integrate_normal(integrable_normal_dir)


    # calibrated_stereo_and_rest(stacked_im, H,W)





