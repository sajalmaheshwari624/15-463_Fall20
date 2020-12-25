import numpy as np
import skimage
from skimage import io
from skimage.color import rgb2gray
from os import walk
import matplotlib.pyplot as plt
import os	
import imageio
import OpenEXR
import Imath
import copy
import sys
import argparse
import scipy
from PIL import Image
import pdb
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk
selem = disk(6)

st = pdb.set_trace

def bilateral_filter_gray(im_spatial, im_intensity, sigma_spatial = 5, sigma_intensity = 5) :
    filtered_image = np.zeros(im_spatial.shape)
    #window_size = np.ceil(sigma_spatial + 1)
    window_size = 5

    [x,y] = np.meshgrid(np.array(np.arange(-window_size, window_size)), np.array(np.arange(-window_size, window_size)))
    spatial_filter = np.exp(-((x**2)+(y**2))/(2*sigma_spatial*sigma_spatial))
    spatial_filter = spatial_filter/np.sum(spatial_filter)

    for i in range(5, im_spatial.shape[0]-5) :
        for j in range(5,im_spatial.shape[1]-5) :
            im_top = int(np.max([i-window_size, 0]))
            im_bottom = int(np.min([(i+window_size),np.shape(im_spatial)[0]]))
            im_left = int(np.max([j-window_size,0]))
            im_right = int(np.min([(j+window_size),np.shape(im_spatial)[1]]))
            im_part = im_intensity[im_top:im_bottom,im_left:im_right]
            intensity_filter = np.exp(-(im_part-im_intensity[i,j])**2/(2*sigma_intensity*sigma_intensity))
            combined_filter = intensity_filter * spatial_filter

            im_ambient_part = im_spatial[im_top:im_bottom,im_left:im_right]
            filtered_image[i,j] = np.sum(combined_filter * im_ambient_part)/(np.sum(combined_filter))

    return filtered_image

def bilateral_filter(im_spatial, im_intensity, sigma_spatial = 2, sigma_intensity = 0.1) :
    image_shape = np.shape(im_spatial)
    image_dim_shape = np.shape(image_shape)[0]
    if image_dim_shape == 3 :
        im_spatial_red = bilateral_filter_gray(im_spatial[:,:,0], im_intensity[:,:,0], sigma_spatial, sigma_intensity)
        im_spatial_green = bilateral_filter_gray(im_spatial[:,:,1], im_intensity[:,:,1], sigma_spatial, sigma_intensity)
        im_spatial_blue = bilateral_filter_gray(im_spatial[:,:,2], im_intensity[:,:,2], sigma_spatial, sigma_intensity)
        filtered_image = np.zeros(np.shape(im_spatial))
        filtered_image[:,:,0] = im_spatial_red
        filtered_image[:,:,1] = im_spatial_green
        filtered_image[:,:,2] = im_spatial_blue
        #filtered_image = filtered_image/np.max(filtered_image)

    else :
        filtered_image = bilateral_filter_gray(im_spatial, im_intensity, sigma_spatial, sigma_intensity)

    #plt.imshow(im_spatial)
    #plt.show()
    #plt.imshow(filtered_image)
    #plt.show()
    return filtered_image

def get_linear_image(im) :
    zero_points = np.where(im <= 0.0404482)
    one_points = np.where(im > 0.0404482)

    im2 = np.copy(im)

    im2[zero_points] = im[zero_points]/12.92
    im2[one_points] = ((im[one_points] + 0.055)/1.055)**2.4
    return im2

def get_details(im_flash, im_ambient) :
    bilateral_image_flash = bilateral_filter(im_flash, im_flash)
    detail_image = (im_flash + 0.0001)/(bilateral_image_flash + 0.0001)
    plt.imshow(detail_image)
    plt.show()
    detailed_image = im_ambient * detail_image
    return detailed_image

def get_mask(im_flash, im_ambient) :

    im_flash_linear = get_linear_image(im_flash)
    im_ambient_linear = get_linear_image(im_ambient)

    im_flash_grayscale = rgb2gray(im_flash)
    im_ambient_grayscale = rgb2gray(im_ambient)

    mask_image = np.zeros(np.shape(im_ambient_grayscale))
    one_regions_specular = np.where(im_flash_grayscale > 0.98)
    one_regions_shadow = np.where(np.abs(im_flash_grayscale - im_ambient_grayscale) < 0.0001)
    mask_image[one_regions_shadow] = 1
    mask_image[one_regions_specular] = 1
    mask_image_closed = closing(mask_image, selem)
    return mask_image_closed

def get_flash_image(im_no_flash, im_flash, sigma_s = 2, sigma_i = 0.1) :
    im_no_flash_base = bilateral_filter(im_no_flash, im_no_flash, sigma_s, sigma_i)
    bilateral_im_diff = np.abs(im_no_flash - im_no_flash_base)
    plt.imshow(bilateral_im_diff)
    plt.show()
    plt.imshow(im_no_flash_base)
    plt.show()
    #plt.imsave('bilateral_diff.png', bilateral_im_diff)
    #plt.imsave('bilateral_output.png', im_no_flash_base)
    im_no_flash_nr = bilateral_filter(im_no_flash, im_flash, sigma_s, sigma_i)
    joint_bilateral_diff = np.abs(im_no_flash_nr - im_no_flash)
    plt.imshow(joint_bilateral_diff)
    plt.show()
    plt.imshow(im_no_flash_nr)
    plt.show()
    #plt.imsave('joint_bilateral_diff.png', joint_bilateral_diff)
    #plt.imsave('joint_bilateral_output.png', im_no_flash_nr)
    im_details = get_details(im_flash, im_no_flash_nr)
    plt.imshow(im_details)
    plt.show()
    detail_im_diff = np.abs(im_details - im_no_flash)
    plt.imshow(detail_im_diff)
    plt.show()
    #plt.imsave('Details_diff.png', detail_im_diff)
    #plt.imsave('Details_with_joint_bilateral.png', im_details)
    mask_image = get_mask(im_flash, im_no_flash)
    mask_image_rgb = np.zeros(np.shape(im_no_flash))
    mask_image_rgb[:,:,0] = mask_image
    mask_image_rgb[:,:,1] = mask_image
    mask_image_rgb[:,:,2] = mask_image
    final_flash_image = (1-mask_image_rgb)*im_details + mask_image_rgb*(im_no_flash_base)
    final_diff = np.abs(im_no_flash - final_flash_image)
    plt.imshow(final_diff)
    plt.show()
    plt.imshow(final_flash_image)
    plt.show()

    return final_flash_image

def preprocess_image(im) :
    return im/255


def boundary_condition(current_im, boundary_im) :
    current_im[:,0] = boundary_im[:,0]
    current_im[:,-1] = boundary_im[:,-1]
    current_im[0,:] = boundary_im[0,:]
    current_im[-1,:] = boundary_im[-1,:]

    return current_im

def get_gradients(im) :
    filter_x = np.array([[1,-1]])
    filter_y = np.array([[1],[-1]])
    im_x = scipy.signal.convolve2d(im, filter_x, mode = 'same')
    im_y = scipy.signal.convolve2d(im, filter_y, mode = 'same')
    return im_x, im_y  

def get_divergence(im_gradient_x, im_gradient_y) :
    filter_x = np.array([[1,-1]])
    filter_y = np.array([[1],[-1]])
    im_xx = scipy.signal.convolve2d(im_gradient_x, filter_x, mode = 'same')
    im_yy = scipy.signal.convolve2d(im_gradient_y, filter_y, mode = 'same')
    im_div = im_xx + im_yy
    return im_div

def conjugate_gradient(I_delta, I_init, B, epsilon = 0.0001, N = 2000) :
    I = boundary_condition(I_init, B)
    laplace_filter = [[0,1,0],[1,-4,1],[0,1,0]]
    r = I_delta - scipy.signal.convolve2d(I, laplace_filter, mode = 'same')
    d = r
    delta_new = np.sum(r*r)
    n = 0
    norm_r = np.linalg.norm(r)
    while norm_r > epsilon and n < N :
        print (n)
        #q = scipy.signal.convolve2d(d, laplace_filter, mode = 'same')
        q = scipy.signal.convolve2d(d, laplace_filter, mode = 'same')
        eta = delta_new/np.sum(d*q)
        I = boundary_condition(I + eta*d, B)
        r = r - eta*q
        delta_old = delta_new
        delta_new = np.sum(r*r)
        beta = delta_new/delta_old
        d = r + beta*d
        n = n+1
        norm_r = np.linalg.norm(r)
    return I

def differentiate_and_integrate(im) :
    I_delta = np.zeros(np.shape(im)[0:2])
    I_init = np.zeros(np.shape(im)[0:2])
    I_final = np.zeros(np.shape(im))
    laplace_filter = [[0,1,0],[1,-4,1],[0,1,0]]
    for i in range(np.shape(im)[2]) :
        I_delta = scipy.signal.convolve2d(im[:,:,i], laplace_filter, mode = 'same')
        I_final[:,:,i] = conjugate_gradient(I_delta, I_init, im[:,:,i])
    return I_final

def get_coherency_map(im_ambient, im_flash) :
    delta_a_x, delta_a_y = get_gradients(im_ambient)
    delta_phi_x, delta_phi_y = get_gradients(im_flash)
    numerator = np.abs(delta_phi_x * delta_a_x + delta_phi_y * delta_a_y)
    denominator = np.sqrt(delta_phi_x**2 + delta_phi_y**2)*np.sqrt(delta_a_x**2 + delta_a_y**2) + 0.0001
    coherency_map = numerator/denominator
    return coherency_map

def get_saturation_map(im_flash, sigma = 40, thresh = 0.9) :
    saturation_map = np.tanh(sigma*(im_flash - thresh))
    saturation_map = (saturation_map - np.min(saturation_map))/(np.max(saturation_map) - np.min(saturation_map))

    return saturation_map

def fused_gradient_field(saturation_map, coherency_map, gradient_field_ambient, gradient_field_flash) :
    return saturation_map*gradient_field_ambient + (1 - saturation_map) * (coherency_map * gradient_field_flash + (1 - coherency_map) * gradient_field_ambient)


def get_flash(im_no_flash, im_flash, t, s) :
    final_im = np.ones(np.shape(im_flash))
    I_init = np.zeros(np.shape(im_flash)[0:2])
    fused_gradient_all_channels_x = np.ones(np.shape(im_flash))
    fused_gradient_all_channels_y = np.ones(np.shape(im_flash))
    ambient_gradient_all_channels_x = np.ones(np.shape(im_flash))
    ambient_gradient_all_channels_y = np.ones(np.shape(im_flash))
    flash_gradient_all_channels_x = np.ones(np.shape(im_flash))
    flash_gradient_all_channels_y = np.ones(np.shape(im_flash))

    for i in range(3) :
        im_x_flash, im_y_flash = get_gradients(im_flash[:,:,i])
        im_x_ambient, im_y_ambient = get_gradients(im_no_flash[:,:,i])
        coherency_map = get_coherency_map(im_no_flash[:,:,i], im_flash[:,:,i])
        saturation_map = get_saturation_map(im_flash[:,:,i], s, t)
        fused_gradient_field_x = fused_gradient_field(saturation_map, coherency_map, im_x_ambient, im_x_flash)
        fused_gradient_field_y = fused_gradient_field(saturation_map, coherency_map, im_y_ambient, im_y_flash)
        gradient_divergence = get_divergence(fused_gradient_field_x, fused_gradient_field_y)
        final_im[:,:,i] = conjugate_gradient(gradient_divergence, I_init, im_flash[:,:,i])

        fused_gradient_all_channels_x[:,:,i] = fused_gradient_field_x*10
        fused_gradient_all_channels_y[:,:,i] = fused_gradient_field_y*10
        ambient_gradient_all_channels_x[:,:,i] = im_x_ambient*10
        ambient_gradient_all_channels_y[:,:,i] = im_y_ambient*10
        flash_gradient_all_channels_x[:,:,i] = im_x_flash*10
        flash_gradient_all_channels_y[:,:,i] = im_y_flash*10


    image_shape = np.shape(im_no_flash)
    if image_shape[2] == 4 :
        final_im[:,:,3] = im_flash[:,:,3]
        
        fused_gradient_all_channels_x[:,:,3] = im_flash[:,:,3]
        fused_gradient_all_channels_y[:,:,3] = im_flash[:,:,3]
        ambient_gradient_all_channels_x[:,:,3] = im_flash[:,:,3]
        ambient_gradient_all_channels_y[:,:,3] = im_flash[:,:,3]
        flash_gradient_all_channels_x[:,:,3] = im_flash[:,:,3]
        flash_gradient_all_channels_x[:,:,3] = im_flash[:,:,3]

    if t == 0.9 and s == 40 :
        plt.imshow(fused_gradient_all_channels_x)
        plt.show()
        plt.imshow(fused_gradient_all_channels_y)
        plt.show()
        plt.imshow(ambient_gradient_all_channels_x)
        plt.show()
        plt.imshow(ambient_gradient_all_channels_y)
        plt.show()
        plt.imshow(flash_gradient_all_channels_x)
        plt.show()
        plt.imshow(flash_gradient_all_channels_y)
        plt.show()

    return final_im


im_no_flash = io.imread('./data/maheshwari/maheshwari_no_flash.JPG')
im_flash = io.imread('./data/maheshwari/maheshwari.JPG')

final_im_bilateral = get_flash_image(im_ambient, im_flash, 1, 0.4)
plt.imshow(final_im_bilateral)
plt.show()
final_im_gradient= get_flash(im_ambient, im_flash, 0.9, 40)
plt.imshow(final_im_gradient)
plt.show()
