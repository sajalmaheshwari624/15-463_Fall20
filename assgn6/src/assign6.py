import numpy as np
import matplotlib.pyplot as plt
import cp_hw6
import skimage
import skimage.filters
import matplotlib.pyplot as plt
import os
import imageio
import copy
import sys
import argparse
import scipy
import random
import copy
from PIL import Image
from skimage import io
from skimage.color import rgb2gray, rgb2xyz
from skimage.transform import resize
from scipy.interpolate import interp2d
#import open3d as o3d

import pdb
st = pdb.set_trace

def is_image_file(filename, IMG_Extension):
    return any(filename.endswith(extension) for extension in IMG_Extension)

def get_delta(data_folder, IMG_Extension) :
    im_stack = []
    for fname in sorted(os.listdir(data_folder)) :
        if is_image_file(fname, IMG_Extension) :
            im = rgb2gray(io.imread(os.path.join(data_folder, fname)))
            im_stack.append(im)
    input_stack = np.stack(im_stack)
    im_max = np.max(input_stack, axis = 0)
    im_min = np.min(input_stack, axis = 0)
    im_shadow = (im_max + im_min)/2
    im_shadow = np.expand_dims(im_shadow, axis = 0)
    im_shadow_repeat = np.repeat(im_shadow, len(im_stack), axis = 0)
    im_delta_stack = input_stack - im_shadow_repeat
    return im_delta_stack, im_max, im_min

#def get_line_from_pts(x_points, y_points) :
def get_subpixel_spatial(y_values, x_values, delta_image) :
    x_values_minus = x_values - 1
    x_values_plus = x_values + 1
    delta_im_minus = delta_image[y_values, x_values_minus]
    delta_im_plus = delta_image[y_values, x_values_plus]
    slope = delta_im_plus - delta_im_minus / 2
    const = delta_im_plus - slope*x_values_plus
    x_values_subpix = -const/(slope + 1e-5)
    return x_values_subpix

def get_line_from_points(x_values, y_values, num_iter = 500, thresh = 4) :
    num_points = np.size(y_values)
    A_full = np.ones((num_points,2))
    b_full = y_values.reshape(num_points,1)
    A_full[:,0] = x_values

    best_line = np.ones((2,1))
    least_diff = 0
    for idx in range(num_iter) :
        indices = random.sample(range(num_points), 2)
        A_mat = np.ones((2,2))
        b_vec = np.ones((2,1))
        A_mat[0,0] = x_values[indices[0]]
        A_mat[1,0] = x_values[indices[1]]
        b_vec[0,0] = y_values[indices[0]]
        b_vec[1,0] = y_values[indices[1]]
        ans,_,_,_ = np.linalg.lstsq(A_mat, b_vec)
        y_values_full = np.matmul(A_full, ans)
        diff_values = np.abs(b_full - y_values_full)
        inlier_indices = np.where(diff_values < thresh)
        if np.size(inlier_indices) > least_diff :
            least_diff = np.size(inlier_indices[0])
            best_line = ans

    if (np.max(x_values) - np.min(x_values)) < thresh :
        best_line[0,0] = None
        best_line[1,0] = np.mean(x_values)
    return best_line

def spatial_edges(im_stack, outlier_bounds, search_bounds, mtx, dist, RMat, tVec) :
    num_images = im_stack.shape[0]
    one_indices = np.where(im_stack < -20/255)
    zero_indices = np.where(im_stack > -20/255)
    im_stack[zero_indices] = 0
    im_stack[one_indices] = 1
    all_left_lines = []
    all_right_lines = []
    all_3d_points = []
    for idx in range(num_images) :
        print (idx)
        current_delta_image = im_stack[idx,:,:]
        gradient_values = np.gradient(current_delta_image, axis = 1)
        gradient_values[:,0:int(outlier_bounds[0]*current_delta_image.shape[1])] = 0
        gradient_values[:,int(outlier_bounds[1]*current_delta_image.shape[1]):] = 0
        y_values = np.arange(search_bounds[0],search_bounds[1])

        x_values_right = np.argmin(gradient_values, axis = 1)
        x_values_left = np.argmax(gradient_values, axis = 1)
        x_values_left = x_values_left[search_bounds[0]:search_bounds[1]]
        x_values_right = x_values_right[search_bounds[0]:search_bounds[1]]

        x_values_left_subpix = get_subpixel_spatial(y_values, x_values_left, current_delta_image)
        x_values_right_subpix = get_subpixel_spatial(y_values, x_values_right, current_delta_image)
        # plt.imshow(current_delta_image)
        # plt.scatter(x_values_left_subpix, y_values)
        # plt.scatter(x_values_right_subpix, y_values)
        # plt.show(block = False)
        # plt.pause(0.3)
        # plt.close()
        left_line = get_line_from_points(x_values_left_subpix, y_values, num_iter = 100, thresh = 4)
        right_line= get_line_from_points(x_values_right_subpix, y_values, num_iter = 100, thresh = 4)
        #coords_on_image(gradient_values, left_line, left_line, left_line, left_line)
        all_left_lines.append(left_line)
        all_right_lines.append(right_line)
        points = np.ones((2,2))
        points[0,0] = x_values_left_subpix[0]
        points[0,1] = y_values[0]
        points[1,0] = x_values_left_subpix[-1]
        points[1,1] = y_values[-1]
        points_3d = get_point_in_plane(points,mtx,dist, RMat, tVec)
        all_3d_points.append(points_3d)

    return all_left_lines, all_right_lines, all_3d_points

def temporal_edges(im_stack, im_max, im_min, total_timestamps, thresh = 30) :
    contrast= np.abs(im_max - im_min)
    im_stack_copy = im_stack
    indices = np.where(im_stack < -20/255)
    im_stack_copy[indices] = 1
    gradient = np.gradient(im_stack, axis = 0)
    time_values = np.argmax(gradient, axis = 0)

    time_values_left = time_values - 1
    time_values_right = time_values + 1

    temporal_im = np.zeros(time_values.shape)

    for i in range(im_max.shape[0]) :
        for j in range(im_max.shape[1]) :
            delta_values_left = im_stack_copy[max(time_values_left[i,j],0), i, j]
            delta_values_right = im_stack_copy[min(time_values_right[i,j], im_stack.shape[0]-1), i, j]
            slope = (delta_values_right - delta_values_left)/2
            const = delta_values_right - slope*time_values_right[i,j]
            temporal_im[i,j] = -const/(slope + 1e-5)
    temporal_im[contrast < thresh/255.] = 0
    temporal_im[temporal_im < 0] = 0
    temporal_im[temporal_im > total_timestamps] = total_timestamps
    temporal_im_vis = temporal_im
    temporal_im_vis = (temporal_im_vis - np.min(temporal_im_vis))/(np.max(temporal_im_vis) - np.min(temporal_im_vis))
    temporal_im_vis = (temporal_im_vis * 32).astype(int)
    plt.imshow(temporal_im_vis, cmap = 'jet')
    plt.show()
    return temporal_im

def visualize(data_folder, all_left_lines_h, all_left_lines_v, all_right_lines_h, all_right_lines_v) :
    count = 0
    for fname in sorted(os.listdir(data_folder)) :
        if is_image_file(fname, IMG_Extension) :
            im = io.imread(os.path.join(data_folder, fname))
            y_values = np.arange(im.shape[0])
            left_h_params = all_left_lines_h[count]
            if np.isnan(left_h_params[0,0]):
                left_h_x_values = np.ones(im.shape[0])*left_h_params[1,0]
            else :
                left_h_x_values = (y_values - left_h_params[1,0])/(left_h_params[0,0] + 1e-5)

            left_v_params = all_left_lines_v[count]
            if np.isnan(left_v_params[0,0]) :
                left_v_x_values = np.ones(im.shape[0])*left_v_params[1,0]
            else :
                left_v_x_values = (y_values - left_v_params[1,0])/(left_v_params[0,0] + 1e-5)
            
            diff_left = np.abs(left_h_x_values - left_v_x_values)
            intersection_left = np.argmin(diff_left)

            right_h_params = all_right_lines_h[count]
            if np.isnan(right_h_params[0,0]) :
                right_h_x_values = np.ones(im.shape[0])*right_h_params[1,0]
            else :
                right_h_x_values = (y_values - right_h_params[1,0])/(right_h_params[0,0] + 1e-5)

            right_v_params = all_right_lines_v[count]
            if np.isnan(right_v_params[0,0]) :
                right_v_x_values = np.ones(im.shape[0])*right_v_params[1,0]
            else :
                right_v_x_values = (y_values - right_v_params[1,0])/(right_v_params[0,0] + 1e-5)
            
            diff_right = np.abs(right_h_x_values - right_v_x_values)
            intersection_right = np.argmin(diff_right)
            print (intersection_left, intersection_right)
            left_v_x_values = left_v_x_values[0:intersection_left]
            right_v_x_values = right_v_x_values[0:intersection_right]
            left_h_x_values = left_h_x_values[intersection_left:]
            right_h_x_values = right_h_x_values[intersection_right:]

            plt.imshow(im)
            plt.plot(left_v_x_values, y_values[0:intersection_left], 'g')
            plt.plot(right_v_x_values, y_values[0:intersection_right], 'b')
            plt.plot(left_h_x_values, y_values[intersection_left:], 'g')
            plt.plot(right_h_x_values, y_values[intersection_right:], 'b')
            plt.show(block = False)
            plt.pause(1)
            plt.close()
            count += 1

def get_point_in_plane(points, mtx, dist, RMat, tVec) :
    camera_frame_ray = cp_hw6.pixel2ray(points, mtx, dist)
    #camera_frame_ray = camera_frame_ray[0,0,:].reshape(3,1)
    camera_frame_ray = np.squeeze(camera_frame_ray, axis = 1)
    multiplier = np.sum(RMat.T[2,:].reshape(3,1)*camera_frame_ray.T, axis = 0)
    const_term = np.sum(RMat.T[2,:].reshape(3,1)*tVec.reshape(3,1))
    coefficient = const_term/multiplier
    world_plane_point = coefficient * camera_frame_ray.T
    return world_plane_point

'''
def get_point_in_plane(point, mtx, dist, rotation, translation):
    
    camera_center = np.matmul(rotation.T, (np.zeros(3).reshape(3,1) - translation))
    
    ray = cp_hw6.pixel2ray(point, mtx, dist)[0]
    
    ray = np.matmul(rotation.T, np.squeeze(ray).T)

    start_point = camera_center.T[0]
    lamda = -start_point[2]/ray[2]
    
    point_3d = (start_point + lamda*(ray))
    point_3d = np.matmul(rotation, point_3d.reshape(3,1)) + translation
    point_3d = point_3d.squeeze()
        
    return point_3d
'''

def plot_plane(normal, p1, p2, p3, p4):
    d = -np.dot(normal.T, p1)
    xx, yy = np.meshgrid(np.arange(2000)-1000, np.arange(2000)-1000)
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    # plt3d = plt.figure().gca(projection='3d')
    ax.plot_surface(xx, yy, z, alpha=0.2)
    # ax = plt.gca()
    # ax.hold(True)
    ax.scatter(p1[0], p1[1], p1[2], color='green')
    ax.scatter(p2[0], p2[1], p2[2], color='red')
    ax.scatter(p3[0], p3[1], p3[2], color='blue')
    ax.scatter(p4[0], p4[1], p4[2], color='black')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def get_plane_from_points(points_v, points_h) :
    num_planes = len(points_v)
    all_planes = []
    for idx in range(num_planes) :
        vert_points = points_v[idx]
        horz_points = points_h[idx]
        vert_diff = np.diff(vert_points, axis = 1)
        horz_diff = np.diff(horz_points, axis = 1)
        normal_vec = np.cross(vert_diff.T, horz_diff.T)
        normal_vec = normal_vec/np.linalg.norm(normal_vec)
        normal = normal_vec.reshape(3,1)
        p1 = vert_points[:,0].reshape(3,1)
        p2 = vert_points[:,1].reshape(3,1)
        p3 = horz_points[:,0].reshape(3,1)
        p4 = horz_points[:,1].reshape(3,1)
        plane_params = [normal_vec, vert_points[:,0]]
        all_planes.append(plane_params)
    return all_planes

def reconstruct_shape(start_row, end_row, start_column, end_column, temporal_im, plane_params_list, mtx, dist, im_zero, total_timestamps) :
    points_3d_all = np.zeros((3,1))
    color_list = []
    for i in range(start_row, end_row) :
        for j in range(start_column, end_column) :
            timestamp = temporal_im[i,j]
            left_w = timestamp - np.floor(timestamp)
            right_w = np.ceil(timestamp) - timestamp
            if np.ceil(timestamp) > total_timestamps :
                right_w = 0
            normal_vector_left = plane_params_list[int(np.floor(timestamp))][0]
            normal_vector_right = plane_params_list[min(int(np.ceil(timestamp)),total_timestamps-1)][0]
            normal_vector = left_w*normal_vector_left + right_w*normal_vector_right

            point_vector_left = plane_params_list[int(np.floor(timestamp))][1]
            point_vector_right = plane_params_list[int(np.ceil(timestamp))][1]
            point_vector = left_w*point_vector_left + right_w*point_vector_right

            point_2d = np.array([j,i])
            point_2d = point_2d.astype('float32')
            ray = cp_hw6.pixel2ray(point_2d, mtx, dist)
            ray = ray.reshape(3,1)
            point_vector = point_vector.reshape(3,1)
            normal_vector = normal_vector.reshape(3,1)
            coefficient = np.sum(point_vector * normal_vector) / np.sum(ray * normal_vector)
            point_3d = coefficient * ray
            if point_3d[2,0] < -2500 or point_3d[2,0] > 1500 or point_3d[1,0]>500 or point_3d[1,0] < -250 or point_3d[0,0] < -500:
                continue
            if point_3d[2,0] < 1000 or point_3d[2,0] > 2500:
                continue
            points_3d_all = np.hstack((points_3d_all, point_3d))
            color_list.append(im_zero[i,j])

    return points_3d_all[:,1:], color_list

def get_intrinsics(filename) :
    intrinsics = np.load(filename)
    mtx = intrinsics['mtx']
    dist = intrinsics['dist']
    return mtx, dist

def get_extrinsics(filename) :
    extrinsics = np.load(filename)
    tvec_h = extrinsics['tvec_h']
    rmat_h = extrinsics['rmat_h']
    tvec_v = extrinsics['tvec_v']
    rmat_v = extrinsics['rmat_v']
    return tvec_h, rmat_h, tvec_v, rmat_v

data_folder = '../data/frog/v2-lr/'
edge_bounds = [0.12, 0.88]
IMG_Extension = ['.png','.jpg']
im_zero = rgb2gray(io.imread(os.path.join(data_folder, '000001.jpg')))
mtx, dist = get_intrinsics('../data/frog-lr-v2/intrinsic_calib.npz')
tvec_h, rmat_h, tvec_v, rmat_v = get_extrinsics('../data/frog-lr-v2/extrinsic_calib.npz')
im_delta_stack_h, im_max, im_min = get_delta(data_folder, IMG_Extension)
im_delta_stack_v, im_max, im_min = get_delta(data_folder, IMG_Extension)
im_delta_stack_t, im_max, im_min = get_delta(data_folder, IMG_Extension)

# hr = 20, 75, 700, 760
#lr = 20,75,330,380
search_bounds_v = [20, 75]
search_bounds_h = [330, 380]
all_left_lines_v, all_right_lines_v, all_3d_points_v = spatial_edges(im_delta_stack_v, edge_bounds,search_bounds_v, mtx, dist, rmat_v, tvec_v)
all_left_lines_h, all_right_lines_h, all_3d_points_h = spatial_edges(im_delta_stack_h, edge_bounds,search_bounds_h, mtx, dist, rmat_h, tvec_h)
world_3d_points = {"vert_points":all_3d_points_v, "horz_points":all_3d_points_h}
np.savez(os.path.join(data_folder,'world_3d_points.npz'), **world_3d_points)
total_timestamps = len(all_left_lines_v)-1
temporal_im = temporal_edges(im_delta_stack_t, im_max, im_min, total_timestamps)
all_plane_params = get_plane_from_points(all_3d_points_v, all_3d_points_h)
np.savez(os.path.join(data_folder,'plane_params.npz'), all_plane_params)
#v1-lr 150,350,150,420
#v1 300,700,300,840
#v2-lr 
all_3d_points, color_list = reconstruct_shape(150,350,150,420, temporal_im, all_plane_params, mtx, dist, im_zero, total_timestamps)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(all_3d_points[0,:],all_3d_points[1,:],all_3d_points[2,:], marker='o', s=0.5, c=np.array(color_list), cmap="gray")
ax.set_xlim3d(-300, 400)
ax.set_ylim3d(-0, 400)
ax.set_zlim3d(1400, 2400)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
visualize(data_folder, all_left_lines_h, all_left_lines_v, all_right_lines_h, all_right_lines_v)