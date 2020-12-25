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
from PIL import Image
import pdb
st = pdb.set_trace

def gamma_correct(im) :
	im = im/255
	thresh_less = np.where(im < 0.04045)
	thresh_higher = np.where(im >= 0.04045)
	im[thresh_less] = im[thresh_less]/12.92
	im[thresh_higher] = ((im[thresh_higher] + 0.055)/1.055)**2.4
	im = im*255
	return im

def ind2sub(array_shape, ind):
    rows = (ind.astype('int') // array_shape[1])
    cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return (rows, cols)

class HW4() :
	def __init__(self, image_path, unstructure_image_path, unstructure_folder_path, self_focal_path) :

		self.image_path = image_path
		self.image = io.imread(image_path)
		self.u_length = 16
		self.v_length = 16
		self.lf_image = self.convert_to_lightfield()


		#self.get_sub_aperture()
		#self.test_far_focus()


		self.depth_values = np.arange(-1.5,0.1,0.1)
		self.aperture_values = np.arange(16,2,-2)
		#self.focal_stack = self.generate_focal_stack(self.lf_image)
		#aif_image, depth_map = self.get_aif_and_depth()
		self.get_aperture_focal_stack()

		# map_1 = self.laplacian_energy()
		# map_2 = self.laplacian_modified()
		# map_3 = self.spatial_frequency_response()
		# map_4 = self.gray_level_variance()
		# self.final_map = (map_1 + map_2 + map_3 + map_4)
		# aif_image, depth_map = self.get_smart_aif_and_depth()

		# self.unstructure_image_path = unstructure_image_path
		# self.unstructure_folder_path = unstructure_folder_path
		# self.template_image, self.window_row_start, self.window_col_start = self.get_template()
		# self.get_cross_correlation()

		# self.self_focal_path = self_focal_path
		# self.self_focal_stack = self.get_self_stack()
		# self_depth_image, self_aif_image = self.get_self_aif_and_depth()

	def convert_to_lightfield(self) :
		im = self.image
		s_length = int(np.shape(im)[0]/self.u_length)
		t_length = int(np.shape(im)[1]/self.v_length)
		#light_field_image = np.ones((u_length, v_length, s_length, t_length, 3))
		light_field_image = np.ones((self.u_length, self.v_length, s_length, t_length, 3))
		for i in range(s_length) :
			for j in range(t_length) :
				light_field_image[:,:,i,j,:] = im[i*16:(i+1)*16, j*16:(j+1)*16,:]
		return light_field_image

	def get_sub_aperture(self) :
		sub_aperture_images = np.zeros(self.image.shape)
		s_length = int(np.shape(self.image)[0]/self.u_length)
		t_length = int(np.shape(self.image)[1]/self.v_length)
		for i in range(self.u_length) :
			for j in range(self.v_length) :
				sub_aperture_images[i*s_length:(i+1)*s_length, j*t_length:(j+1)*t_length,:] = self.lf_image[i,j,:,:,:]
		plt.imshow(sub_aperture_images/255)
		plt.show()

	def test_far_focus(self) :
		s_length = int(np.shape(self.image)[0]/self.u_length)
		t_length = int(np.shape(self.image)[1]/self.v_length)
		far_focus_image = np.zeros((s_length, t_length, 3))
		num_stacks = self.u_length*self.v_length
		for i in range(self.u_length) :
			for j in range(self.v_length) :
				far_focus_image = far_focus_image + self.lf_image[i,j,:,:,:]/num_stacks
		#plt.imshow(far_focus_image/255)
		#plt.show()

	def generate_focal_stack(self, lf_image, u_limit = 16, v_limit = 16) :
		x_range = np.arange(0, lf_image.shape[2],1)
		y_range = np.arange(0, lf_image.shape[3],1)
		s_length = int(np.shape(self.image)[0]/self.u_length)
		t_length = int(np.shape(self.image)[1]/self.v_length)
		focal_stack = np.zeros((len(self.depth_values), s_length, t_length, 3))
		maxUV = (self.u_length - 1) / 2
		u = np.arange(self.u_length) - maxUV
		v = np.arange(self.v_length) - maxUV;

		for index,depth in enumerate(self.depth_values) :
			focal_stack_image = np.zeros((s_length, t_length, 3))
			for i in range(u_limit) :
				for j in range(v_limit) :
					current_im = lf_image[i,j,:,:,:]
					interp_function_red = interp2d(y_range, x_range, current_im[:,:,0], kind = 'linear')
					interp_function_green = interp2d(y_range, x_range, current_im[:,:,1], kind = 'linear')
					interp_function_blue = interp2d(y_range, x_range, current_im[:,:,2], kind = 'linear')
					interp_image_red = interp_function_red(np.arange(0 + depth*v[j], t_length + depth*v[j], 1), np.arange(0 - depth*u[i], s_length - depth*u[i], 1))
					interp_image_green = interp_function_green(np.arange(0 + depth*v[j], t_length + depth*v[j], 1), np.arange(0 - depth*u[i], s_length - depth*u[i], 1))
					interp_image_blue = interp_function_blue(np.arange(0 + depth*v[j], t_length + depth*v[j], 1), np.arange(0 - depth*u[i], s_length  - depth*u[i], 1))
					im_rgb = np.dstack((interp_image_red, interp_image_green, interp_image_blue))
					focal_stack_image += im_rgb/(u_limit * v_limit)
			# plt.imshow(focal_stack_image/255)
			# plt.title(str(depth))
			# plt.show()
			focal_stack[index,:,:,:] = focal_stack_image
		return focal_stack

	def get_aif_and_depth(self) :
		num_images = len(self.depth_values)
		s_length = int(np.shape(self.image)[0]/self.u_length)
		t_length = int(np.shape(self.image)[1]/self.v_length)
		sharpness_maps = np.zeros((num_images, s_length, t_length))

		for i in range(num_images) :
			focal_slice = self.focal_stack[i,:,:,:]
			gamma_focal_slice = gamma_correct(focal_slice)
			focal_slice_xyz = rgb2xyz(gamma_focal_slice)
			focal_slice_luminance = focal_slice_xyz[:,:,1]
			low_freq_image = skimage.filters.gaussian(focal_slice_luminance, sigma=0.5)
			high_freq_image = focal_slice_luminance - low_freq_image
			high_freq_image = high_freq_image ** 2
			sharpness_map = skimage.filters.gaussian(high_freq_image, sigma=4)
			sharpness_maps[i,:,:] = sharpness_map

		total_sharpness_sum = np.sum(sharpness_maps, axis = 0)
		total_sharpness_sum_stacked = np.dstack((total_sharpness_sum, total_sharpness_sum, total_sharpness_sum))

		aif_image = np.zeros((s_length, t_length, 3))
		depth_map = np.zeros((s_length,t_length))
		for i in range(num_images) :
			focal_slice = self.focal_stack[i,:,:,:]
			sharpness_map = sharpness_maps[i,:,:]
			sharpness_map_stacked = np.dstack((sharpness_map, sharpness_map, sharpness_map))
			aif_image += sharpness_map_stacked * focal_slice / total_sharpness_sum_stacked
			depth_map += self.depth_values[i] * sharpness_map / total_sharpness_sum
		#plt.imshow(aif_image/255)
		#plt.show()
		#plt.imshow(depth_map, cmap = 'gray')
		#plt.show()
		return aif_image, depth_map

	def get_aperture_focal_stack(self) :
		lf_image = self.lf_image
		s_length = int(np.shape(self.image)[0]/self.u_length)
		t_length = int(np.shape(self.image)[1]/self.v_length)
		afi_pixels = [1000,41000,81000,121000,161000]
		aperture_focal_stack = np.zeros((len(self.aperture_values)*s_length, len(self.depth_values)*t_length, 3))
		for i,aper in enumerate(self.aperture_values) :
			print (i)
			filtered_focal_stack = self.generate_focal_stack(self.lf_image, aper, aper)
			for j in range(len(self.depth_values)) :
				aperture_focal_stack[i*s_length:(i+1)*s_length, j*t_length:(j+1)*t_length,:] = filtered_focal_stack[j,:,:,:]

		dense_depth_map = np.zeros((s_length, t_length))
		count = 0
		for i in range(s_length) :
			for j in range(t_length) :
				count += 1
				aperture_coords = np.arange(i,aperture_focal_stack.shape[0], s_length)
				focal_coords = np.arange(j, aperture_focal_stack.shape[1], t_length)
				focal_variance = np.zeros((len(self.depth_values),1))
				if count in afi_pixels :
					afi_image = np.zeros((len(self.depth_values), len(self.aperture_values)))
				for k,fc in enumerate(focal_coords) :
					current_shape = aperture_focal_stack[aperture_coords[:], fc, :].shape
					current_stack = aperture_focal_stack[aperture_coords[:], fc, :]
					current_stack_reshape = current_stack.reshape(1, current_shape[0], current_shape[1])
					aperture_pixels = rgb2xyz(current_stack_reshape)
					aperture_pixels = aperture_pixels[:,:,1]
					aperture_variance = np.var(aperture_pixels)
					focal_variance[k] = aperture_variance
					if count in afi_pixels :
						afi_image[k,:] = aperture_pixels

				if count in afi_pixels :
					plt.imshow(afi_image/255)
					plt.show()

				focus_level = np.argmin(focal_variance)
				dense_depth_map[i,j] = self.depth_values[focus_level]

		plt.imshow(aperture_focal_stack/255)
		plt.show()
		plt.imshow(dense_depth_map, cmap = 'gray')
		plt.show()

	def get_template(self) :
		im = io.imread(self.unstructure_image_path)
		plt.imshow(im)
		points = plt.ginput(4)
		plt.close()

		x0 = int(points[0][1])
		y0 = int(points[0][0])
		x1 = int(points[1][1])
		y1 = int(points[3][0])
		template = rgb2gray(im[x0:x1, y0:y1, :])
		plt.imshow(template)
		plt.show()
		return template, x0, y0

	def get_cross_correlation(self) :
		count = 0
		window_size = 200
		#for filename in os.listdir(self.unstructure_folder_path) :
		for i in range(60) :
			#52i
			count += 1
			print (count)
			current_image_path = self.unstructure_folder_path + str(count) + '.jpg'
			#current_image_path = self.unstructure_folder_path + str(count) + '.png'
			current_im = rgb2gray(io.imread(current_image_path))
			current_im = current_im[self.window_row_start - window_size: self.window_row_start + window_size, self.window_col_start - window_size : self.window_col_start + window_size]
			correlation_output = scipy.signal.correlate2d(current_im - np.mean(current_im), self.template_image - np.mean(self.template_image), mode = 'same')
			max_index = np.argmax(correlation_output)
			row_val, col_val = ind2sub(current_im.shape, max_index)
			row_val = row_val + self.window_row_start - window_size
			col_val = col_val + self.window_col_start - window_size
			if count == 1 :
				initial_row_val = row_val
				initial_col_val = col_val
				initial_image = (io.imread(current_image_path))
				initial_image = initial_image.astype(float)
			else :
				shift_row = row_val - initial_row_val
				shift_col = col_val - initial_col_val

				rgb_im = io.imread(current_image_path)
				col_range = np.arange(0,rgb_im.shape[1])
				row_range = np.arange(0,rgb_im.shape[0])

				interp_function_red = interp2d(col_range, row_range, rgb_im[:,:,0], kind = 'linear')
				interp_function_green = interp2d(col_range, row_range, rgb_im[:,:,1], kind = 'linear')
				interp_function_blue = interp2d(col_range, row_range, rgb_im[:,:,2], kind = 'linear')

				interp_image_red = interp_function_red(np.arange(shift_col, shift_col + rgb_im.shape[1], 1), np.arange(shift_row, shift_row + rgb_im.shape[0], 1))
				interp_image_green = interp_function_green(np.arange(shift_col, shift_col + rgb_im.shape[1], 1), np.arange(shift_row, shift_row + rgb_im.shape[0], 1))
				interp_image_blue = interp_function_blue(np.arange(shift_col, shift_col + rgb_im.shape[1], 1), np.arange(shift_row, shift_row + rgb_im.shape[0], 1))

				new_shifted_image = np.dstack((interp_image_red, interp_image_green, interp_image_blue))
				initial_image += new_shifted_image

		initial_image = initial_image / count
		plt.imshow(initial_image/255)
		plt.show()


	def laplacian_energy(self) :
		num_images = len(self.depth_values)
		s_length = int(np.shape(self.image)[0]/self.u_length)
		t_length = int(np.shape(self.image)[1]/self.v_length)
		sharpness_maps = np.zeros((num_images, s_length, t_length))

		for i in range(num_images) :
			focal_slice = self.focal_stack[i,:,:,:]
			gamma_focal_slice = gamma_correct(focal_slice)
			focal_slice_xyz = rgb2xyz(gamma_focal_slice)
			focal_slice_luminance = focal_slice_xyz[:,:,1]
			laplace_image = scipy.ndimage.laplace(focal_slice_luminance)
			laplace_image_squared = laplace_image ** 2
			laplace_energy_image = scipy.ndimage.uniform_filter(laplace_image_squared, 3)
			sharpness_maps[i,:,:] = laplace_energy_image

		total_sharpness_sum = np.sum(sharpness_maps, axis = 0)
		sharpness_maps = sharpness_maps/total_sharpness_sum

		return sharpness_maps

	def spatial_frequency_response(self) :
		num_images = len(self.depth_values)
		s_length = int(np.shape(self.image)[0]/self.u_length)
		t_length = int(np.shape(self.image)[1]/self.v_length)
		sharpness_maps = np.zeros((num_images, s_length, t_length))

		for i in range(num_images) :
			focal_slice = self.focal_stack[i,:,:,:]
			gamma_focal_slice = gamma_correct(focal_slice)
			focal_slice_xyz = rgb2xyz(gamma_focal_slice)
			focal_slice_luminance = focal_slice_xyz[:,:,1]
			gradient_x = np.gradient(focal_slice_luminance, axis = 0)
			gradient_y = np.gradient(focal_slice_luminance, axis = 1)

			gradient_x = gradient_x**2
			gradient_y = gradient_y**2

			filtered_image_x = scipy.ndimage.uniform_filter(gradient_x, 3)
			filtered_image_y = scipy.ndimage.uniform_filter(gradient_y, 3)

			spatial_freq_output = np.sqrt(filtered_image_x, filtered_image_y)
			sharpness_maps[i,:,:] = spatial_freq_output


		total_sharpness_sum = np.sum(sharpness_maps, axis = 0)
		sharpness_maps = sharpness_maps/total_sharpness_sum

		return sharpness_maps

	def gray_level_variance(self) :
		num_images = len(self.depth_values)
		s_length = int(np.shape(self.image)[0]/self.u_length)
		t_length = int(np.shape(self.image)[1]/self.v_length)
		sharpness_maps = np.zeros((num_images, s_length, t_length))

		for i in range(num_images) :
			focal_slice = self.focal_stack[i,:,:,:]
			gamma_focal_slice = gamma_correct(focal_slice)
			focal_slice_xyz = rgb2xyz(gamma_focal_slice)
			focal_slice_luminance = focal_slice_xyz[:,:,1]
			low_freq_focal_slice_luminance = (focal_slice_luminance - scipy.ndimage.uniform_filter(focal_slice_luminance))**2
			sharpness_maps[i,:,:] = low_freq_focal_slice_luminance

		total_sharpness_sum = np.sum(sharpness_maps, axis = 0)
		sharpness_maps = sharpness_maps/total_sharpness_sum

		return sharpness_maps

	def laplacian_modified(self) :
		num_images = len(self.depth_values)
		s_length = int(np.shape(self.image)[0]/self.u_length)
		t_length = int(np.shape(self.image)[1]/self.v_length)
		sharpness_maps = np.zeros((num_images, s_length, t_length))

		for i in range(num_images) :
			focal_slice = self.focal_stack[i,:,:,:]
			gamma_focal_slice = gamma_correct(focal_slice)
			focal_slice_xyz = rgb2xyz(gamma_focal_slice)
			focal_slice_luminance = focal_slice_xyz[:,:,1]
			laplace_image_x = np.abs(scipy.signal.convolve2d(focal_slice_luminance, np.array([[-1,2,-1]]), mode = 'same'))
			laplace_image_y = np.abs(scipy.signal.convolve2d(focal_slice_luminance, np.array([[-1],[2],[-1]]), mode = 'same'))

			laplace_image = laplace_image_x + laplace_image_y
			laplace_energy_image = scipy.ndimage.uniform_filter(laplace_image, 3)
			sharpness_maps[i,:,:] = laplace_energy_image
		
		total_sharpness_sum = np.sum(sharpness_maps, axis = 0)
		sharpness_maps = sharpness_maps/total_sharpness_sum

		return sharpness_maps

	def get_smart_aif_and_depth(self) :
		num_images = len(self.depth_values)
		s_length = int(np.shape(self.image)[0]/self.u_length)
		t_length = int(np.shape(self.image)[1]/self.v_length)

		aif_image = np.zeros((s_length, t_length, 3))
		depth_map = np.zeros((s_length,t_length))

		total_sharpness_sum = np.sum(self.final_map, axis = 0)
		total_sharpness_sum_stacked = np.dstack((total_sharpness_sum, total_sharpness_sum, total_sharpness_sum))

		for i in range(num_images) :
			focal_slice = self.focal_stack[i,:,:,:]
			sharpness_map = self.final_map[i,:,:]
			sharpness_map_stacked = np.dstack((sharpness_map, sharpness_map, sharpness_map))
			aif_image += sharpness_map_stacked * focal_slice / total_sharpness_sum_stacked
			depth_map += -self.depth_values[i] * np.exp(-sharpness_map)

		plt.imshow(aif_image/255)
		plt.show()
		plt.imshow(depth_map, cmap = 'gray')
		plt.show()
		return aif_image, depth_map


	def get_self_stack(self, num_captures = 8) :
		folder_path = self.self_focal_path
		focal_stack = []
		for i in range(num_captures) :
			image_path = folder_path + str(i+1) + '.jpg'
			image = io.imread(image_path)
			focal_stack.append(image)
		return focal_stack

	def get_self_aif_and_depth(self) :
		num_images = len(self.self_focal_stack)
		s_length = int(np.shape(self.self_focal_stack[0])[0])
		t_length = int(np.shape(self.self_focal_stack[0])[1])
		sharpness_maps = np.zeros((num_images, s_length, t_length))

		for i in range(num_images) :
			print (i)
			focal_slice = self.self_focal_stack[i]
			gamma_focal_slice = gamma_correct(focal_slice)
			focal_slice_xyz = rgb2xyz(gamma_focal_slice)
			focal_slice_luminance = focal_slice_xyz[:,:,1]
			low_freq_image = skimage.filters.gaussian(focal_slice_luminance, sigma=0.5)
			high_freq_image = focal_slice_luminance - low_freq_image
			high_freq_image = high_freq_image ** 2
			sharpness_map = skimage.filters.gaussian(high_freq_image, sigma=8)
			sharpness_maps[i,:,:] = sharpness_map

		total_sharpness_sum = np.sum(sharpness_maps, axis = 0)
		total_sharpness_sum_stacked = np.dstack((total_sharpness_sum, total_sharpness_sum, total_sharpness_sum))

		aif_image = np.zeros((s_length, t_length, 3))
		depth_map = np.zeros((s_length,t_length))
		for i in range(num_images) :
			print (i)
			focal_slice = self.self_focal_stack[i]
			sharpness_map = sharpness_maps[i,:,:]
			sharpness_map_stacked = np.dstack((sharpness_map, sharpness_map, sharpness_map))
			aif_image += sharpness_map_stacked * focal_slice / total_sharpness_sum_stacked
			depth_map += (1000-100*i) * sharpness_map / total_sharpness_sum

		depth_map = (depth_map - np.min(depth_map))/(np.max(depth_map) - np.min(depth_map))
		plt.imshow(aif_image/255)
		plt.show()
		plt.imshow(depth_map, cmap = 'gray')
		plt.show()
		return aif_image, depth_map




if __name__ == "__main__": 
	HW4('./data/chessboard_lightfield.png', './data/unstructured_lf/1.jpg','./data/unstructured_lf/', './data/self_focal_stack/')

