import numpy as np
import skimage
from skimage import io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os	
from os import walk
import imageio
import cp_hw2 as hw2
import OpenEXR
import Imath
import copy
import sys
import argparse
import pdb
st = pdb.set_trace

def get_exp_time(val) :
	exp_times = (1/2048)*np.exp2(val)
	return exp_times

def get_image_pixels(image_path, freq = 200) :
	im = io.imread(image_path)
	return im[::freq, ::freq]


def weights_uniform(z, linearization = True) :
	if isinstance(z, np.ndarray) :
		weights = np.ones(z.shape)
		if linearization == False :
			lower_bound = np.where(z < 0.05)
			upper_bound = np.where(z > 0.95)
		else :
			lower_bound = np.where(z < 0)
			upper_bound = np.where(z > 255)

		weights[lower_bound] = 0
		weights[upper_bound] = 0
		return weights
	else :
		if linearization == False :
			if z < 0.05 :
				return 0
			if z > 0.95 :
				return 1
		else :
			if z < 0 :
				return 0
			elif z > 255 :
				return 0
			else :
				return 1

def weights_tent(z, linearization = True) :
	minusz = 1-z
	if isinstance(z, np.ndarray) :
		weights = np.ones(z.shape)
		if linearization == False :
			minusz = 255-z
			weights = np.minimum(z, minusz)
			lower_bound = np.where(z < 0.05)
			upper_bound = np.where(z > 0.95)
		else :
			weights = np.minimum(z, minusz)/255
			lower_bound = np.where(z < 0)
			upper_bound = np.where(z > 255)

		weights[lower_bound] = 0
		weights[upper_bound] = 0
		return weights
	else :
		if linearization == False :
			if z < 0.05 :
				return 0
			elif z > 0.95 :
				return 0
			else :
				return min(z, 1-z)
		else :
			if z < 1 :
				return 0;
			elif z > 254 :
				return 0;
			else :
				return min(z, 1-z)/255

def weights_gaussian(z, linearization = True) :
	if isinstance(z, np.ndarray) :
		weights = np.ones(z.shape)
		if linearization == False :
			weights = np.exp(8*(z-0.5)**2)
			lower_bound = np.where(z < 0.05)
			upper_bound = np.where(z > 0.95)
		else :
			weights = np.exp(8*(z/255-0.5)**2)
			lower_bound = np.where(z < 0)
			upper_bound = np.where(z > 255)

		weights[lower_bound] = 0
		weights[upper_bound] = 0
		return weights
	else :
		if linearization == False :
			if z < 0.05 :
				return 0
			elif z > 0.95 :
				return 0
			else :
				return np.exp(8*(z-0.5)**2)
		else :
			if z < 0 :
				return 0;
			elif z > 255 :
				return 0;
			else :
				return np.exp(8*(z/255-0.5)**2)

def weights_photon(z, exp_time, linearization = True) :
	weights = np.ones(z.shape)
	max_exp_time = get_exp_time(15)
	normalized_exp_time = exp_time/max_exp_time
	#weights = normalized_exp_time * weights
	weights = exp_time * weights
	if linearization == False :
		lower_bound = np.where(z < 0.05)
		upper_bound = np.where(z > 0.95)
	else :		
		lower_bound = np.where(z < 0)
		upper_bound = np.where(z > 255)

	weights[lower_bound] = 0
	weights[upper_bound] = 0
	return weights

def weights_optimal(z, exp_time, mean_val, variance_val) :
	weights = np.zeros(z.shape)
	weights = (exp_time * exp_time)/(mean_val * z + variance_val)
	lower_bound = np.where(z < 0.05)
	upper_bound = np.where(z > 0.95)
	weights[lower_bound] = 0
	weights[upper_bound] = 0
	return weights

def get_weights(z, weight_type, linearization = True) :
	if weight_type == 'uniform' :
		return weights_uniform(z,linearization)
	if weight_type == 'tent' :
		return weights_tent(z,linearization)
	else :
		return weights_gaussian(z,linearization)

def get_matrix_block(im_path, exp_val, weight_type = "uniform") :
	image_pixels = get_image_pixels(im_path)
	exp_time = get_exp_time(exp_val)
	image_pixels[np.where(image_pixels > 255)] = 255
	image_pixels[np.where(image_pixels < 0)] = 0
	image_pixels = np.reshape(image_pixels, (-1, 1))
	if weight_type != 'photon' :
		image_pixels_weights = get_weights(image_pixels, weight_type)
		#print (image_pixels_weights)
	else :
		image_pixels_weights = weights_photon(image_pixels, exp_time)

	#st()
	#print (np.sum(image_pixels_weights))
	A_matrix = np.zeros((0,0))
	b_vector = []
	for index in range(image_pixels.shape[0]) :
		color_row = np.zeros((1,256))
		color_row[0,image_pixels[index,0]] = image_pixels_weights[index,0]
		radiance_row = np.zeros((1,image_pixels.shape[0]))
		radiance_row[0, index] = image_pixels_weights[index,0]
		b_entry = image_pixels_weights[index,0] * np.log(exp_time)
		pixel_row = np.hstack((color_row, radiance_row))
		if A_matrix.shape[0] == 0 :
			A_matrix = pixel_row
		else :
			A_matrix = np.vstack((A_matrix, pixel_row))
		b_vector.append(b_entry)
	b_val= np.asarray(b_vector)
	b = np.reshape(b_val, (-1,1))
	#print (np.sum(A_matrix[256:]))
	return A_matrix,b

def get_linear_func(image_folder, weight_type = "uniform") :
	A = np.zeros((0,0))
	b = np.zeros((0,0))
	for i in range(0,16) :
		path = image_folder + 'exposure' + str(i+1) + '.jpg'
		A_mat, b_vec = get_matrix_block(path, i, weight_type)
		if A.shape[0] == 0 :
			A = A_mat
			b = b_vec
		else :
			A = np.vstack((A, A_mat))
			b = np.vstack((b, b_vec))

	laplacian_submatrix = np.zeros((256, A.shape[1]))
	b_submatrix = np.zeros((256,1))
	lambda_val = 10
	for i in range(256) :
		if weight_type != 'photon' :
			if i == 0 :
				laplacian_submatrix[i,0] = -2*get_weights(i, weight_type)*lambda_val
				laplacian_submatrix[i,1] = 1*get_weights(i, weight_type)*lambda_val
			else :
				laplacian_submatrix[i,i] = -2*get_weights(i, weight_type)*lambda_val
				laplacian_submatrix[i,i-1] = 1*get_weights(i, weight_type)*lambda_val
				laplacian_submatrix[i,i+1] = 1*get_weights(i, weight_type)*lambda_val
			if i == 255 :
				laplacian_submatrix[i,i+1] = 0
		else :
			if i == 0 :
				laplacian_submatrix[i,0] = -2*lambda_val
				laplacian_submatrix[i,1] = 1*lambda_val
			else :
				laplacian_submatrix[i,i] = -2*lambda_val
				laplacian_submatrix[i,i-1] = 1*lambda_val
				laplacian_submatrix[i,i+1] = 1*lambda_val
			if i == 255 :
				laplacian_submatrix[i,i+1] = 0


	#Add index = 1 in A and b
	extra_row = np.ones((1,A.shape[1]))
	b = np.vstack((b, np.array([[1]])))
	A = np.vstack((A, extra_row))
	A = np.vstack((A, laplacian_submatrix))
	b = np.vstack((b, b_submatrix))
	sol = np.linalg.lstsq(A,b)
	x = sol[0]
	x = x[0:256]
	return x

def merge_into_hdr_jpeg(in_folder, func, weight_type = "uniform", islog = False) :
	for i in range(0,16) :
		print (i)
		path = in_folder + 'exposure' + str(i+1) + '.jpg'
		im = io.imread(path)
		ldr_image = im/255
		linear_im = np.exp(func[im[:]])
		linear_im = linear_im[:,:,:,0]/255
		weighted_ldr_image = get_weights(ldr_image, weight_type, False)
		exp_time = get_exp_time(i)
		if i == 0 :
			if islog :
				numerator = weighted_ldr_image * (np.log(linear_im)-np.log(exp_time))
			else :
				numerator = weighted_ldr_image * linear_im/exp_time
			denominator = weighted_ldr_image
		else :
			if islog :
				numerator += weighted_ldr_image * (np.log(linear_im)-np.log(exp_time))
			else :
				numerator += weighted_ldr_image * linear_im/exp_time
			denominator += weighted_ldr_image

	zero_weights = np.where(denominator == 0)
	denominator[zero_weights] = 1
	numerator[zero_weights] = linear_im[zero_weights]/exp_time
	if islog == True :
		numerator[zero_weights] = linear_im[zero_weights] - np.log(exp_time)

	hdr_image = numerator/(denominator)
	if islog :
		hdr_image = np.exp(hdr_image)

	imname = 'hdr_jpeg_' + weight_type + str(islog) + '.exr'
	hw2.writeEXR(imname, hdr_image)
	#plt.imshow(hdr_image)
	#plt.show()
	return hdr_image, imname


def merge_into_hdr_tiff(in_folder, weight_type = "uniform", islog = False) :
	for i in range(0,7) :
		print (i)
		path = in_folder + 'exposure' + str(i+1) + '.tiff'
		im = io.imread(path)
		ldr_image = im
		linear_im = ldr_image
		weighted_ldr_image = get_weights(ldr_image, weight_type, False)
		exp_time = get_exp_time(i)
		if i == 0 :
			if islog :
				numerator = weighted_ldr_image * (np.log(linear_im + 0.0001)-np.log(exp_time))
			else :
				numerator = weighted_ldr_image * linear_im/exp_time
			denominator = weighted_ldr_image
		else :
			if islog :
				numerator += weighted_ldr_image * (np.log(linear_im + 0.0001)-np.log(exp_time))
			else :
				numerator += weighted_ldr_image * linear_im/exp_time
			denominator += weighted_ldr_image

	zero_weights = np.where(denominator == 0)
	denominator[zero_weights] = 1
	numerator[zero_weights] = linear_im[zero_weights]/exp_time
	if islog == True :
		numerator[zero_weights] = np.log(linear_im[zero_weights] + 0.0001) - np.log(exp_time)

	hdr_image = numerator/(denominator)
	if islog :
		hdr_image = np.exp(hdr_image)

	imname = 'hdr_tiff_' + weight_type + str(islog) + '.exr'
	hw2.writeEXR(imname, hdr_image)
	#plt.imshow(hdr_image)
	#plt.show()
	return hdr_image, imname


def merge_into_hdr_optimal(in_folder, dark_image, mean_val = 2.069, variance_val = 62.01, dark_exp = 1/8) :
	for i in range(0,16) :
		print (i)
		exp_time = get_exp_time(i)
		path_linear = in_folder + 'exposure' + str(i+1) + '.tiff'
		im = io.imread(path_linear)
		dark_im = dark_image[8:-8, 8:-8, :]
		im = im[8:-8, 8:-8, :]
		im = im - (exp_time/dark_exp)*dark_im
		ldr_image = im/65536
		linear_im = io.imread(path_linear)
		linear_im = linear_im[8:-8, 8:-8, :]/65536
		weighted_ldr_image = weights_optimal(ldr_image, exp_time, mean_val, variance_val)
		if i == 0 :
			numerator = weighted_ldr_image * linear_im/exp_time
			denominator = weighted_ldr_image
		else :
			numerator += weighted_ldr_image * linear_im/exp_time
			denominator += weighted_ldr_image

	zero_weights = np.where(denominator == 0)
	denominator[zero_weights] = 1
	numerator[zero_weights] = linear_im[zero_weights]/exp_time

	hdr_image = numerator/(denominator)

	imname = 'hdr_tiff_self_captured.exr'
	hw2.writeEXR(imname, hdr_image)
	return hdr_image,imname

def get_pts(impath, num_pts) :
	x = impath.split('.')
	im = io.imread(impath)
	if x[-1] == 'tiff' :
		im = im/65536
	f = plt.imshow(im)
	pts = plt.ginput(n = num_pts, timeout = num_pts*5)
	plt.close()
	return pts

def get_patches(pts, hdr_path, isHDR = False) :
	hdr_image = OpenEXR.InputFile(hdr_path)
	dw = hdr_image.header()['displayWindow']
	size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
	data = [np.frombuffer(c, np.float32).reshape(size) for c in hdr_image.channels('RGB', Imath.PixelType(Imath.PixelType.FLOAT))]
	img = np.dstack(data)
	linear_image = hw2.lRGB2XYZ(img)
	linear_image = linear_image[:,:,1]
	patches = []
	patch_size = 10
	for pt in pts :
		ptx = int(pt[0])
		pty = int(pt[1])
		if isHDR :
			im_patch = linear_image[pty-patch_size:pty+patch_size, ptx-patch_size:ptx+patch_size]
		else :
			im_patch = img[pty-patch_size:pty+patch_size, ptx-patch_size:ptx+patch_size, :]
		patches.append(im_patch)
	return patches

def evaluate_hdr(patches) :
	avg_illum = []
	for patch in patches :
		curr_illum = np.mean(patch)
		avg_illum.append(np.log(curr_illum))
	#plt.plot(avg_illum)
	#plt.show()
	x = [0,1,2,3,4,5]
	A = np.vstack([x, np.ones(len(x))]).T
	_,res,_,_ = np.linalg.lstsq(A, avg_illum)
	return res[0]

def get_wb_affine_transform(patches, gt_patch_values) :
	A = np.zeros((0,4))
	b_mat = np.zeros((0,3))
	for i in range(4) :
		for j in range(6) :
			curr_patch = patches[i,j,:,:,:]
			curr_mean = np.mean(curr_patch, axis = (0,1))
			curr_mean = np.hstack((curr_mean, np.array([1])))
			A = np.vstack((A,curr_mean))
			b_red = gt_patch_values[0][i,j]
			b_green = gt_patch_values[1][i,j]
			b_blue = gt_patch_values[2][i,j]
			b_vec = np.array([b_red, b_green, b_blue])
			b_vec = b_vec.reshape(1,3)
			b_mat = np.vstack((b_mat, b_vec))

	affine_trans = np.linalg.lstsq(A,b_mat)
	ans = np.ones((400,600,3))
	ans_1 = np.ones((400,600,3))
	for i in range(4) :
		for j in range(6) :
			curr_patch = patches[i,j,:,:,:]
			curr_mean = np.mean(curr_patch, axis = (0,1))
			curr_mean = np.hstack((curr_mean, np.array([1])))
			curr_mean = np.matmul(curr_mean,affine_trans[0])
			ans[i*100:(i+1)*100,j*100:(j+1)*100,:] = curr_mean

			b_red = gt_patch_values[0][i,j]
			b_green = gt_patch_values[1][i,j]
			b_blue = gt_patch_values[2][i,j]
			b_vec = np.array([b_red, b_green, b_blue])
			b_vec = b_vec.reshape(1,3)
			ans_1[i*100:(i+1)*100,j*100:(j+1)*100,:] = b_vec

	# plt.imshow(ans)
	# plt.show()
	# plt.imshow(ans_1)
	# plt.show()
	return affine_trans[0]

def apply_white_balance(hdr_path, affine_transform, wb_points) :
	save_path = hdr_path.split('.ex')
	save_name = save_path[0] + '_wb.exr'
	hdr_image = OpenEXR.InputFile(hdr_path)
	dw = hdr_image.header()['displayWindow']
	size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
	data = [np.frombuffer(c, np.float32).reshape(size) for c in hdr_image.channels('RGB', Imath.PixelType(Imath.PixelType.FLOAT))]
	img = np.dstack(data)
	rgb_reshaped = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
	extra_row = np.ones((rgb_reshaped.shape[0],1))
	reshape_img_homo = np.hstack((rgb_reshaped,extra_row))
	result = np.dot(reshape_img_homo,affine_transform)
	result = result.reshape(img.shape)
	# Results better without applying AWB
	'''
	wb_points_array = np.array(wb_points)
	wb_points_reshape = wb_points_array.reshape(4,6,2,1)
	white_points = wb_points_reshape[3,0,:,:]
	white_points = np.reshape(white_points, (-1,1))
	white_patch = result[int(white_points[1])-10:int(white_points[1])+10, int(white_points[0])-10:int(white_points[0])+10, :]
	white_patch_mean = np.mean(white_patch, axis = (0,1))
	red_gain = white_patch_mean[1]/white_patch_mean[0]
	blue_gain = white_patch_mean[2]/white_patch_mean[0]
	result[:,:,0] = result[:,:,0]*red_gain
	result[:,:,2] = result[:,:,2]*blue_gain
	'''
	hw2.writeEXR(save_name, result)
	return result, save_name

def gamma_and_brightness(im) :
	im_rgb = copy.copy(im)
	im1 = im_rgb[:,:,0]
	im2 = im_rgb[:,:,1]
	im3 = im_rgb[:,:,2]
	imred_lower_bound = im_rgb[:,:,0] < 0.0031308
	imred_upper_bound = im_rgb[:,:,0] >= 0.0031308

	imgreen_lower_bound = im_rgb[:,:,1] < 0.0031308
	imgreen_upper_bound = im_rgb[:,:,1] >= 0.0031308

	imblue_lower_bound = im_rgb[:,:,2] < 0.0031308
	imblue_upper_bound = im_rgb[:,:,2] >= 0.0031308

	im1[imred_lower_bound] = 12.92*im1[imred_lower_bound]
	im1[imred_upper_bound] = (1+0.055)*(im1[imred_upper_bound])**(1/2.4) - 0.055

	im2[imgreen_lower_bound] = 12.92*im2[imgreen_lower_bound]
	im2[imgreen_upper_bound] = (1+0.055)*(im2[imgreen_upper_bound])**(1/2.4) - 0.055

	im3[imblue_lower_bound] = 12.92*im3[imblue_lower_bound]
	im3[imblue_upper_bound] = (1+0.055)*(im3[imblue_upper_bound])**(1/2.4) - 0.055

	im_gamma = np.dstack((im1, im2, im3))
	return im_gamma

def tone_mapping(hdr_path, K = 0.15, B = 0.95, color_space = 'RGB') :
	save_path = hdr_path.split('.ex')
	save_name = save_path[0] + str(K) + str(B) + color_space + '.png'
	hdr_image = OpenEXR.InputFile(hdr_path)
	dw = hdr_image.header()['displayWindow']
	size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
	data = [np.frombuffer(c, np.float32).reshape(size) for c in hdr_image.channels('BGR', Imath.PixelType(Imath.PixelType.FLOAT))]
	img = np.dstack(data)
	img = np.clip(img, 0, None)
	if color_space == 'RGB' :
		img_tonemapped = np.zeros(np.shape(img))
		im_hdr = np.exp(np.mean(np.log(img + 0.0001)))
		for channel in range(3) :
			im_tilda_hdr = (K/im_hdr)*img[:,:,channel]
			im_tilda_white = B*np.max(im_tilda_hdr)
			im_tonemapped_channel = (im_tilda_hdr*(1 + im_tilda_hdr/(im_tilda_white*im_tilda_white)))/(1+im_tilda_hdr)
			img_tonemapped[:,:,channel] = im_tonemapped_channel
		im_gamma_correct = gamma_and_brightness(img_tonemapped)
		plt.imshow(im_gamma_correct)
		plt.savefig(save_name)
		return im_gamma_correct
	else :
		XYZ_im = hw2.lRGB2XYZ(img)
		x_channel = XYZ_im[:,:,0]/(XYZ_im[:,:,0] + XYZ_im[:,:,1] + XYZ_im[:,:,2])
		y_channel = XYZ_im[:,:,1]/(XYZ_im[:,:,0] + XYZ_im[:,:,1] + XYZ_im[:,:,2])
		Y_image = XYZ_im[:,:,1]
		im_hdr = np.exp(np.mean(np.log(Y_image + 0.0001)))
		im_tilda_hdr = (K/im_hdr)*Y_image
		im_tilda_white = B*np.max(im_tilda_hdr)
		im_tonemapped = (im_tilda_hdr*(1 + im_tilda_hdr/(im_tilda_white*im_tilda_white)))/(1+im_tilda_hdr)
		Y_channel = im_tonemapped
		X_final, Y_final, Z_final = hw2.xyY_to_XYZ(x_channel, y_channel, Y_channel)
		XYZ_final = np.dstack((X_final, Y_final, Z_final))
		im_rgb = hw2.XYZ2lRGB(XYZ_final)
		im_gamma_correct = gamma_and_brightness(im_rgb)
		plt.imshow(im_gamma_correct)
		plt.savefig(save_name)
		return im_gamma_correct


def get_dark_frame(in_folder) :
	rgb_weights = [0.2989, 0.5870, 0.1140]
	for i in range(50) :
		path_linear = in_folder + 'exposure' + str(i) + '.tiff'
		im_current = io.imread(path_linear)
		#im_current = np.dot(im_current[...,:3], rgb_weights)
		if i == 0 :
			dark_im = im_current
		else :
			dark_im += im_current
	dark_im = dark_im / 50
	return dark_im

def ramp_image_pocessing(in_folder, dark_im) :
	rgb_weights = [0.2989, 0.5870, 0.1140]

	path_linear = in_folder + 'exposure0.tiff'
	im_current = io.imread(path_linear)
	im_current = np.dot(im_current[...,:3], rgb_weights)
	im_current = im_current - dark_im
	im_current = im_current/65536
	im_current = im_current[0:4000, 0:4000]
	f = plt.imshow(im_current)
	pts = plt.ginput(n = 5, timeout = 5*5)
	plt.close()
	hist_array = np.zeros((50,5))
	for i in range(50) :
		path_linear = in_folder + 'exposure' + str(i) + '.tiff'
		im_current = io.imread(path_linear)
		im_current = np.dot(im_current[...,:3], rgb_weights)
		im_current = im_current - dark_im
		im_current = im_current[0:4000, 0:4000]
		for j,pt in enumerate(pts) :
			hist_array[i,j] =  im_current[int(pt[1]), int(pt[0])]

	for i in range(5) :
		current_pt = hist_array[:,i]
		plt.hist(current_pt, bins=100, alpha=0.5, label="red")
		plt.show()
		plt.close()
	plt.close()

	for i in range(50) : #50
		print (i)
		path_linear = in_folder + 'exposure' + str(i) + '.tiff'
		im_current = io.imread(path_linear)
		#im_current = np.dot(im_current[...,:3], rgb_weights)
		im_current = im_current - dark_im
		im_current = im_current[0:5000, 0:4000]
		if i == 0 :
			im_summation = im_current
		else :
			im_summation += im_current
	im_mean = np.round(im_summation/50) #50

	for i in range(50) : #50
		print (i)
		path_linear = in_folder + 'exposure' + str(i) + '.tiff'
		im_current = io.imread(path_linear)
		#im_current = np.dot(im_current[...,:3], rgb_weights)
		im_current = im_current - dark_im
		im_current = im_current[0:5000, 0:4000]
		if i == 0 :
			im_variance = (im_current - im_mean)**2
		else :
			im_variance += (im_current - im_mean)**2
	im_variance = im_variance/49 #49

	return im_mean, im_variance


def save_points_into_file(filename) :
	pts_gray = get_pts(filename, 6)
	pts_wb = get_pts(filename, 24)
	np.savez('pts_gray.npz', jpeg_pts = pts_gray)
	np.savez('pts_wb.npz', jpeg_pts = pts_wb)


def get_hdr_image(foldername, weight_type, isLinear = False, isLogMerging = False) :
	if isLinear == False :
		linear_weights = get_linear_func(foldername, weight_type)
		hdr_image, im_name = merge_into_hdr_jpeg(foldername, linear_weights, weight_type, isLogMerging)
	else :
		hdr_image, im_name = merge_into_hdr_tiff(foldername, weight_type, isLogMerging)
	return hdr_image, im_name

def evaluate_hdr_image(hdr_image_name, hdr_points_file_name = 'pts_gray.npz') :
	hdr_points = np.load(hdr_points_file_name)
	pts_gray = hdr_points['jpeg_pts']
	hdr_patches = get_patches(pts_gray, hdr_image_name, True)
	error = evaluate_hdr(hdr_patches)
	return error

def get_wb_image(hdr_image_name, wb_points_file_name = 'pts_wb.npz') :
	wb_points = np.load(wb_points_file_name)
	pts_wb = wb_points['jpeg_pts']
	wb_patches = get_patches(pts_wb, hdr_image_name, False)
	wb_patches = np.array(wb_patches)
	patch_size = 10
	gt_patch_values = hw2.read_colorchecker_gm()
	wb_patches = wb_patches.reshape(4,6,patch_size*2,patch_size*2,3)
	affine_transform = get_wb_affine_transform(wb_patches, gt_patch_values)
	wb_image, im_path_name = apply_white_balance(hdr_image_name, affine_transform, pts_wb)
	return wb_image, im_path_name


def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__" :
	#parser = argparse.ArgumentParser(description='Process some integers.')
	# parser.add_argument('save_points', type=str2bool, help='Check if points need to be saved')
	# parser.add_argument('save_points_filename', type=str, help='Path to image to select points for evaluation and WB')
	#parser.add_argument('hdr_folder_path', type = str, help='Path to HDR image folder path')
	# parser.add_argument('evaluate_hdr', type = str2bool, help = 'Is evaluate HDR on')
	# parser.add_argument('K', type = float, help = 'Key value')
	# parser.add_argument('B', type = float, help = 'Burn value')
	#parser.add_argument('weight', type = str, help = 'Weight to select for merging')
	#parser.add_argument('merging', type = str2bool, help = 'Select merging type log or normal')
	#parser.add_argument('isLinear', type = str2bool, help = 'Select if linear or non-linear merging')
	# parser.add_argument('color_space', type = str, help = 'Select color space for tonemapping')
	# parser.add_argument('noise_calibration', type = str2bool, help = 'Perform noise calibration and merge using noise optimal weights')
	# parser.add_argument('dark_frame_folder', type = str, help = 'Dark frame path')
	# parser.add_argument('ramp_image_folder', type = str, help = 'Ramp images path')
	# parser.add_argument('hdr_folder_path_optimal', type = str, help = 'Exopsure stackpath')

	#args = parser.parse_args()
	# if args.save_points == True :
	# 	save_points_into_file(args.save_points_filename)

	#hdr_image, im_name = get_hdr_image('../data/project/', 'uniform', True, False)
	tonemapped_im = tone_mapping('hdr_image.exr')
	st()


	# if args.noise_calibration == True :
	# 	dark_image = get_dark_frame(args.dark_frame_folder)
	# 	im_mean, im_variance = ramp_image_pocessing(args.ramp_image_folder, dark_image)
	# 	im_mean_reshape = np.reshape(im_mean,(-1,1))
	# 	im_var_reshape = np.reshape(im_variance,(-1,1))
	# 	xcoord = []
	# 	ycoord = []
	# 	count = 0
	# 	for value in im_unique_values :
	# 		if count % 1000  == 0:
	# 			print (count)
	# 		count += 1
	# 		if value > 0 :
	# 			xcoord.append(value)
	# 			y_current = np.mean(im_var_reshape[np.where(im_mean_reshape == value)])
	# 			ycoord.append(y_current)

	# 	plt.plot(xcoord, ycoord)
	# 	plt.show()
	# 	A = np.vstack([xcoord, np.ones(len(xcoord))]).T
	# 	b = ycoord
	# 	res,_,_,_ = np.linalg.lstsq(A, b)
	# 	hdr_im, imname = merge_into_hdr_optimal(args.hdr_folder_path_optimal, dark_image, mean_val = res[0], variance_val =res[1], dark_exp = 1/8)
	# 	tone_mapped_image = tone_mapping(imname, color_space = 'RGB')
