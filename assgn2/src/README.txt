The code consists of two files. 
The first file is the assign2.py.
The second file is the one that has been already provided  - cp_hw2.py

The code takes in the following arguments :
	save_points of type bool
		If this is true, an image pops up and we select the patches corresponding to the HDR evaluation and color checker patches for WB. These are saved in npz files to be used hereafter
	save_points_filename of type str - To select the image from which the points mentioned above will be selected

	hdr_folder_path of type str - Gives the path to folder where the exposure stack is stored

	evaluate_hdr of type bool
		If this is True, the HDR is evaluated using logarithm of illumination and the error is printed

	K of type float - Key value

	B of type float - Burn value

	weight of type str to select one of 'uniform', 'tent', 'gaussian', 'photon' weighting schemes

	merging of type bool
		If this is True, the merging occurs in Log domain

	isLinear of type bool
		If this is True, we assume we have non-linear images. The code then runs the code to find the g according to the weights selected to linearize the images and then merges the linearized images according to the merging method selected

	color_space of type str to select 'RGB' or 'XYZ' domain for tonemapping 

	noise_calibration of type bool
		If this is set to True, we run the noise calibration code to find the mean and the variance which is the gain and the additive noise value, and calculates the noise optimal weights, merges the image assuming linear images, performs tonemapping in RGB domain and outputs the final tonemapped image.

	dark_frame_folder of type str - Dark frame path
	ramp_image_folderof type str - Ramp images path
	hdr_folder_path_optimal of type = str - Exopsure stack path for optimal merging


Example command : python assign2.py False ../src ../data/door_stack/tiff_images/ False 0.15 0.90 tent True True RGB True dark_folder_path ramp_folder_path exposure_stack_folder_path
