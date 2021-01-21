import os
import cv2
import math
import glob
import imageio
import numpy as np
import tensorflow as tf

#----------Network training input preprocessing----------
def load_original_images(data_dir,label_dir,cut_off,train):
	"""
	Load original image
	"""
	images_list = os.listdir(data_dir)
	images_list.sort(key=lambda x:int(x[:-4]))

	label_list = os.listdir(label_dir)
	label_list.sort(key=lambda x:int(x[:-4]))

	images = []
	labels = []

	if train:
		for i in range(cut_off):
			img = cv2.imread(data_dir + images_list[i],1)
			images.append(img)

			lab = imageio.imread(label_dir + label_list[i])
			labels.append(lab)

	else:
		for i in range(cut_off,int(len(images_list))):
			img = cv2.imread(data_dir + images_list[i],1)
			images.append(img)

			lab = imageio.imread(label_dir + label_list[i])
			labels.append(lab)

	return images, labels

def get_mark_point(pic_num,txt_dir):
	"""
	Load target coordinates
	pic_num:Picture number
	txt_dir:Coordinate document address
	"""

	with open(txt_dir, "r") as f:		
		linenum = 2 

		mark_point = []

		for line in f.readlines():

			#Skip the first two lines
			if linenum != 0:
				linenum -= 1
				continue
			
			#Jump to the target coordinates corresponding to the pic_num picture
			if (pic_num - 1) != 0 : 
				if line[0] == "x":
					pic_num -= 1
				continue	
			
			#Separate strings by fixed delimiter
			pointline = line.strip().split("," + " ") 
			if pointline[0] == "":
				break

			x_1 = int(float(pointline[0][:6]))
			y_1 = int(float(pointline[1][:6]))
			x_2 = int(float(pointline[2][:6]))
			y_2 = int(float(pointline[3][:6]))

			#Vehicle coordinate center point
			mark_point_x = (x_1 + x_2) / 2
			mark_point_y = (y_1 + y_2) / 2

			mark_point.append([mark_point_x,mark_point_y])

	#print("get_mark_point done")
	return mark_point

def sample_data(all_images, all_label, cut_off, batch_size, image_size, txt_dir):
	"""
	Sampling and preprocessing the original image 
	Hybrid sampling method: target-oriented sampling and completely random sampling = 2:1
	all_images: Store a list of all pictures
	all_label: Store a list of all labels
	cut_off: Training set and test set separator
	batch_size: Batch size
	image_size: Sampling size
	txt_dir: Target coordinate file address 
	"""

	img_num = np.arange(0, cut_off, 1)
	
	images_data = []
	label_data = []
	
	#resize:Final uniform image size
	resize = image_size 

	for n in img_num:

		#if n % 2 == 0:
		#	continue

		img = all_images[n]
		lab = all_label[n]

		#Get the coordinate points of all vehicles in the picture
		mark_point = get_mark_point(n + 1,txt_dir)

		#Disrupt the order of coordinate points
		nn = np.arange(int(len(mark_point)))
		np.random.shuffle(nn)

		for i in nn:
			
			image_size = int(np.random.uniform(0.3 * resize, 2 * resize, 1))

			#Random shift
			x = np.random.randint(-128,128,1)
			mark_point[i][0] += x

			y = np.random.randint(-128,128,1)
			mark_point[i][1] += y

			#Random lighting transformation coefficient
			a = np.random.uniform(0.5,1.5,1)
			b = np.random.uniform(-50,50,1)

			#Random rotation coefficient
			c = np.random.uniform(0,360,1)

			#-----Take the vehicle position as the center plus the offset in two directions for sampling-----

			#Determine whether the sampling range in the four directions overflows, if overflow, skip to sampling the next coordinate
			if (mark_point[i][0] - (image_size / 2) < 0) or (mark_point[i][1] - (image_size / 2) < 0) or (mark_point[i][0] + (image_size / 2) > img.shape[1]) or (mark_point[i][1] + (image_size / 2) > img.shape[0]):	
				continue
			else:
			#----- +images -----
				img1 = img[int(mark_point[i][1] - (image_size / 2)) : int(mark_point[i][1] + (image_size / 2)),
							int(mark_point[i][0] - (image_size / 2)) : int(mark_point[i][0] + (image_size / 2))]
				#Random lighting transformation
				img2 = cv2.convertScaleAbs(src = img1,alpha=a,beta=b)

				#Rotation matrix
				rows, cols, channels = img2.shape
				rotate = cv2.getRotationMatrix2D((rows * 0.5, cols * 0.5), c, 1)
				#Random rotation
				img2 = cv2.warpAffine(img2, rotate, (cols, rows))

				#Normalize to uniform size
				img2 = cv2.resize(img2,(resize,resize))
				images_data.append(img2)

			#----- -labels -----
				lab1 = lab[int(mark_point[i][1] - (image_size / 2)) : int(mark_point[i][1] + (image_size / 2)),
							int(mark_point[i][0] - (image_size / 2)) : int(mark_point[i][0] + (image_size / 2))]


				lab1 = cv2.warpAffine(lab1, rotate, (cols, rows))

				lab1 = cv2.resize(lab1,(resize,resize))
				label_data.append(lab1)

		#-----Completely random sampling-----

		for i in range(150): 

			image_size = int(np.random.uniform(0.3 * resize, 2 * resize, 1))

			x_ = np.random.randint(image_size / 2,img.shape[0] - image_size / 2,1)
			y_ = np.random.randint(image_size / 2,img.shape[1] - image_size / 2,1)

			a = np.random.uniform(0.5,1.5,1)
			b = np.random.uniform(-50,50,1)
			c = np.random.uniform(0,360,1)

		#----- -images -----

			img1 = img[int(x_ - (image_size / 2)) : int(x_ + (image_size / 2)),
							int(y_ - (image_size / 2)) : int(y_ + (image_size / 2))]
				
			img2 = cv2.convertScaleAbs(src = img1,alpha=a,beta=b)

			rows, cols, channels = img2.shape
			rotate = cv2.getRotationMatrix2D((rows * 0.5, cols * 0.5), c, 1)

			img2 = cv2.warpAffine(img2, rotate, (cols, rows))

			img2 = cv2.resize(img2,(resize,resize))
			images_data.append(img2)

		#----- -labels -----
		
			lab1 = lab[int(x_ - (image_size / 2)) : int(x_ + (image_size / 2)),
						int(y_ - (image_size / 2)) : int(y_ + (image_size / 2))]

			lab1 = cv2.warpAffine(lab1, rotate, (cols, rows))

			lab1 = cv2.resize(lab1,(resize,resize))
			label_data.append(lab1)

	print("load img done")

	images_data_s = []
	label_data_s = []

	#Shuffle the sampled picture
	index = np.arange(int(len(label_data)))
	np.random.shuffle(index)
	
	for i in index:
		images_data_s.append(images_data[i])
		label_data_s.append(label_data[i])

	return images_data_s, label_data_s

#----------Training process logging----------
def write_log(callback, name, loss, batch_no):
	"""
	Write training summary to TensorBoard
	"""
	summary = tf.Summary()
	summary_value = summary.value.add()
	summary_value.simple_value = loss
	summary_value.tag = name
	callback.writer.add_summary(summary, batch_no)
	callback.writer.flush()

#----------Network test input and output processing----------
def load_and_normalized_test_images(data_dir,label_dir):
	"""
	Load and normalize the test set image
	"""
	all_images = []
	all_label = []

	images_list = os.listdir(data_dir)
	images_list.sort(key=lambda x:int(x[:-4]))

	label_list = os.listdir(label_dir)
	label_list.sort(key=lambda x:int(x[:-4]))

	for i in range(len(images_list)):
		img = cv2.imread(data_dir + images_list[i])
		all_images.append(img)

		lab = imageio.imread(label_dir + label_list[i])
		all_label.append(lab)

	X = np.array(all_images)
	X = X / 255 
	X = X.astype(np.float32)

	Y = np.array(all_label)
	Y = Y / 255 
	Y = Y.astype(np.float32)

	return X,Y

def cut_images(images, image_size, overlap):
	"""
	Cropping large-scale original images with overlaps in order into small-scale images
	overlap:Crop the length of the overlapping part of the picture
	"""
	cut_images = []

	for i in images:

		num_x = math.ceil((i.shape[0] - overlap) / (image_size - overlap))
		num_y = math.ceil((i.shape[1] - overlap) / (image_size - overlap))
		image = []

		i = np.pad(i,((0,image_size),(0,image_size),(0,0)),'constant') 
		j = 0

		for x in range(num_x):
			for y in range(num_y):

				tmp = i[x * image_size - (x * overlap) : (x + 1) * image_size - (x * overlap),
						y * image_size - (y * overlap): (y + 1) * image_size - (y * overlap), :]
	
				j += 1
				image.append(tmp)

		cut_images.append(image)

	print("cut images done")
	return cut_images,num_x,num_y

def save_test_output(result_image, output_dir, image_size, target_size, overlap, num_x, num_y, epoch):
	"""
	The test output of the network is spliced back to an image corresponding to the original image in order and saved
	result_image: the output of the network test
	output_dir: the address to save the picture
	image_size: the image size of the network test output
	target_size: original image size
	overlap: the length of the overlapping part of adjacent pictures when cutting
	num_x: The number of cut pictures in the x direction (this parameter is returned by cut_images)
	num_y: The number of cut pictures in the y direction (this parameter is returned by cut_images)
	eopch: The number of network training rounds corresponding to the current test
	"""
	for i in range(len(result_image)):

		output_image = np.zeros(target_size)
		output_image = np.pad(output_image,((0,image_size),(0,image_size),(0,0)),'constant')

		#Splicing sliding step
		stride = image_size - overlap

		j = 0 
		for x in range(num_x):
			for y in range(num_y):

				output_image[x * image_size - (x * overlap) : (x + 1) * image_size - (x * overlap),
						y * image_size - (y * overlap): (y + 1) * image_size - (y * overlap), :] = np.maximum(result_image[i][j][0],
																						  output_image[x * image_size - (x * overlap) : (x + 1) * image_size - (x * overlap),
																						y * image_size - (y * overlap): (y + 1) * image_size - (y * overlap), :])
				j +=1
		img = output_image[0:target_size[0],0:target_size[1]] * 255
		cv2.imwrite((output_dir + "%d_%d.jpg" % (i,epoch)), img)

	print("-----save images done!-----" + "\n")


