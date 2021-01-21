import time
import numpy as np

import data_processing

def test(data_dir,label_dir,output_dir,image_size,model_num,fcrn_test_model,target_size,overlap):
	"""
	Call the model to test on the test set
	data_dir: test set address
	label_dir: test set label address
	output_dir: test output address
	image_size: The size of the image sent to the network test
	model_num: the serial number of the test model
	fcrn_test_model: the tested model
	target_size: original image size
	overlap: the overlap length between each small image when cutting the original image
	"""
	#Load test set
	images,_ = data_processing.load_and_normalized_test_images(data_dir,label_dir) 

	cut_time = time.time()
	
	#Cut the original image of the test set in order
	cut_images,num_x,num_y = data_processing.cut_images(images,image_size,overlap) 
	#Print cutting time
	print("cut_time:",(time.time()- cut_time))

	result_image = []

	test_time = time.time()
	print("\n"+"predict images...")

	#Predict one by one
	for i in cut_images:
		result = []
		for j in i:

			#predict
			j = np.reshape(j,(1,image_size,image_size,3))
			j = fcrn_test_model.predict([j])

			result.append(j)
		result_image.append(result)

	print("test time:",time.time() - test_time)

	#Splice small images into large images in order and save
	data_processing.save_test_output(result_image,output_dir,image_size,target_size,overlap,
								  num_x,num_y,model_num)
	





