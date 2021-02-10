import os
import cv2
import time
import numpy as np

import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

import PFFN_net
import data_processing
import PFFN_test

def BCD_loss(y_true, y_pred):
	"""
	Bray-Curtis distance
	"""
	loss = tf.reduce_mean(tf.abs(y_pred - y_true)) / (tf.reduce_mean(tf.abs(y_pred + y_true)) + 0.0001)

	return loss

def l_r(epoch):
	"""
	Step learning rate
	"""
	if epoch >= 1200:
		return 0.000008
	if epoch >= 600:
		return 0.00004
	else:
		return 0.0002

#-----------------------------train-----------------------------------
def train_PFFN():

	#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

	train = True

	data_dir = "./data/vehicle/"
	label_dir = "./data/2/"
	txt_dir = "./data/vehicle_txt/vehicle_munich_combine2.txt"

	test_data_dir = "./data/vehicle_test/"
	test_label_dir = "./data/label_test/"

	output_dir = "./test_results_Munich/test_results_PCD/"

	model_name = 'MobileNetV2'

	#Separator of training set and test set
	cut_off = 10
	image_size = 256
	
	#The number of residual blocks added in the middle layer of the network
	res_num = 0
	image_num = 5

	epochs = 1000
	batch_size = 16
	learning_rate = 0.0002

	optimizer = Adam(lr=learning_rate, beta_1=0.5, beta_2=0.999)

	#Load pre-trained model
	F = PFFN_net.build_F_Multiple_output(image_size,model_name)
	#Initialize PCD
	PFFN_model = PFFN_net.build_PFFN(image_size,res_num,F_name = model_name,F = F,F_trainable = True,SeparableConv=False)

	#Define model loss function and optimizer
	PFFN_model.compile(loss=BCD_loss, optimizer=optimizer)

	tensorboard = TensorBoard(log_dir="./logs/PCD{}".format(time.time()),
							write_images=True, write_grads=True, write_graph=True)
	tensorboard.set_model(PFFN_model)

	images, labels = data_processing.load_original_images(data_dir,label_dir,cut_off,train=True)
	rows, cols = labels[0].shape
	target_size = (rows, cols, 1)

	for epoch in range(epochs):
		
		start_time = time.time()
		print("loading...")

		X, Y = data_processing.sample_data(images,labels,cut_off,batch_size,image_size,txt_dir)

		X = np.array(X)
		X = X / 255 
		X = X.astype(np.float32)

		Y = np.array(Y)
		Y = Y / 255 
		Y = Y.astype(np.float32)

		#Print data loading time
		load_time = time.time() - start_time
		print("load_time:",load_time)

		print("Epoch:{}".format(epoch))
		start_time = time.time()

		losses = []

		# set new lr
		K.set_value(PFFN_model.optimizer.lr, l_r(epoch))  

		num_batches = int(X.shape[0] / batch_size)

		for index in range(num_batches):

			image_batch = X[index * batch_size:(index + 1) * batch_size]
			label_batch = Y[index * batch_size:(index + 1) * batch_size]
			label_batch = np.reshape(label_batch,(batch_size,image_size,image_size,1))
			"""
			Train model
			"""
			loss = PFFN_model.train_on_batch(image_batch, label_batch)
			losses.append(loss)

		print("loss:", np.mean(losses))

		#Record loss to log
		data_processing.write_log(tensorboard, 'PFFN_loss', np.mean(losses), epoch)

		#Print training time
		print("Time:", (time.time() - start_time))

		if epoch % 5 == 0: 

			print("-----Sample some images and save them-----")

			#Randomly take two pictures from the training set
			n = int(np.random.randint(0,X.shape[0] - 2,1))

			image = X[n:n + 2]
			label = Y[n:n + 2]

			result_image = PFFN_model.predict([image])
			
			cv2.imwrite("./results/{}_0_image.png".format(epoch),image[0] * 255)
			cv2.imwrite("./results/{}_0_result_image.png".format(epoch),result_image[0] * 255)
			cv2.imwrite("./results/{}_0_label.png".format(epoch),label[0] * 255)

			cv2.imwrite("./results/{}_1_image.png".format(epoch),image[1] * 255)
			cv2.imwrite("./results/{}_1_result_image.png".format(epoch),result_image[1] * 255)
			cv2.imwrite("./results/{}_1_label.png".format(epoch),label[1] * 255)

		if (epoch + 1) % 100 == 0:
			""" 
			Save models
			"""
			PFFN_model.save("./model/PCD_model_%d.h5" % epoch)
			print("---save!---" + "\n")

		if (epoch + 1) % 10 == 0:

			print("-----testing-----")
			"""
			test models
			"""
			PFFN_test.test(data_dir=test_data_dir,label_dir=test_label_dir,output_dir=output_dir,
							image_size=image_size,model_num=epoch,PFFN_test_model=PFFN_model,
							target_size=target_size,overlap=int(image_size / 4))

		print("--------------------------")



if __name__ == '__main__':

	train_PCD()


