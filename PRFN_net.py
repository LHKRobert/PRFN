from keras import Input, Model,regularizers
from keras.models import clone_model
from keras.layers import Dense,Conv2D,Conv2DTranspose,Add,Reshape,ReLU
from keras.layers import Concatenate,Lambda,Activation, Dropout,MaxPooling2D,UpSampling2D,BatchNormalization
from keras.applications import VGG16,ResNet50,InceptionV3,DenseNet121,MobileNetV2

regularizer_l2 = 0.00001 # l2 regularization coefficient
drop = 0 # drop ratio

def size_opetion(x,filters,kernel_size,size_op,res):
	"""
	The part of the residual block that operates on the image size
	x:Input
	filters:Number of convolution kernels
	kernel_size:Convolution kernel size
	size_op:"down" downsampling the picture; "none" does not change the picture size;
	"up" upsampling the picture
	res:0 is the branch of the residual block; 1 is the main path of the residual block
	"""

	padding = "same"

	if size_op == "down":
		if res == 0:
			x = Conv2D(filters=filters[1], kernel_size=kernel_size, strides=2, padding=padding, 
				kernel_regularizer=regularizers.l2(regularizer_l2))(x)

		else:
			x = Conv2D(filters=filters[0], kernel_size=kernel_size, strides=2, padding=padding, 
				kernel_regularizer=regularizers.l2(regularizer_l2))(x)

	if size_op == "none":
		if res == 0:
				x = Conv2D(filters=filters[0], kernel_size=kernel_size, strides=1, padding=padding, 
				  kernel_regularizer=regularizers.l2(regularizer_l2))(x)

	if size_op == "up":
		x = UpSampling2D()(x)
		if res == 0:
			x = Conv2D(filters=filters[1], kernel_size=kernel_size, strides=1, padding=padding, 
				kernel_regularizer=regularizers.l2(regularizer_l2))(x)
		else:
			x = Conv2D(filters=filters[0], kernel_size=kernel_size, strides=1, padding=padding, 
				kernel_regularizer=regularizers.l2(regularizer_l2))(x)

	return x

def residual_block(x,filters,kernel_size,size_op,block_num):
	"""
	Residual block definition
	x:Input
	filters:Number of convolution kernels
	kernel_size:Convolution kernel size
	size_op:"down" downsampling the picture; "none" does not change the picture size;
	"up" upsampling the picture
	block_num:Which residual block (this parameter has no actual function, it is only easy to locate when an error is reported)
	"""

	padding = "same"

	res0 = size_opetion(x,filters,kernel_size,size_op,res=0)

	res1 = size_opetion(x,filters,kernel_size,size_op,res=1)
	res1 = BatchNormalization()(res1)
	res1 = Activation(activation='relu')(res1) 

	res1 = Conv2D(filters=filters[1], kernel_size=kernel_size, strides=1, padding=padding, 
				kernel_regularizer=regularizers.l2(regularizer_l2))(res1)

	res1 = BatchNormalization()(res1)
	
	res = Add()([res0, res1])
	res = Activation(activation='relu')(res)

	return res

def build_F_Multiple_output(image_size,F_name,clone=False):
	"""
	Build feature network to extract image features
	image_size:The size of the input image
	F_name:Selected pre-trained model name
	clone:"True" is the clone model, that is, the network structure of the pre-trained model is loaded but the weight is not loaded; "False" is the structure and weight of the pre-trained model
	"""
	input_shape = (image_size,image_size,3)
	input_layer = Input(shape=input_shape)

	#-----Load a pre-trained model trained on 'Imagenet' dataset-----

	if F_name == 'VGG16':
		base_model = VGG16(input_shape=input_shape,weights="./The address of the .h5 file")
		model = Model(inputs=base_model.input, outputs=base_model.get_layer('Choose the layer you want').output)		

	if F_name == 'InceptionV3':
		base_model = InceptionV3(input_shape=input_shape,weights="./The address of the .h5 file")
		model = Model(inputs=base_model.input, outputs=base_model.get_layer('Choose the layer you want').output)

	if F_name == 'ResNet50':
		base_model = ResNet50(input_shape=input_shape,weights="./The address of the .h5 file")
		model = Model(inputs=base_model.input, outputs=[base_model.get_layer('Choose the layer you want').output])

	if F_name == 'MobileNetV2':
		base_model = MobileNetV2(input_shape=input_shape, weights="./model/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5")
		base_model.summary()
		model = Model(inputs=base_model.input, outputs=[base_model.get_layer('block_1_expand_relu').output,
														base_model.get_layer('block_3_expand_relu').output,
														base_model.get_layer('block_6_expand_relu').output,
														base_model.get_layer('block_13_expand_relu').output])

	
	if clone:
		model = clone_model(model)

	features_1,features_2,features_3,features_4 = model(input_layer)

	# Create a Keras model
	model = Model(inputs=[input_layer], outputs=[features_1,features_2,features_3,features_4])
	return model


def build_PRFN(image_size, F_name, F, F_trainable):
	"""
	F: per-training model
	"""
	input_shape_1 = (image_size,image_size,3)
	input_layer_1 = Input(shape = input_shape_1)

	F.trainable = F_trainable
	features_1,features_2,features_3,features_4 = F([input_layer_1])

	f = [features_1,features_2,features_3,features_4]
	filters_f = [64,128,256,256]

	if F_name == 'VGG16':
		pass
	else:
		for i in range(len(f)):

			#if i == 4:
			f[i] = UpSampling2D()(f[i])

			f[i] = Conv2D(filters=filters_f[i], 
							kernel_size=1, strides=1, padding='same', 
							kernel_regularizer=regularizers.l2(regularizer_l2))(f[i])

			f[i] = BatchNormalization()(f[i])
			f[i] = Activation(activation='relu')(f[i])

	x_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', 
			  kernel_regularizer=regularizers.l2(regularizer_l2))(input_layer_1) #(256,256,64)
	x_1 = Activation(activation='relu')(x_1)

	x_2 = residual_block(x_1,[64,64],3,size_op="down",block_num='down_1') #128

	x_3 = residual_block(x_2,[128,128],3,size_op="down",block_num='down_2') #64
	x_3 = residual_block(x_3,[128,128],3,size_op="none",block_num='none_2') #64

	x_4 = residual_block(x_3,[256,256],3,size_op="down",block_num='down_3') #32
	x_4 = residual_block(x_4,[256,256],3,size_op="none",block_num='none_3') #32

	x_5 = residual_block(x_4,[512,512],3,size_op="down",block_num='down_4') #16
	x_5 = residual_block(x_5,[512,512],3,size_op="none",block_num='none_4') #16

	x = residual_block(x_5,[512,512],3,size_op="none",block_num='none') #16
	x = residual_block(x_5,[512,512],3,size_op="none",block_num='none')

	x = conv_bn_relu_drop(x,filters=256,kernel_size=1,strides=1,padding='same',activation='relu')

	#----------Multi-level concatenate----------
	x_m = Concatenate(axis=-1)([x_2,f[0]])
	x_m = conv_bn_relu_drop(x_m,filters=64,kernel_size=1,strides=1,padding='same',activation='relu')
	x_m = residual_block(x_m,[64,64],3,size_op="down",block_num='down')

	x_m = Concatenate(axis=-1)([x_m,x_3,f[1]])
	x_m = conv_bn_relu_drop(x_m,filters=128,kernel_size=1,strides=1,padding='same',activation='relu')
	x_m = residual_block(x_m,[128,128],3,size_op="down",block_num='down')

	x_m = Concatenate(axis=-1)([x_m,x_4,f[2]])
	x_m = conv_bn_relu_drop(x_m,filters=256,kernel_size=1,strides=1,padding='same',activation='relu')
	x_m = residual_block(x_m,[256,256],3,size_op="down",block_num='down')

	x_m = Concatenate(axis=-1)([x_m,x_5,f[3]])
	x_m = conv_bn_relu_drop(x_m,filters=512,kernel_size=1,strides=1,padding='same',activation='relu')
	x_m = residual_block(x_m,[512,512],3,size_op="none",block_num='none') #16
	x_m = conv_bn_relu_drop(x_m,filters=256,kernel_size=1,strides=1,padding='same',activation='relu')

	#----------core----------
	x = Concatenate(axis=-1)([x,f[3],x_m])

	#----------Decoder----------

	x = residual_block(x,[512,256],3,size_op="up",block_num='up_1')#32
	x = residual_block(x,[256,256],3,size_op="none",block_num='up_1')

	x = Concatenate(axis=-1)([x,x_4,f[2]])
	x = residual_block(x,[256,128],3,size_op="up",block_num='up_2') #64
	x = residual_block(x,[128,128],3,size_op="none",block_num='up_1')

	x = Concatenate(axis=-1)([x,x_3,f[1]])
	x = residual_block(x,[128,64],3,size_op="up",block_num='up_3') #128
	x = residual_block(x,[64,64],3,size_op="none",block_num='up_1')

	x = Concatenate(axis=-1)([x,x_2,f[0]])
	x = residual_block(x,[64,64],3,size_op="up",block_num='up_4') #256

	x = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', 
			kernel_regularizer=regularizers.l2(regularizer_l2))(x)

	x = Activation(activation='sigmoid')(x)

	output = x

	PRFN_res_model = Model(inputs=[input_layer_1], outputs=[output], name='PRFN')

	return PRFN_res_model

