from keras import Input, Model,regularizers
from keras.models import clone_model

from keras.layers import Dense,Conv2D,Conv2DTranspose,Add,Reshape,SeparableConv2D,ReLU
from keras.layers import Concatenate,Lambda,Activation, Dropout,MaxPooling2D,UpSampling2D,BatchNormalization

from keras.applications import VGG16,ResNet50,InceptionV3,DenseNet121,MobileNetV2

regularizer_l2 = 0.00001 # l2 regularization coefficient
drop = 0 # drop ratio

def size_opetion(x,filters,kernel_size,size_op,res,SeparableConv):
	"""
	The part of the residual block that operates on the image size
	x:Input
	filters:Number of convolution kernels
	kernel_size:Convolution kernel size
	size_op:"down" downsampling the picture; "none" does not change the picture size;
	"up" upsampling the picture
	res:0 is the branch of the residual block; 1 is the main path of the residual block
	SeparableConv:"True" to use depth separable convolution; "False" to use ordinary convolution
	"""

	padding = "same"

	if size_op == "down":
		if res == 0:
			if not SeparableConv:
				x = Conv2D(filters=filters[1], kernel_size=kernel_size, strides=2, padding=padding, 
				  kernel_regularizer=regularizers.l2(regularizer_l2))(x)
			else:
				x = SeparableConv2D(filters=filters[1], kernel_size=kernel_size, strides=2, padding=padding,
						depthwise_regularizer=regularizers.l2(regularizer_l2),
						pointwise_regularizer=regularizers.l2(regularizer_l2))(x)
				pass

		else:
			if not SeparableConv:
				x = Conv2D(filters=filters[0], kernel_size=kernel_size, strides=2, padding=padding, 
				  kernel_regularizer=regularizers.l2(regularizer_l2))(x)
			else:
				x = SeparableConv2D(filters=filters[0], kernel_size=kernel_size, strides=2, padding=padding,
						depthwise_regularizer=regularizers.l2(regularizer_l2),
						pointwise_regularizer=regularizers.l2(regularizer_l2))(x)

	if size_op == "none":
		if res == 0:
				pass
		else:
			if not SeparableConv:
				x = Conv2D(filters=filters[0], kernel_size=kernel_size, strides=1, padding=padding, 
				  kernel_regularizer=regularizers.l2(regularizer_l2))(x)
			else:
				x = SeparableConv2D(filters=filters[0], kernel_size=kernel_size, strides=1, padding=padding,
						depthwise_regularizer=regularizers.l2(regularizer_l2),
						pointwise_regularizer=regularizers.l2(regularizer_l2))(x)

	if size_op == "up":
		x = UpSampling2D()(x)
		if res == 0:
			if not SeparableConv:
				x = Conv2D(filters=filters[1], kernel_size=kernel_size, strides=1, padding=padding, 
				  kernel_regularizer=regularizers.l2(regularizer_l2))(x)
			else:
				x = SeparableConv2D(filters=filters[1], kernel_size=kernel_size, strides=1, padding=padding,
						depthwise_regularizer=regularizers.l2(regularizer_l2),
						pointwise_regularizer=regularizers.l2(regularizer_l2))(x)
		else:
			if not SeparableConv:
				x = Conv2D(filters=filters[0], kernel_size=kernel_size, strides=1, padding=padding, 
				  kernel_regularizer=regularizers.l2(regularizer_l2))(x)
			else:
				x = SeparableConv2D(filters=filters[0], kernel_size=kernel_size, strides=1, padding=padding,
						depthwise_regularizer=regularizers.l2(regularizer_l2),
						pointwise_regularizer=regularizers.l2(regularizer_l2))(x)

	return x

def residual_block(x,filters,kernel_size,size_op,block_num,SeparableConv=False):
	"""
	Residual block definition
	x:Input
	filters:Number of convolution kernels
	kernel_size:Convolution kernel size
	size_op:"down" downsampling the picture; "none" does not change the picture size;
	"up" upsampling the picture
	block_num:Which residual block (this parameter has no actual function, it is only easy to locate when an error is reported)
	SeparableConv:"True" to use depth separable convolution; "False" to use ordinary convolution
	"""

	padding = "same"

	res0 = size_opetion(x,filters,kernel_size,size_op,res=0,SeparableConv=SeparableConv)

	res1 = size_opetion(x,filters,kernel_size,size_op,res=1,SeparableConv=SeparableConv)
	res1 = BatchNormalization()(res1)
	res1 = Activation(activation='relu')(res1) 

	if not SeparableConv:
		res1 = Conv2D(filters=filters[1], kernel_size=kernel_size, strides=1, padding=padding, 
				   kernel_regularizer=regularizers.l2(regularizer_l2))(res1)
	else:
		res1 = SeparableConv2D(filters=filters[1], kernel_size=kernel_size, strides=1, padding=padding,
						depthwise_regularizer=regularizers.l2(regularizer_l2),
						pointwise_regularizer=regularizers.l2(regularizer_l2))(res1)

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
		base_model = VGG16(input_shape=input_shape,weights="./vgg16_weights_tf_dim_ordering_tf_kernels.h5")
		model = Model(inputs=base_model.input, outputs=base_model.get_layer('Choose the layer you want').output)		

	if F_name == 'InceptionV3':
		base_model = InceptionV3(input_shape=input_shape,weights="./inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
		model = Model(inputs=base_model.input, outputs=base_model.get_layer('Choose the layer you want').output)

	if F_name == 'ResNet50':
		base_model = ResNet50(input_shape=input_shape,weights="./resnet50_weights_tf_dim_ordering_tf_kernels.h5")
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


def build_PFFN(image_size, res_num, F_name, F, F_trainable, SeparableConv=False):
	"""
	Construct an PFFN network built using deep separable convolutional residual blocks, 
	and splice the high-dimensional image features output by the pre-trained imagenet feature extractor in the middle layer
	image_size:The size of the input image
	res_num:Number of residual blocks for network center feature extraction
	F_name:Selected pre-trained model name
	F:Loaded pre-trained model
	F_trainable:"True" means that the pre-trained model can continue to be trained, and "Fasle" will set the pre-trained model as untrainable
	"""
	input_shape_1 = (image_size,image_size,3) # (None,None,3)
	input_layer_1 = Input(shape = input_shape_1)

	F.trainable = F_trainable
	features_1,features_2,features_3,features_4,features_5 = F([input_layer_1])

	f = [features_1,features_2,features_3,features_4,features_5]
	filters_f = [64,128,256,512]

	#The channel transformation of the intermediate output of the pre-training model
	
	for i in range(len(f)):

		if SeparableConv:
			f[i] = SeparableConv2D(filters=filters_f[i], 
							kernel_size=3, strides=1, padding='same',
							depthwise_regularizer=regularizers.l2(regularizer_l2),
							pointwise_regularizer=regularizers.l2(regularizer_l2))(f[i])
		else:
			f[i] = Conv2D(filters=filters_f[i], 
							kernel_size=3, strides=1, padding='same', 
							kernel_regularizer=regularizers.l2(regularizer_l2))(f[i])

		f[i] = BatchNormalization()(f[i])
		f[i] = Activation(activation='relu')(f[i])
		f[i] = Dropout(drop)(f[i])

	x_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', 
			  kernel_regularizer=regularizers.l2(regularizer_l2))(input_layer_1) #(256,256,64)
	x_1 = Activation(activation='relu')(x_1)

	x_2 = residual_block(x_1,[64,64],3,size_op="down",block_num='down_1',SeparableConv=SeparableConv) #128
	x_2 = Dropout(drop)(x_2)

	x_3 = residual_block(x_2,[64,128],3,size_op="down",block_num='down_2',SeparableConv=SeparableConv) #64
	x_3 = Dropout(drop)(x_3)

	x_4 = residual_block(x_3,[128,256],3,size_op="down",block_num='down_3',SeparableConv=SeparableConv) #32
	x_4 = Dropout(drop)(x_4)

	x_5 = residual_block(x_4,[256,512],3,size_op="down",block_num='down_4',SeparableConv=SeparableConv) #16
	x_5 = Dropout(drop)(x_5)

	x = residual_block(x_5,[512,512],3,size_op="none",block_num='none_1',SeparableConv=SeparableConv) #16
	x = Dropout(drop)(x)
	x = Concatenate(axis=-1)([x,f[3]])

	x = residual_block(x,[512,256],3,size_op="up",block_num='up_1',SeparableConv=SeparableConv)#32
	x = Dropout(drop)(x)

	x = Concatenate(axis=-1)([x,x_4,f[2]])

	x = residual_block(x,[256,128],3,size_op="up",block_num='up_2',SeparableConv=SeparableConv) #64
	x = Dropout(drop)(x)

	x = Concatenate(axis=-1)([x,x_3,f[1]])

	x = residual_block(x,[128,64],3,size_op="up",block_num='up_3',SeparableConv=SeparableConv) #128
	x = Dropout(drop)(x)

	x = Concatenate(axis=-1)([x,x_2,f[0]])

	x = residual_block(x,[64,64],3,size_op="up",block_num='up_4',SeparableConv=SeparableConv) #256
	x = Dropout(drop)(x)

	x = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', 
			kernel_regularizer=regularizers.l2(regularizer_l2))(x)

	x = Activation(activation='sigmoid')(x)

	output = x

	PFFN_res_model = Model(inputs=[input_layer_1], outputs=[output], name='PFFN')

	return PFFN_res_model
