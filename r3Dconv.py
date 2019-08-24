import tensorflow as tf
from tensorflow.python.training import moving_averages

RESBLOCKS = {18: (2, 2, 2, 2), 34: (3, 4, 6, 3)}

def st_conv(t, inputs, in_filters, out_filters, kernel_size, stride=[1, 1, 1], pad=[1, 1], mode=False, name=None, mid_filter=None):
	with tf.variable_scope(name, default_name='debug_conv'
			, initializer=tf.initializers.variance_scaling()
			, reuse=tf.AUTO_REUSE):
		if mid_filter:
			mid_filter = mid_filter
		else:
			# calculate Mi
			mid_filter = t * (kernel_size[1] * kernel_size[2] * in_filters * out_filters)
			mid_filter /= (kernel_size[1] * kernel_size[2] * in_filters + t * out_filters)
			mid_filter = int(mid_filter)

		sp_conv_filter = tf.get_variable('spatial_weights'
		                                 , [1, kernel_size[1], kernel_size[2], in_filters, mid_filter]
		                                 , trainable=mode
		                                 , dtype=tf.float32 )

		t_conv_filter = tf.get_variable('temporal_weights'
										, [kernel_size[0], 1, 1, mid_filter, out_filters]
										, trainable=mode
										, dtype=tf.float32 )

		spatial_stride = [1, 1, stride[1], stride[2], 1]
		temporal_stride = [1, stride[0], 1, 1, 1]
		spatial_padding = tf.pad(inputs, [[0, 0],[0, 0],[pad[1], pad[1]],[pad[1], pad[1]],[0, 0]], mode='CONSTANT')
		# spatial conv:
		# sp_conv = tf.nn.conv3d(inputs, sp_conv_filter, strides=spatial_stride, padding='SAME', name=f'{name}_spconv')
		sp_conv = tf.nn.conv3d(spatial_padding, sp_conv_filter, strides=spatial_stride, padding='VALID', name=f'{name}_spconv')
		sp_conv = block_batch_norm(sp_conv, mode=mode)

		sp_conv = tf.nn.relu(sp_conv, name=f'{name}_relu')

		# temporal padding:
		temporal_padding = tf.pad(sp_conv, [[0, 0], [pad[0], pad[0]], [0, 0], [0, 0], [0, 0]], mode='CONSTANT')
		# temporal conv:
		# out = tf.nn.conv3d(sp_conv, t_conv_filter, strides=temporal_stride, padding='SAME', name=f'{name}_tconv')
		out = tf.nn.conv3d(temporal_padding, t_conv_filter, strides=temporal_stride, padding='VALID', name=f'{name}_tconv')

	return out


def st_conv_block(t, inputs, in_filters, out_filters, down_sample=False, mode=False, block_name=None):
	stride = [1, 1, 1]
	unit_id = block_name.split("_")[1]
	if down_sample:
		stride = [2, 2, 2]
	shortcut = inputs
	conv_1 = st_conv(t, inputs, in_filters, out_filters, [3,3,3], stride=stride, pad=[1,1],  mode=mode, name='conv_1')
	#
	conv_1 = block_batch_norm(conv_1, mode=mode, name='conv_1/bn')
	conv_1 = tf.nn.relu(conv_1, name='conv_1_relu')
	conv_2 = st_conv(t, conv_1, out_filters, out_filters, [3,3,3], stride=[1, 1, 1], pad=[1,1], mode=mode, name='conv_2')
	conv_2 = block_batch_norm(conv_2, mode=mode, name='conv_2/bn')
	# project shortcut
	if in_filters != out_filters:
		shortcut = project_shortcut(shortcut, in_filters, out_filters, mode, name=f'shortcut_projection_{block_name}')
	conv_2 += shortcut
	return tf.nn.relu(conv_2, name=f'unit_{unit_id}')


#TODO: ResNet-v2 bottleneck
# def st_conv_bottleneck_block(t, inputs, in_filters, out_filters, down_sample=False, mode=True, block_name=None):
# 	stride = [1,1,1]
# 	if down_sample:
# 		stride = [2,2,2]
# 	shortcut = inputs
# 	conv_1 = st_conv(t, inputs, in_filters, out_filters, [1,1,1], stride=stride, name=f'{block_name}_v2/conv_1')
# 	conv_2 = st_conv(t, conv_1, out_filters, out_filters, [3,3,3], mode=mode, name=f'{block_name}_v2/conv_2')
# 	conv_3 = st_conv(t, conv_2, out_filters, out_filters, [1,1,1], mode=mode, name=f'{block_name}_v2/conv_3')
# 	# resize conv_2 for add:
# 	# project shortcut
# 	if in_filters != out_filters:
# 		shortcut = project_shortcut(shortcut, in_filters, out_filters, mode, name=f'shortcut_projection_{block_name}')
# 	conv_3 += shortcut
# 	return tf.nn.relu(conv_3)


def project_shortcut(x, in_filter, out_filter, mode=True, name=None):
	with tf.variable_scope(name, default_name='project_shortcut', reuse=tf.AUTO_REUSE):
		# if in_filter != out_filter:
		stride = [1,2,2,2,1]
		shortcut_conv_filter = tf.get_variable('shortcut_conv_filter'
		                                , [1, 1, 1, in_filter, out_filter]
		                                , trainable=mode
		                                , dtype=tf.float32
										, initializer=tf.initializers.variance_scaling())

		x = tf.nn.conv3d(x, shortcut_conv_filter, strides=stride, padding='VALID', name=f'{name}_shortcut_conv')
		x = block_batch_norm(x, mode=mode)
		return x


def block_batch_norm(x, mode=False, name=None):
	if not name:
		name = 'batch_norm'
	with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
		control_inputs = []
		params_shape = [x.get_shape()[-1]]
		# offset
		beta = tf.get_variable('beta'
							   , params_shape
							   , trainable=mode
							   , dtype=tf.float32
							   , initializer=tf.constant_initializer(0.0, tf.float32))
		# scale
		gamma = tf.get_variable('gamma'
								, params_shape
								, trainable=mode
								, dtype=tf.float32
								, initializer=tf.constant_initializer(1.0, tf.float32))

		if mode:
			mean, variance = tf.nn.moments(x, [0, 1, 2, 3], name='moments')
			moving_mean = tf.get_variable('moving_mean'
										  , params_shape
										  , dtype=tf.float32
										  , initializer=tf.constant_initializer(0.0, tf.float32)
										  , trainable=False)
			moving_variance = tf.get_variable('moving_variance'
											  , params_shape
											  , dtype=tf.float32
											  , initializer=tf.constant_initializer(1.0, tf.float32)
											  , trainable=False)
			# moving_mean = moving_mean * decay + mean * (1 - decay)
			# moving_variance = moving_variance * decay + variance * (1 - decay)
			update_moving_avg = moving_averages.assign_moving_average(moving_mean, mean, 0.9)
			update_moving_var = moving_averages.assign_moving_average(moving_variance, variance, 0.9)
			control_inputs = [update_moving_avg, update_moving_var]
		else:
			mean = tf.get_variable('moving_mean'
								   , params_shape
								   , dtype=tf.float32
			                       , initializer=tf.constant_initializer(0.0, tf.float32)
			                       , trainable=False)
			variance = tf.get_variable('moving_variance'
			                           , params_shape
									   , dtype=tf.float32
			                           , initializer=tf.constant_initializer(1.0, tf.float32)
			                           , trainable=False)
			# tf.summary.histogram(mean.op.name, mean)
			# tf.summary.histogram(variance.op.name, variance)

		# BN：((x-mean)/var)*gamma+beta
		with tf.control_dependencies(control_inputs):
			y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
		y.set_shape(x.get_shape())
		return y

# fc
def fully_connected(x, in_dim, out_dim):
	with tf.variable_scope('fully_connected', reuse=tf.AUTO_REUSE):
		# reshape
		x = tf.reshape(x, [-1, in_dim])
		w = tf.get_variable('fc_weights', [in_dim, out_dim]
							, dtype=tf.float32)

		b = tf.get_variable('biases', [out_dim], initializer=tf.initializers.variance_scaling())
		x = tf.nn.xw_plus_b(x, w, b)
		return x

# L2_loss
def decay(weight_decay_rate):
	costs = []
	for var in tf.trainable_variables():
		if var.op.name.find(r'weights') > 0:
			costs.append(tf.nn.l2_loss(var))
	cost_decay = tf.multiply(weight_decay_rate, tf.add_n(costs))
	tf.summary.scalar('l2_loss', cost_decay)
	return cost_decay


def load_weights(weights_dir, graph, session):
	import pickle as pkl
	import numpy as np
	with open(weights_dir, 'rb') as f:
		weights = pkl.load(f)

	with graph.as_default():
		model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		for var in model_vars:
			if var.name[:-2] not in weights.keys():
				continue
			x = weights[var.name[:-2]]
			if len(x.shape) == 5:
				x = np.transpose(x, [2, 3, 4, 1, 0])
			elif x.shape == (400, 512):
				x = np.transpose(x, [-1, 0])
			var.load(x, session)


class R25DNet_V1:

	def __init__(self, layer_depth, num_class, batch_size, weight_decay_rate, *args, **kwargs):
		self.num_class = num_class
		self.batch_size = batch_size
		self.weight_decay_rate = weight_decay_rate
		self.blocks_depth = RESBLOCKS[layer_depth]

	def build_graph(self, inputs, train_mode=False):
		t = 3
		end_points = {}
		with tf.variable_scope('init'):
			out = st_conv(t, inputs, 3, 64, [3, 7, 7], stride=[1, 2, 2], pad=[1,3],  mode=train_mode, name='conv_1', mid_filter=45)
			out = block_batch_norm(out, train_mode)
			out = tf.nn.relu(out, name='relu_0')
			end_points[out.name[:-2]] = out

		block_num = 0
		filter_in = 64
		filter_out = 64

		for depth in self.blocks_depth:
			for block in range(depth):
				block_id = block_num + block
				down_sample = filter_in != filter_out
				with tf.variable_scope(f'block_{block_id}', reuse=tf.AUTO_REUSE):
					out = st_conv_block(t, out, filter_in, filter_out, down_sample=down_sample
					                    , mode=train_mode, block_name=f'block_{block_id}')
					end_points[out.name[:-2]] = out
				if filter_in != filter_out:
					filter_in = filter_out
			block_num+=depth
			filter_out *= 2

		with tf.variable_scope('average_pool'):
			t_filter = out.shape[1]
			out = tf.nn.avg_pool3d(out, ksize=[1, t_filter, 7, 7, 1], strides=[1, 1, 1, 1, 1], padding='VALID',
			                       name='global_pool')
			end_points[out.name[:-2]] = out

		# fc + Softmax
		out_fc = fully_connected(out, filter_in, self.num_class)
		end_points['fc'] = out_fc
		predictions = tf.nn.softmax(out_fc)
		end_points['logits'] = predictions

		return out, end_points


if __name__=='__main__':
	import sys
	sys.path.append("..")

	import warnings

	warnings.filterwarnings("ignore", category=FutureWarning)

	import tensorflow as tf
	from utils.mdl_utils import mask_unused_gpus

	from models.tf_mdl import TensorflowR3DModel

	batch_size = 2
	weight_decay_rate = 0.05

	f_strides = {8: 3, 16: 3, 32: 3}

	_ = mask_unused_gpus()
	gpu_options = tf.GPUOptions(allow_growth=True)

	depth = 18  # 18, 34
	time_depth = 16  # 8, 16, 32
	stride = f_strides[time_depth]

	dataset = 'kinetics'  # 'kinetics', 'kinetics+sports1m'
	layers = ['average_pool/global_pool']
	model_name = f'r2_1-{depth}_l{time_depth}-{dataset}'

	# model_ctr_kwargs = {'layer_depth': depth
	# 	, 'num_class': 400
	# 	, 'batch_size': batch_size
	# 	, 'weight_decay_rate': weight_decay_rate}

	mdl_id = model_name

	# video temporal blocks 16 frames, 3 frames stride:
	frames_per_video = 75  # total vid size = 2.5 s
	offset = frames_per_video - (time_depth * stride)
	print(offset)
	processed_im_hw = 112
	preprocess = None
	# preprocess = lambda img: tf.image.per_image_standardization(tf.image.crop_to_bounding_box(
	# 	tf.image.resize_images(img, [processed_im_hw, 200], preserve_aspect_ratio=True)
	# 	, 0, 44, processed_im_hw, processed_im_hw))
	# preprocess = lambda img: tf.image.crop_to_bounding_box(
	# 	tf.image.resize_images(img, [processed_im_hw, 200], preserve_aspect_ratio=True)
	# 	, 0, 44, processed_im_hw, processed_im_hw)

	input_kwargs = {'layer_depth': depth
		, 'num_class': 400
		, 'batch_size': batch_size
		, 'weight_decay_rate': weight_decay_rate
        , 'frames_per_video': frames_per_video
		, 'stride': stride
		, 'offset': offset
		, 'preprocess': preprocess
		, 'im_height': processed_im_hw, 'im_width': processed_im_hw
		, 'prefetch': batch_size, 'num_procs': 8}

	ex = TensorflowR3DModel.init(identifier=mdl_id, net_name=model_name, time_depth=time_depth
	                             ,**input_kwargs)

	vid_path = ['/braintree/home/fksato/HACS_total/training/Applying sunscreen/Daily_sunscreen_use_slows_skin_aging_even_in_middle_age_0.mp4']

	activations = ex(vid_path, layers)

	print(activations)