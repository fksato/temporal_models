import tensorflow as tf
from tensorflow.python.training import moving_averages

RESBLOCKS = {18: (2, 2, 2, 2), 34: (3, 4, 6, 3)}

def st_conv(t, inputs, in_filters, out_filters, kernel_size, stride=[1, 1, 1], mode=True, name=None, mid_filter=None):
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
		# spatial conv:
		sp_conv = tf.nn.conv3d(inputs, sp_conv_filter, strides=spatial_stride, padding='SAME', name=f'{name}_spconv')
		sp_conv = block_batch_norm(sp_conv, mode=mode)
		sp_conv = tf.nn.relu(sp_conv, name=f'{name}_relu')

		# temporal conv:
		out = tf.nn.conv3d(sp_conv, t_conv_filter, strides=temporal_stride, padding='SAME', name=f'{name}_tconv')

	return out


def st_conv_block(t, inputs, in_filters, out_filters, down_sample=False, mode=True, block_name=None):
	stride = [1, 1, 1]
	unit_id = block_name.split("_")[1]
	if down_sample:
		stride = [2, 2, 2]
	shortcut = inputs
	conv_1 = st_conv(t, inputs, in_filters, out_filters, [3,3,3], mode=mode, name='conv_1')
	#
	conv_1 = block_batch_norm(conv_1, mode=mode, name='conv_1/bn')
	conv_1 = tf.nn.relu(conv_1, name='conv_1_relu')
	conv_2 = st_conv(t, conv_1, out_filters, out_filters, [3,3,3], stride=stride, mode=mode, name='conv_2')
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

		x = tf.nn.conv3d(x, shortcut_conv_filter, strides=stride, padding='SAME', name=f'{name}_shortcut_conv')
		x = block_batch_norm(x, mode=mode)
		return x


def block_batch_norm(x, mode=True, name=None):
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
			tf.summary.histogram(mean.op.name, mean)
			tf.summary.histogram(variance.op.name, variance)

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

	def build_graph(self, inputs, train_mode=True):
		t = 3
		end_points = {}
		with tf.variable_scope('init'):
			out = st_conv(t, inputs, 3, 64, [3, 7, 7], stride=[1, 2, 2],  mode=train_mode, name='conv_1', mid_filter=45)
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

		#
		if train_mode:
			out = tf.nn.dropout(out, keep_prob=0.5)

		# glob_avg_pooling when test
		if not train_mode:
			with tf.variable_scope('average_pool'):
				out = tf.reduce_mean(out, [1, 2, 3], keepdims=True, name='global_pool')
				end_points[out.name[:-2]] = out

		# fc + Softmax
		out_fc = fully_connected(out, filter_in, self.num_class)
		end_points['fc'] = out_fc
		predictions = tf.nn.softmax(out_fc)
		end_points['logits'] = predictions

		return out, end_points
