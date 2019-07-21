import sys
sys.path.append("..")

import xarray as xr

import tensorflow as tf
import pickle as pk
import numpy as np

import tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def load_act_from_group(group_path, act_index, labels, frame_block_cnts, shuffle=False):
	# np.random.seed(1234)
	if shuffle:
		data = list(zip(act_index, labels))
		np.random.shuffle(data)
		data = list(zip(*data))
		act_index, labels = list(data[0]), list(data[1])

	act_index = [frame_block_cnts * index + i for index in act_index for i in range(frame_block_cnts)]
	labels = [label for label in labels for _ in range(frame_block_cnts)]

	with open(group_path, 'rb') as f:
		activations = pk.load(f)
		activations = activations['data']

	group_activations = activations[act_index].values.astype(np.float32)
	group_stim_paths = activations[act_index].stimulus_path.values

	assert len(labels) == group_activations.shape[0] == group_stim_paths.shape[0]

	return group_stim_paths, group_activations, labels


def tf_load_act_from_group(group_path, act_index, labels, frame_block_cnts, shuffe=False):

	def loader(group_path, act_index, labels, frame_block_cnts, shuffe=False):
		stim_paths, act, action_labels = tf.py_func(load_act_from_group
		                                            , [group_path, act_index, labels, frame_block_cnts, shuffe]
		                                            , [tf.string, tf.float32, tf.int32])
		return tf.data.Dataset.from_tensor_slices(stim_paths), tf.data.Dataset.from_tensor_slices(act)\
			, tf.data.Dataset.from_tensor_slices(action_labels)

	return tf.data.Dataset.zip(loader(group_path, act_index, labels, frame_block_cnts, shuffe))


def activations_dataset(group_paths, act_index, group_labels, frame_block_cnts, batch_size, train=True, num_procs=8):
	if not isinstance(group_paths, list):
		input_dataset = tf_load_act_from_group(group_paths, act_index, group_labels, frame_block_cnts, shuffe=train)
	else:
		group_path_ds = tf.data.Dataset.from_tensor_slices(group_paths)
		act_index_ds = tf.data.Dataset.from_generator(lambda: act_index, tf.int32, output_shapes=[None])
		labels_ds = tf.data.Dataset.from_generator(lambda: group_labels, tf.int32, output_shapes=[None])

		grouped_ds = tf.data.Dataset.zip((group_path_ds, act_index_ds, labels_ds))

		# shuffle groups
		if train:
			grouped_ds = grouped_ds.shuffle(buffer_size=len(group_paths), seed=1234)

		input_dataset = grouped_ds.interleave(lambda path, idx, labels:
		                                      tf_load_act_from_group(path, idx, labels, frame_block_cnts=frame_block_cnts
		                                                             , shuffe=train)
		                                      , cycle_length=num_procs
		                                      , block_length=batch_size
		                                      , num_parallel_calls=num_procs)

	input_dataset = input_dataset.batch(batch_size=batch_size).prefetch(buffer_size=batch_size)

	return input_dataset


def check_valid_indices(group_paths, indices, labels):
	invalid_group = [group for group in indices.keys() if len(indices[group]) == 0]
	if len(invalid_group) > 0:
		_group_paths = [i for ix, i in enumerate(group_paths) if ix not in invalid_group]
		[(indices.pop(i), labels.pop(i)) for i in invalid_group]
	else:
		_group_paths = group_paths

	return _group_paths, indices, labels


def package_predictions(paths, labels, predictions):
	# pickle coords/dims:
	ds = xr.DataArray(predictions
	                  , coords={'stim_path': ('stimuli', paths)
	                            , 'label' : ('stimuli', labels)}
	                  , dims=['stimuli', 'predictions'])
	ds = ds.set_index(stimuli=['stim_path', 'label'], append=True)
	return ds


class LogRegModel:
	def __init__(self, features, num_class, data_X, data_y, lr=1e-4, weight_decay=None
	             , activation=None):
		self.feature_size = features
		self.num_class = num_class
		self.weight_decay = weight_decay or 0.5
		self.fc_dropout_keep_prob = tf.placeholder(tf.float32)
		self.lr = lr
		self._opt = tf.train.AdamOptimizer(learning_rate=self.lr)
		self.build(data_X, data_y, activation)

	def build(self, inputs, labels, activation):
		self._fc_layer(inputs, activation)
		self._make_loss(labels)
		self.network_summary = tf.summary.merge_all()

	def _fc_layer(self, inputs, activation=None):
		self.weights = tf.get_variable('W', [self.feature_size, self.num_class]
		                          , initializer=tf.contrib.layers.xavier_initializer()
		                          , regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
		biases = tf.get_variable('b', [self.num_class]
		                         , initializer=tf.initializers.constant()
		                         )
		self.saver = tf.train.Saver({'W': self.weights, 'b': biases})

		self.predictions = tf.nn.dropout(inputs, keep_prob=self.fc_dropout_keep_prob, name="dropout_out")

		self.predictions = tf.add(tf.matmul(self.predictions, self.weights), biases)

		if activation is not None:
			self.predictions = getattr(tf.nn, activation)(self.predictions, name=activation)

		# if self.log_summary:
		with tf.name_scope('wieghts_summary'):
			mean_w = tf.reduce_mean(self.weights)
			std = tf.reduce_mean(tf.square(self.weights-mean_w))
			tf.summary.histogram('mean', mean_w)
			tf.summary.histogram('std', std)
			tf.summary.histogram('max', tf.reduce_max(self.weights))
			tf.summary.histogram('min', tf.reduce_min(self.weights))
			tf.summary.histogram('max', self.weights)
		with tf.name_scope('fc_activations_summary'):
			_preds = tf.nn.softmax(self.predictions)
			_argmax = tf.argmax(_preds)
			tf.summary.histogram('fc_softmax', _preds)
			tf.summary.histogram('fc_argamx', _argmax)

	def _make_loss(self, input_labels):
		with tf.variable_scope('loss'):
			logits = self.predictions

			self.acc, self.acc_op = tf.metrics.accuracy(labels=input_labels,
			                                            predictions=tf.argmax(tf.nn.softmax(logits, axis=-1),
			                                                                  axis=-1))

			self.classification_error = tf.reduce_mean(
				tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=input_labels))
			self.reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
			self.total_loss = tf.add(self.classification_error, self.reg_loss)
			self.tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
			self.train_op = self._opt.minimize(self.total_loss, var_list=self.tvars,
			                                   global_step=tf.train.get_or_create_global_step())

		# if self.log_summary:
		with tf.name_scope('training_summary'):
			tf.summary.scalar('classification_error', self.classification_error)
			tf.summary.scalar('regularization_loss', self.reg_loss)
			tf.summary.scalar('accuracy', self.acc)


def fit(graph, model, handle, train_iterator, val_iterator, train_val_iterator, activations_cnt, val_cnt, num_epoch
        , batch_size, mdl_name, keep_prob=1.0, TOL=1e-4, log_rate=10, gpu_options=None, checkpoint_save=False):

	train_writer = tf.summary.FileWriter(f'/braintree/home/fksato/Projects/models/log_regression/model_summaries'
	                                     f'/{mdl_name}_train', graph)
	val_writer = tf.summary.FileWriter(f'/braintree/home/fksato/Projects/models/log_regression/model_summaries'
	                                     f'/{mdl_name}_val')

	with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		sess.run(tf.global_variables_initializer())
		train_val_string = sess.run(train_val_iterator.string_handle())

		for step in tqdm.tqdm(range(num_epoch), unit_scale=1, desc="Epoch Training"):
			sess.run(train_iterator)
			sess.run(tf.local_variables_initializer())
			for minibatch_train in tqdm.tqdm(range(0, activations_cnt, batch_size), unit_scale=batch_size
					, desc="Traininng"):
				try:
					train_summary, _, train_err_loss, train_reg_loss, _op = sess.run([model.network_summary
					                                                            , model.train_op
					                                                            , model.classification_error
					                                                            , model.reg_loss, model.acc_op]
						, feed_dict={handle: train_val_string, model.fc_dropout_keep_prob: keep_prob})
				except tf.errors.OutOfRangeError:
					break

			train_acc = sess.run(model.acc)
			train_writer.add_summary(train_summary, step)

			if train_err_loss < TOL:
				print(f'Converged with train accuracy: {train_acc:.4f} train error loss: {train_err_loss:.4f} '
				      f'reg_loss: {train_reg_loss:.4f}\n')
				break

			if (step + 1) % log_rate == 0 and step > 0:
				sess.run(val_iterator)
				sess.run(tf.local_variables_initializer())
				for minibatch_val in tqdm.tqdm(range(0, val_cnt, batch_size), unit_scale=batch_size
						, desc="Validation"):
					try:
						val_summary, val_loss, val_reg_loss, _op = sess.run([model.network_summary
							                                       , model.classification_error
							                                       , model.reg_loss, model.acc_op]
																, feed_dict={handle: train_val_string
						                                               , model.fc_dropout_keep_prob: 1.0})
					except tf.errors.OutOfRangeError:
						pass

				val_acc = sess.run(model.acc)
				val_writer.add_summary(val_summary, step)

				print(f'\nEpoch: {step + 1}')
				print(f'Training accuracy: {train_acc:.4f} Training loss: {train_err_loss:.4f}'
				      f', regularization loss: {train_reg_loss:.4f}')
				print(f'Validation accuracy: {val_acc:.4f} Validation loss: {val_loss:.4f}'
				      f', regularization loss: {val_reg_loss:.4f}\n')

				if checkpoint_save:
					model.saver.save(sess, f'/braintree/home/fksato/Projects/models/'
					f'log_regression/check_points/{mdl_name}_{step}.ckpt')

		model.saver.save(sess, f'/braintree/home/fksato/Projects/models/log_regression/check_points/'
		f'{mdl_name}_final.ckpt')


def predict(graph, model, mdl_name, iter_get_next, test_cnt, batch_size, save_load=False, checkpoint_name="final", gpu_options=None):

	preds = np.zeros((test_cnt, 200), dtype=np.float)
	stim_paths = []
	labels = []
	with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		if save_load:
			model.saver.restore(sess, f'/braintree/home/fksato/Projects/models/log_regression/check_points/'
				f'{mdl_name}_{checkpoint_name}.ckpt')

		sess.run(tf.global_variables_initializer())
		batch_start = 0
		for minibatch_test in tqdm.tqdm(range(0, test_cnt, batch_size), unit_scale=batch_size
				, desc="Testing"):
			try:
				frame_paths, frame_labels, predictions = sess.run([iter_get_next[0], iter_get_next[2]
					                                                  , model.predictions]
				                                                  , feed_dict={model.fc_dropout_keep_prob: 1.0})
				preds[batch_start:batch_start+predictions.shape[0]] = predictions
				stim_paths += list(frame_paths)
				labels += list(frame_labels)
				batch_start += predictions.shape[0]
			except tf.errors.OutOfRangeError:
				pass

		assert preds.shape[0] == len(stim_paths) == len(labels)

	return package_predictions(stim_paths, labels, preds)


def main(train, mdl_code, weight_decay, batch_size=64, lr=1e-4, num_epoch=1000, keep_prob=1.0, log_rate=10, TOL=1e-4, verbose=True
         , save_load=False, save_name='DEBUG', num_procs=8, *args, **kwargs):

	from models import HACS_ACTION_CLASSES as actions
	from utils.mdl_utils import get_tain_test_groups
	from logistic_regression.run_logistic_regression import mdls, mdl_ids, mdl_frames_blocks, input_size, confirm
	from utils.mdl_utils import mask_unused_gpus
	from math import ceil

	groupped_path_cnt = {'0': 2, '1': 25, '2': 25, '3': 25, '4': 15}

	mask_unused_gpus()
	gpu_options = tf.GPUOptions(allow_growth=True)

	mdl_name = mdls[mdl_code]
	mdl_id = mdl_ids[mdl_code]
	frames_block_cnt = mdl_frames_blocks[mdl_code]
	feature_size = input_size[mdl_code]
	max_vid_count = groupped_path_cnt[mdl_code]
	# max_vid_count = 25
	num_epoch = num_epoch
	batch_size = ceil(batch_size/frames_block_cnt) * frames_block_cnt # guarantee full videos are represented per batch
	log_rate = log_rate
	TOL = TOL

	num_class = 200

	result_caching = '/braintree/home/fksato/.result_caching/'
	model_act = 'model_tools.activations.core.VideoActivationsExtractorHelper._from_paths_stored/'
	result_cache_dir = f'{result_caching}{model_act}'

	if mdl_code == '0':
		group_paths = [f'{result_cache_dir}identifier={mdl_id},stimuli_identifier={mdl_name}' \
			f'_full_HACS-200_group_{i}_reduced_set.pkl' for i in range(1001)]
	else:
		group_paths = [f'{result_cache_dir}identifier={mdl_id},stimuli_identifier={mdl_name}' \
			f'_full_HACS-200_group_{i}.pkl'for i in range(81)]

	if mdl_code == '0':
		assert len(group_paths) == 1001
	else:
		assert len(group_paths) == 81

	if verbose:
		print(f'::::::::LOGISTIC REGRESSION::::::::::::')
		print(f':::::::::MODEL INFORMATION:::::::::::::')
		print(f'MODE: {"train" if train else "predict"}')
		print(f'Performance evaluation on {mdl_name}\nmodel id: {mdl_id}\nframe/block count: {frames_block_cnt}')
		print(f'activation directory: {result_cache_dir}')
		print(f'file name pattern: identifier={mdl_id},stimuli_identifier={mdl_name}_full_HACS-200_group_<group-num>'
		      f'{"_reduced_set" if mdl_code == "0" else ""}.pkl')

		if not confirm():
			return -1

	test_size = .1
	validation_size = .1
	vid_dir = '/braintree/home/fksato/HACS_total/training'
	train_idx, test_idx, val_idx, train_labels, test_labels, val_labels \
		= get_tain_test_groups(actions, validation_size, test_size, max_vid_count, vid_dir)

	# Pycharm is annoying
	train_group_paths, val_group_paths, test_group_paths = None, None, None
	train_cnt, val_cnt, test_cnt = 0, 0, 0

	if train:
		# check valid indx:
		val_group_paths, val_idx, val_labels = check_valid_indices(group_paths, val_idx, val_labels)
		train_group_paths, train_idx, train_labels = check_valid_indices(group_paths, train_idx, train_labels)
		train_cnt = sum([len(train_idx[group]) for group in train_idx.keys()])
		train_cnt = train_cnt * frames_block_cnt
		val_cnt = sum([len(val_idx[group]) for group in val_idx.keys()]) * frames_block_cnt

		train_idx, train_labels = train_idx.values(), train_labels.values()
		val_idx, val_labels = val_idx.values(), val_labels.values()

	else:
		test_group_paths, test_idx, test_labels = check_valid_indices(group_paths, test_idx, test_labels)
		test_cnt = sum([len(test_idx[group]) for group in test_idx.keys()]) * frames_block_cnt

		test_idx, test_labels = test_idx.values(), test_labels.values()

	# mdl:
	graph = tf.Graph()
	activation = None

	with graph.as_default():
		if train:
			train_dataset = activations_dataset(train_group_paths, train_idx, train_labels, frames_block_cnt, batch_size
			                                    , train=True, num_procs=num_procs)

			val_dataset = activations_dataset(val_group_paths, val_idx, val_labels, frames_block_cnt, batch_size
			                                  , train=False, num_procs=num_procs)

			train_val_iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
			train_iterator = train_val_iterator.make_initializer(train_dataset)
			val_iterator = train_val_iterator.make_initializer(val_dataset)

			# Feedable iterator assigns each iterator a unique string handle it is going to work on
			handle = tf.placeholder(tf.string, shape=[])
			iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types,
			                                               train_dataset.output_shapes)
		else:
			test_dataset = activations_dataset(test_group_paths, test_idx, test_labels, frames_block_cnt
			                                   , batch_size, train=False, num_procs=num_procs)
			activation = 'softmax'
			iterator = test_dataset.make_one_shot_iterator()

		next_element = iterator.get_next()

		model = LogRegModel(feature_size, num_class, next_element[1], next_element[2], lr=lr, weight_decay=weight_decay
		                    , activation=activation)

	mdl_save_name = f'{mdl_name}_{save_name}'
	if train:
		fit(graph, model, handle, train_iterator, val_iterator, train_val_iterator, train_cnt, val_cnt, num_epoch
		    , batch_size, mdl_save_name, keep_prob=keep_prob, TOL=TOL, log_rate=log_rate, gpu_options=gpu_options
		    , checkpoint_save=save_load)
	else:
		preds = predict(graph, model, mdl_name, next_element, test_cnt, batch_size, save_load, gpu_options)
		print(f'saving: {test_cnt/frames_block_cnt}')
		preds = preds.reset_index(['stimuli'])
		preds.to_netcdf(f'/braintree/home/fksato/Projects/models/log_regression/predictions'
		                f'/{mdl_save_name}_predictions.nc')


if __name__=='__main__':

	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("train", type=lambda x: (str(x).lower() in ['true','1', 'yes'])
	                    , help="train model: 1, predict: 0")
	parser.add_argument("mdl_code", help="0: alexnet, 1: resnet18, 2: resent34, 3: resnet 2.5D 18, 4: resnet 2.5D 34")
	parser.add_argument("weight_decay", help="weight decay for the FC weights. suggested values: 5e-4 for alexnet"
	                                         ", 1e-2 for ResNets", type=float)
	parser.add_argument("save_load", type=lambda x: (str(x).lower() in ['true','1', 'yes'])
	                    , help="1: save/load model for training/predict 0: dont save/load")
	parser.add_argument("-s", "--save_name", dest='save_name', default='DEBUG', help='save name of model training/testing')
	parser.add_argument("-n", "--num_proc", dest='num_proc', default=8, help="set number of processors", type=int)
	parser.add_argument("-b", "--batch_size", dest='batch_size', default=64, help="set size of batch", type=int)
	parser.add_argument("-r", "--lr", dest='lr', default=1e-4, help="learning rate", type=float)
	parser.add_argument("-e", "--num_epoch", dest='num_epoch', default=1000, type=int)
	parser.add_argument("-k", "--keep_prob", dest='keep_prob', default=1.0, help="keep probability. suggested to be set to 1.0", type=float)
	parser.add_argument("-l", "--log_rate", dest='log_rate', default=10, type=int)
	parser.add_argument("-t", "--tolerance", dest='TOL', default=1e-4, type=float)
	parser.add_argument("-v", "--verbose", dest='verbose', action="store_true", help="verbosity")
	args = parser.parse_args()
	main(**vars(args))



	"""
	#  train, mdl_code, weight_decay, batch_size=64, lr=1e-4, num_epoch=1000, log_rate=10, TOL=1e-4, verbose=True
	#          , save_load=False
	# main(True, '1', weight_decay=1e-6, batch_size=128, lr=1e-1, num_epoch=1000, log_rate=10, TOL=1e-4, verbose=False,
	# 		     save_load=False, save_name="_ARCHITECTURE_CHECK_DEBUG", num_procs=4)
	
	# main(True, '1', weight_decay=1e-6, batch_size=128, lr=1e-1, num_epoch=1000, log_rate=10, TOL=1e-4, verbose=False,
	# 	     save_load=False, save_name="_ARCHITECTURE_CHECK_DEBUG", num_procs=4)
	
	# main(False, '1', weight_decay=1./512, batch_size=64, num_epoch=1000, log_rate=10, TOL=1e-4, verbose=False, save_load=False)
	# main('0', verbose=False)

	"""

	"""
	:::::::::::::DATASET TESTING::::::::::::
	
	
	DEBUG HANDLE CHECK
	with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		train_val_string = sess.run(train_val_iterator.string_handle())
		sess.run(train_iterator)
		test_train_path, test_train_acts, test_train_labels = sess.run([next_element[0], next_element[1], next_element[2]], feed_dict={handle: train_val_string})
		# data = sess.run([next_element], feed_dict={handle: train_val_string})
		print(f'test_train_path {test_train_path}, test_train_acts {test_train_acts}, test_train_labels {test_train_labels}')

		
	DEBUG CHECK FOR SINGLE FILES ACTIVATIONS
	
	train_idx, train_labels = list(train_idx)[0], list(train_labels)[0]
	val_idx, val_labels = list(val_idx)[0], list(val_labels)[0]
	train_group_paths, val_group_paths = train_group_paths[0], val_group_paths[0]
		
		
	
	from models import HACS_ACTION_CLASSES as actions
	from utils.mdl_utils import get_tain_test_groups

	mdl_name = 'resnet18'
	mdl_id = 'resnet18'
	max_vid_count = 25
	frames_block_cnt = 60

	result_caching = '/braintree/home/fksato/.result_caching/'
	model_act = 'model_tools.activations.core.VideoActivationsExtractorHelper._from_paths_stored/'
	result_cache_dir = f'{result_caching}{model_act}'

	group_paths = [f'{result_cache_dir}identifier={mdl_id},stimuli_identifier={mdl_name}' \
		               f'_full_HACS-200_group_{i}.pkl' for i in range(81)]

	test_size = .1
	validation_size = .1
	vid_dir = '/braintree/home/fksato/HACS_total/training'
	train_idx, test_idx, val_idx, train_labels, test_labels, val_labels \
		= get_tain_test_groups(actions, validation_size, test_size, max_vid_count, vid_dir)

	indices = [[frames_block_cnt * idx + i for idx in index for i in range(frames_block_cnt)] for index in
	           train_idx.values()]
	labels = [[lab for lab in label for _ in range(frames_block_cnt)] for label in train_labels.values()]

	graph = tf.Graph()
	with graph.as_default():
		test_ds = activations_dataset(group_paths, indices, labels, batch_size=81, train=True, num_procs=2)
		iterator = test_ds.make_one_shot_iterator()
		next_elem = iterator.get_next()

	with tf.Session(graph=graph) as sess:
		data = sess.run(next_elem)
		print('check')
		data = sess.run(next_elem)
		print('check')
		data = sess.run(next_elem)
		print('check')
		
		
	# OLD CODE
	
	batch_size = ceil(batch_size/frames_per_video) * frames_per_video
	input_dataset = grouped_ds.interleave(lambda path, idx, labels: tf_load_act_from_group(path, idx, labels)
	                                      , cycle_length=num_procs
	                                      , block_length=batch_size
	                                      , num_parallel_calls=num_procs)
	input_dataset = grouped_ds.flat_map(lambda path, idx, labels: tf_load_act_from_group(path, idx, labels))
	
	# OLD SHUFFLE STRATEGY

	def _shuffle_group_idx(group_indices, group_labels):
		np.random.seed(1234)
		from copy import deepcopy
		z_data = np.squeeze([[list(zip(index, label)) for index, label in zip(group_indices, group_labels)]])
		pre_shuffle = deepcopy(z_data)
		[np.random.shuffle(x) for x in z_data]
		check = [set(post) == set(pre) for pre, post in zip(pre_shuffle, z_data)]
		assert len(check) == z_data.shape[0]
		assert any(check)
		del pre_shuffle
		s_index, s_labels = np.array([list(zip(*zipped))[0] for zipped in z_data]), np.array(
			[list(zip(*zipped))[1] for zipped in z_data])
		assert s_index.shape == s_labels.shape
		return s_index, s_labels
		
		
	
	val_idx = [[frames_block_cnt * idx + i for idx in index for i in range(frames_block_cnt)] for index in
	           val_idx.values()]

	val_labels = [[lab for lab in label for _ in range(frames_block_cnt)] for label in val_labels.values()]

	shuffle inices per group: (shuffle by videos)
	np.random.seed(123)
	s_train_idx, s_train_labels = _shuffle_group_idx(train_idx.values(), train_labels.values())

	s_train_idx = [[frames_block_cnt * idx + i for idx in index for i in range(frames_block_cnt)] for index in
	           s_train_idx]
	s_train_labels = [[lab for lab in label for _ in range(frames_block_cnt)] for label in s_train_labels]

	shuffle inices per group: (shuffle by videos)
	np.random.seed(123)
	s_train_idx = [[frames_block_cnt * idx + i for idx in index for i in range(frames_block_cnt)] for index in
			           s_train_idx]
			s_train_labels = [[lab for lab in label for _ in range(frames_block_cnt)] for label in s_train_labels]
	s_train_idx, s_train_labels = _shuffle_group_idx(indices, labels)
	train_cnt_batch = ceil(train_cnt / batch_size) * batch_size
	val_cnt_batch = ceil(val_cnt / batch_size) * batch_size
	
	
	
	
	test_cnt_batch = ceil(test_cnt / batch_size) * batch_size
	
	test_idx = [[frames_block_cnt * idx + i for idx in index for i in range(frames_block_cnt)] for index in
	           test_idx.values()]

	test_labels = [[lab for lab in label for _ in range(frames_block_cnt)] for label in test_labels.values()]
		
	"""


