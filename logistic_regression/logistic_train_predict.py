import sys
sys.path.append("..")

import xarray as xr

import tensorflow as tf
import pickle as pk
import numpy as np

import tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def load_act_from_group(group_path, act_index, labels, frames_block_cnt):
	with open(group_path, 'rb') as f:
		activations = pk.load(f)
		activations = activations['data']

	group_labels = np.array([label for label in labels for _ in range(frames_block_cnt)])
	idx = [frames_block_cnt * index + i for i in range(frames_block_cnt) for index in act_index]
	group_activations = activations[idx].values
	group_stim_paths = activations[idx].stimulus_path.values
	assert group_labels.shape[0] == group_activations.shape[0] == group_stim_paths.shape[0]
	return group_stim_paths, group_activations, group_labels


def tf_load_act_from_group(group_path, act_index, labels, frames_block_cnt):
	def loader(group_path, act_index, labels, frames_block_cnt):
		stim_paths, act, action_labels = tf.py_func(load_act_from_group
		                                            , [group_path, act_index, labels, frames_block_cnt]
		                                            , [tf.string, tf.float32, tf.int32])
		return tf.data.Dataset.from_tensor_slices(stim_paths), tf.data.Dataset.from_tensor_slices(act)\
			, tf.data.Dataset.from_tensor_slices(action_labels)

	return tf.data.Dataset.zip(loader(group_path, act_index, labels, frames_block_cnt))


def activations_dataset(group_paths, act_index, group_labels, frames_block_cnt, batch_size, train=True):
	indices = [idx for idx in act_index.values()]
	labels = [labs for labs in group_labels.values()]

	group_path_ds = tf.data.Dataset.from_tensor_slices(group_paths)
	act_index_ds = tf.data.Dataset.from_generator(lambda: indices, tf.int32, output_shapes=[None])
	labels_ds = tf.data.Dataset.from_generator(lambda: labels, tf.int32, output_shapes=[None])
	# labels_ds = labels_ds.map(lambda x: tf.one_hot(x, 200, dtype=tf.int32))

	input_dataset = tf.data.Dataset.zip((group_path_ds, act_index_ds, labels_ds))
	input_dataset = input_dataset. \
		flat_map(lambda x, y, z: tf_load_act_from_group(x, y, z, frames_block_cnt)).batch(batch_size=batch_size)

	if train:
		input_dataset.shuffle(buffer_size=10000, seed=1234)
	return input_dataset


def check_valid_indices(group_paths, indices, labels):
	invalid_group = [group for group in indices.keys() if len(indices[group]) == 0]
	if len(invalid_group) > 0:
		_group_paths = [i for ix, i in enumerate(group_paths) if ix not in invalid_group]
		[(indices.pop(i), labels.pop(i)) for i in invalid_group]
	else:
		_group_paths = group_paths

	return _group_paths, indices, labels


def package_predictions(paths, predictions):
	# pickle coords/dims:
	ds = xr.DataArray(predictions
	                  , coords={'stim_path': paths
	                            , 'class': ('predictions', np.arange(predictions.shape[1]))
	                            , 'probability': ('predictions', np.arange(predictions.shape[1]))}
	                  , dims=['stim_path', 'predictions'])
	ds = ds.set_index(predictions=['class', 'probability'], append=True)
	return ds


class LogRegModel:
	def __init__(self, features, num_class, data_X, data_y, lr=1e-4, fc_dropout_keep_prob=1, weight_decay=None
	             , activation=None):
		self.feature_size = features
		self.num_class = num_class
		self.n_class = num_class
		self.weight_decay = weight_decay or 0.5
		self.fc_dropout_keep_prob = fc_dropout_keep_prob
		self.lr = lr
		self._opt = tf.train.AdamOptimizer(learning_rate=self.lr)
		self.build(data_X, data_y, activation)

	def build(self, inputs, labels, activation):
		self._fc_layer(inputs, activation)
		self._make_loss(labels)

	def _fc_layer(self, inputs, activation=None):
		self.weights = tf.get_variable('W', [self.feature_size, self.num_class]
		                          , initializer=tf.contrib.layers.xavier_initializer(seed=0)
		                          , regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
		biases = tf.get_variable('b', [self.num_class]
		                         , initializer=tf.initializers.constant()
		                         , regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
		self.saver = tf.train.Saver({'W': self.weights, 'b': biases})

		out = tf.nn.dropout(inputs, keep_prob=self.fc_dropout_keep_prob, name="dropout_out")
		out = tf.add(tf.matmul(out, self.weights), biases)

		if activation is not None:
			out = getattr(tf.nn, activation)(out, name=activation)

		self.predictions = out

	def _make_loss(self, input_labels):
		with tf.variable_scope('loss'):
			logits = self.predictions

			self.classification_error = tf.reduce_mean(
				tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=input_labels))
			self.reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
			self.total_loss = tf.add(self.classification_error, self.reg_loss)
			self.tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
			self.train_op = self._opt.minimize(self.total_loss, var_list=self.tvars,
			                                   global_step=tf.train.get_or_create_global_step())



def fit(graph, model, handle, train_iterator, val_iterator, train_val_iterator, activations_cnt, val_cnt, num_epoch
        , batch_size, mdl_name, TOL=1e-4, log_rate=10, gpu_options=None, checkpoint_save=False):
	with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		sess.run([tf.global_variables_initializer()])
		train_val_string = sess.run(train_val_iterator.string_handle())

		for step in tqdm.tqdm(range(num_epoch), unit_scale=1, desc="Epoch Training"):
			sess.run(train_iterator)
			for minibatch_train in tqdm.tqdm(range(0, activations_cnt, batch_size), unit_scale=batch_size
					, desc="Traininng"):
				try:
					_, train_err_loss, train_reg_loss = sess.run(
						[model.train_op, model.classification_error, model.reg_loss]
						, feed_dict={handle: train_val_string})
				except tf.errors.OutOfRangeError:
					break

			if train_err_loss < TOL:
				print(f'Converged with train error loss: {train_err_loss} reg_loss: {train_reg_loss}\n')
				break

			if step % log_rate == 0:
				sess.run(val_iterator)
				for minibatch_val in tqdm.tqdm(range(0, val_cnt, batch_size), unit_scale=batch_size
						, desc="Validation"):
					try:
						val_loss, val_reg_loss = sess.run([model.classification_error, model.reg_loss]
						                                  , feed_dict={handle: train_val_string})
					except tf.errors.OutOfRangeError:
						pass
				print(f'\nEpoch: {step + 1}')
				print(f'Training loss: {train_err_loss:.4f}, regularization loss: {train_reg_loss:.4f}')
				print(f'Validation loss: {val_loss:.4f}, regularization loss: {val_reg_loss:.4f}\n')
				if checkpoint_save:
					model.saver.save(sess, f'/braintree/home/fksato/Projects/models/'
					f'log_regression/check_points/{mdl_name}-{model.lr}-t-test.ckpt')
		if checkpoint_save:
			model.saver.save(sess, f'/braintree/home/fksato/Projects/models/log_regression/check_points/'
			f'{mdl_name}_final-t-test.ckpt')


def predict(graph, model, mdl_name, iter_get_next, test_cnt, batch_size, save_load=False, gpu_options=None):
	from brainio_base.assemblies import merge_data_arrays
	preds=None
	with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		if save_load:
			model.saver.restore(sess, f'/braintree/home/fksato/Projects/models/log_regression/check_points/'
				f'{mdl_name}-{model.lr}-t-test.ckpt')

		sess.run([tf.global_variables_initializer()])
		for minibatch_test in tqdm.tqdm(range(0, test_cnt, batch_size), unit_scale=batch_size
				, desc="Testing"):
			try:
				frame_paths, labels, predictions = sess.run([iter_get_next[0], iter_get_next[2], model.predictions])
				if preds is None:
					preds = package_predictions(frame_paths, predictions)
				else:
					preds = merge_data_arrays([preds, package_predictions(frame_paths, predictions)])
			except tf.errors.OutOfRangeError:
				pass
	return preds


def main(train, mdl_code, batch_size=64, lr=1e-4, num_epoch=1000, log_rate=10, TOL=1e-4, verbose=True, save_load=False
         , *args, **kwargs):
	from models import HACS_ACTION_CLASSES as actions
	from utils.mdl_utils import get_tain_test_groups
	from logistic_regression.run_logistic_regression import mdls, mdl_ids, mdl_frames_blocks, input_size, confirm
	from utils.mdl_utils import mask_unused_gpus

	mask_unused_gpus()
	gpu_options = tf.GPUOptions(allow_growth=True)

	mdl_name = mdls[mdl_code]
	mdl_id = mdl_ids[mdl_code]
	frames_block_cnt = mdl_frames_blocks[mdl_code]
	feature_size = input_size[mdl_code]
	max_vid_count = 25

	num_epoch = num_epoch
	batch_size = batch_size
	log_rate = log_rate
	TOL = TOL

	num_class = 200

	result_caching = '/braintree/home/fksato/.result_caching/'
	model_act = 'model_tools.activations.core.VideoActivationsExtractorHelper._from_paths_stored/'
	result_cache_dir = f'{result_caching}{model_act}'

	group_paths = [f'{result_cache_dir}identifier={mdl_id},stimuli_identifier={mdl_name}_full_HACS-200_group_{i}.pkl'
	               for i in range(81)]
	if verbose:
		print(f'::::::::LOGISTIC REGRESSION::::::::::::')
		print(f':::::::::MODEL INFORMATION:::::::::::::')
		print(f'MODE: {"train" if train else "predict"}')
		print(f'Performance evaluation on {mdl_name}\nmodel id: {mdl_id}\nframe/block count: {frames_block_cnt}')
		print(f'activation directory: {result_cache_dir}')
		print(f'file name pattern: identifier={mdl_id},'
		      f'stimuli_identifier={mdl_name}_full_HACS-200_group_<group number>.pkl')

		if not confirm():
			return -1

	test_size = .1
	validation_size = .1
	# max_vid_count = 25
	vid_dir = '/braintree/home/fksato/HACS_total/training'
	train_idx, test_idx, val_idx, train_labels, test_labels, val_labels \
		= get_tain_test_groups(actions, validation_size, test_size, max_vid_count, vid_dir)

	if train:
		# check valid indx:
		val_group_paths, val_idx, val_labels = check_valid_indices(group_paths, val_idx, val_labels)
		train_cnt = sum([len(train_idx[group]) for group in train_idx.keys()])
		train_cnt = train_cnt * frames_block_cnt
		val_cnt = sum([len(val_idx[group]) for group in val_idx.keys()]) * frames_block_cnt
		weight_decay = 1. / (1e-3 * train_cnt)
	else:
		test_group_paths, test_idx, test_labels = check_valid_indices(group_paths, test_idx, test_labels)
		test_cnt = sum([len(test_idx[group]) for group in test_idx.keys()]) * frames_block_cnt
		weight_decay = None

	# mdl:
	graph = tf.Graph()
	activation = None

	with graph.as_default():
		if train:
			train_dataset = activations_dataset(group_paths, train_idx, train_labels, frames_block_cnt, batch_size
			                                    , train=True)
			val_dataset = activations_dataset(val_group_paths, val_idx, val_labels, frames_block_cnt, batch_size
			                                  , train=False)

			train_val_iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
			train_iterator = train_val_iterator.make_initializer(train_dataset)
			val_iterator = train_val_iterator.make_initializer(val_dataset)

			# Feedable iterator assigns each iterator a unique string handle it is going to work on
			handle = tf.placeholder(tf.string, shape=[])
			iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types,
			                                               train_dataset.output_shapes)
		else:
			test_dataset = activations_dataset(test_group_paths, test_idx, test_labels, frames_block_cnt, batch_size
			                                   , train=False)
			activation = 'softmax'

			iterator = test_dataset.make_one_shot_iterator()

		next_element = iterator.get_next()

		model = LogRegModel(feature_size, num_class, next_element[1], next_element[2], lr=lr, weight_decay=weight_decay
		                    , activation=activation)

	if train:
		fit(graph, model, handle, train_iterator, val_iterator, train_val_iterator, train_cnt, val_cnt, num_epoch
		    , batch_size, mdl_name, TOL=TOL, log_rate=log_rate, gpu_options=gpu_options, checkpoint_save=save_load)
	else:
		preds = predict(graph, model, mdl_name, next_element, test_cnt, batch_size, save_load, gpu_options)
		print(f'saving: {test_cnt/frames_block_cnt}')
		# reset_index for netcdf serialization:
		preds = preds.reset_index(['stim_path', 'predictions'])
		preds.to_netcdf(f'/braintree/home/fksato/temp/{mdl_name}_with_labels.nc')


if __name__=='__main__':
	#  train, mdl_code, batch_size=64, lr=1e-4, num_epoch=1000, log_rate=10, TOL=1e-4, verbose=True, save_load=False
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("train", type=lambda x: (str(x).lower() in ['true','1', 'yes'])
	                    , help="train model: 1, predict: 0")
	parser.add_argument("mdl_code", help="0: alexnet, 1: resnet18, 2: resent34, 3: resnet 2.5D 18, 4: resnet 2.5D 34")
	parser.add_argument("save_load", type=lambda x: (str(x).lower() in ['true','1', 'yes'])
	                    , help="1: save/load model for training/predict 0: dont save/load")
	parser.add_argument("-b", "--batch_size", dest='batch_size', default=64, help="set size of batch")
	parser.add_argument("-r", "--lr", dest='lr', default=1e-4, help="learning rate", type=float)
	parser.add_argument("-e", "--num_epoch", dest='num_epoch', default=1000, type=int)
	parser.add_argument("-l", "--log_rate", dest='log_rate', default=10, type=int)
	parser.add_argument("-t", "--tolerance", dest='TOL', default=1e-4, type=float)
	parser.add_argument("-v", "--verbose", dest='verbose', action="store_true", help="verbosity")
	args = parser.parse_args()
	main(**vars(args))

	# main(False, '3', 64, 1000, log_rate=10, TOL=1e-3, verbose=False, save_load=False)
	# main('0', verbose=False)

