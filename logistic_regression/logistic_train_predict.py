import sys
sys.path.append("..")

import xarray as xr

import tensorflow as tf
import pickle as pk
import numpy as np

import tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def load_act_from_group(group_path, frame_block_cnts, shuffle=False):
	# np.random.seed(1234)
	with open(group_path, 'rb') as f:
		activations = pk.load(f)
		try:
			activations = activations['data']
		except KeyError:
			pass

	num_vids = int(activations.shape[0] / frame_block_cnts)
	act_idx = [i for i in range(num_vids)]
	if shuffle:
		np.random.shuffle(act_idx)

	act_index = [frame_block_cnts * index + i for index in act_idx for i in range(frame_block_cnts)]

	group_activations = activations[act_index].values.astype(np.float32)
	group_stim_paths = activations[act_index].vid_path.values
	group_labels = activations[act_index].action_label.values.astype(np.int32)

	assert group_labels.shape[0] == group_activations.shape[0] == group_stim_paths.shape[0]

	return group_stim_paths, group_activations, group_labels


def tf_load_act_from_group(group_path, frame_block_cnts, shuffe=False):

	def loader(group_path, frame_block_cnts, shuffe=False):
		stim_paths, act, action_labels = tf.py_func(load_act_from_group
		                                            , [group_path, frame_block_cnts, shuffe]
		                                            , [tf.string, tf.float32, tf.int32])
		return tf.data.Dataset.from_tensor_slices(stim_paths), tf.data.Dataset.from_tensor_slices(act)\
			, tf.data.Dataset.from_tensor_slices(action_labels)

	return tf.data.Dataset.zip(loader(group_path, frame_block_cnts, shuffe))


def activations_dataset(group_paths, frame_block_cnts, batch_size, train=True, num_procs=8):
	if not isinstance(group_paths, list):
		input_dataset = tf_load_act_from_group(group_paths, frame_block_cnts, shuffe=train)
	else:
		input_dataset = tf.data.Dataset.from_tensor_slices(group_paths)

		# shuffle groups
		if train:
			input_dataset = input_dataset.shuffle(buffer_size=len(group_paths), seed=1234)

		input_dataset = input_dataset.interleave(lambda path:
		                                      tf_load_act_from_group(path, frame_block_cnts=frame_block_cnts, shuffe=train)
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
	                            , 'label' : ('stimuli', labels)
	                            , 'pred' : ('stimuli', np.argmax(predictions, axis=1))}
	                  , dims=['stimuli', 'predictions'])
	return ds


class LogRegModel:
	def __init__(self, features, num_class, data_X, data_y, lr=1e-4, weight_decay=None
	             , activation=None, make_summary=False):
		self.feature_size = features
		self.num_class = num_class
		self.weight_decay = weight_decay or 0.5
		self.fc_dropout_keep_prob = tf.placeholder(tf.float32)
		self.lr = lr
		self._opt = tf.train.AdamOptimizer(learning_rate=self.lr)
		self.build(data_X, data_y, activation, make_summary)

	def build(self, inputs, labels, activation, make_summary=False):
		self._fc_layer(inputs, activation)
		self._make_loss(labels)
		if make_summary:
			# if self.log_summary:
			with tf.name_scope('wieghts_summary'):
				mean_w = tf.reduce_mean(self.weights)
				std = tf.reduce_mean(tf.square(self.weights - mean_w))
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
				# if self.log_summary:
			with tf.name_scope('training_summary'):
				tf.summary.scalar('classification_error', self.classification_error)
				tf.summary.scalar('regularization_loss', self.reg_loss)
				tf.summary.scalar('accuracy', self.acc)
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


def fit(graph, model, handle, train_iterator, val_iterator, train_val_iterator, activations_cnt, val_cnt, num_epoch
        , batch_size, mdl_name, keep_prob=1.0, TOL=1e-4, log_rate=10, gpu_options=None, checkpoint_save=False, checkpoint_load=None, start_epoc=0):

	train_writer = tf.summary.FileWriter(f'/braintree/home/fksato/Projects/models/log_regression/model_summaries'
	                                     f'/{mdl_name}_train', graph)
	val_writer = tf.summary.FileWriter(f'/braintree/home/fksato/Projects/models/log_regression/model_summaries'
	                                     f'/{mdl_name}_val')

	with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		sess.run(tf.global_variables_initializer())
		if checkpoint_load is not None:
			model.saver.restore(sess, checkpoint_load)
		train_val_string = sess.run(train_val_iterator.string_handle())

		for step in tqdm.tqdm(range(start_epoc, num_epoch), unit_scale=1, desc="Epoch Training"):
			sess.run(train_iterator)
			sess.run(tf.local_variables_initializer())
			for minibatch_train in tqdm.tqdm(range(0, activations_cnt, batch_size), unit_scale=batch_size
					, desc="Training"):
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


def predict(graph, model, mdl_load_name, iter_get_next, test_cnt, batch_size, save_load=False, checkpoint_name="final", gpu_options=None):

	preds = np.zeros((test_cnt, 200), dtype=np.float)
	stim_paths = []
	labels = []
	with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		sess.run(tf.global_variables_initializer())
		if save_load:
			model.saver.restore(sess, f'/braintree/home/fksato/Projects/models/log_regression/check_points/'
				f'{mdl_load_name}_{checkpoint_name}.ckpt')
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


def confirm():
	while True:
		ans = input("confirm (y/n)?")
		if ans not in ['y', 'Y', 'n', 'N']:
			print('please enter y or n.')
			continue
		if ans == 'y' or ans == 'Y':
			return True
		if ans == 'n' or ans == 'N':
			return False


def main(train, mdl_code, weight_decay, batch_size=64, lr=1e-4, num_epoch=1000, keep_prob=1.0, log_rate=10, TOL=1e-4, verbose=True
         , save_load=False, save_name='DEBUG', num_procs=8, checkpoint_name='', checkpoint_load=None, start_epoch=0, *args, **kwargs):

	from utils.mdl_utils import mask_unused_gpus
	from math import ceil
	from glob import glob

	model_names = {'0': 'alexnet', '1': 'resnet18', '2': 'resnet34'
		, '3': 'r25D_d18_l8_8_CLIPS_NONE', '4': 'r25D_d18_l16_16_CLIPS_NONE'
		, '5': 'r25D_d34_l32_32_CLIPS_NONE_ft_ig65m', '6': 'r25D_d18_l16_60_FULL'
		, '7': 'r25D_d152_l32_32_CLIPS_NONE_ft_ig65m'}
	mdl_frames_blocks = {'0': 60, '1': 60, '2': 60, '3': 8, '4': 4, '5': 2, '6': 1, '7': 2}
	input_size = {'0': 4096, '1': 512, '2': 512, '3': 512, '4': 512, '5': 512, '6': 512, '7': 2048}
	num_class = 200

	mask_unused_gpus()
	gpu_options = tf.GPUOptions(allow_growth=True)

	mdl_name = model_names[mdl_code]
	frames_block_cnt = mdl_frames_blocks[mdl_code]
	feature_size = input_size[mdl_code]
	num_epoch = num_epoch
	batch_size = ceil(batch_size/frames_block_cnt) * frames_block_cnt # guarantee full videos are represented per batch
	log_rate = log_rate
	TOL = TOL

	# result_caching = '/braintree/home/fksato/.result_caching/'
	model_act = f'/braintree/home/fksato/Projects/models/model_data/{mdl_name}'

	print(f'::::::::LOGISTIC REGRESSION::::::::::::')
	print(f':::::::::MODEL INFORMATION:::::::::::::')
	print(f'MODE: {"train" if train else "predict"}')
	print(f'Performance evaluation on {mdl_name}\nframe/block count: {frames_block_cnt}')
	print(f'activation directory: {model_act}')
	print(f'{save_load}')
	if verbose and not confirm():
		return -1

	TRAIN_CNT = 90427 * frames_block_cnt
	TEST_CNT = 9945 * frames_block_cnt

	# mdl:
	graph = tf.Graph()
	activation = None
	make_summary = False

	with graph.as_default():
		if train:
			if checkpoint_load:
				checkpoint_load = f'/braintree/home/fksato/Projects/models/log_regression/check_points/{checkpoint_load}.ckpt'
			make_summary = True
			train_group_paths = glob(f'{model_act}/training_activations/*.pkl')
			validation_group_paths = glob(f'{model_act}/validation_activations/*.pkl')

			train_dataset = activations_dataset(train_group_paths, frames_block_cnt, batch_size, train=True
			                                    , num_procs=num_procs)

			val_dataset = activations_dataset(validation_group_paths, frames_block_cnt, batch_size, train=False
			                                  , num_procs=num_procs)

			train_val_iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
			train_iterator = train_val_iterator.make_initializer(train_dataset)
			val_iterator = train_val_iterator.make_initializer(val_dataset)

			# Feedable iterator assigns each iterator a unique string handle it is going to work on
			handle = tf.placeholder(tf.string, shape=[])
			iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types,
			                                               train_dataset.output_shapes)
		else:
			test_group_paths = glob(f'{model_act}/testing_activations/*.pkl')
			test_dataset = activations_dataset(test_group_paths, frames_block_cnt, batch_size, train=False
			                                   , num_procs=num_procs)
			activation = 'softmax'
			iterator = test_dataset.make_one_shot_iterator()

		next_element = iterator.get_next()

		model = LogRegModel(feature_size, num_class, next_element[1], next_element[2], lr=lr, weight_decay=weight_decay
		                    , activation=activation, make_summary=make_summary)

	chkpt_mdl_save_name = f'{mdl_name}_{save_name}'

	if train:
		fit(graph, model, handle, train_iterator, val_iterator, train_val_iterator, TRAIN_CNT, TEST_CNT, num_epoch
		    , batch_size, chkpt_mdl_save_name, keep_prob=keep_prob, TOL=TOL, log_rate=log_rate, gpu_options=gpu_options
		    , checkpoint_save=save_load, checkpoint_load=checkpoint_load, start_epoc=start_epoch)
	else:
		preds = predict(graph, model, chkpt_mdl_save_name, next_element, TEST_CNT, batch_size, save_load, checkpoint_name, gpu_options)
		print(f'saving: {TEST_CNT/frames_block_cnt}')
		fname = f'/braintree/home/fksato/Projects/models/log_regression/predictions/{chkpt_mdl_save_name}_{checkpoint_name}_predictions.pkl'
		with open(fname, 'wb') as f:
			pk.dump(preds, f, protocol=pk.HIGHEST_PROTOCOL)


if __name__=='__main__':
	import os
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("train", type=lambda x: (str(x).lower() in ['true','1', 'yes'])
	                    , help="train model: 1, predict: 0")
	parser.add_argument("mdl_code", help="0: alexnet, 1: resnet18, 2: resent34, 3: resnet 2.5D 18, 4: resnet 2.5D 34")
	parser.add_argument("save_load", type=lambda x: (str(x).lower() in ['true','1', 'yes'])
	                    , help="1: save/load model for training/predict 0: dont save/load")
	parser.add_argument("-w", "--weight_decay", dest='weight_decay', default=1e-6
	                    , help="weight decay for the FC weights. suggested values: 5e-6 for alexnet, 1e-6 for ResNets"
	                    , type=float)
	parser.add_argument("-s", "--save_name", dest='save_name', default='DEBUG', help='save name of model training/testing')
	parser.add_argument("-c", "--checkpoint_name", dest='checkpoint_name', default='final',
	                    help='check point name to load for model for predictions')
	parser.add_argument("-n", "--num_proc", dest='num_proc', default=8, help="set number of processors", type=int)
	parser.add_argument("-b", "--batch_size", dest='batch_size', default=64, help="set size of batch", type=int)
	parser.add_argument("-r", "--lr", dest='lr', default=1e-4, help="learning rate", type=float)
	parser.add_argument("-e", "--num_epoch", dest='num_epoch', default=1000, type=int)
	parser.add_argument("-k", "--keep_prob", dest='keep_prob', default=1.0, help="keep probability. suggested to be set to 1.0", type=float)
	parser.add_argument("-l", "--log_rate", dest='log_rate', default=10, type=int)
	parser.add_argument("-t", "--tolerance", dest='TOL', default=1e-4, type=float)
	parser.add_argument("-p", "--checkpoint_load", dest='checkpoint_load', default=None,
	                    help='check point name to load for model for restarting training. Ignore for prediction')
	parser.add_argument("-i", "--start_epoch", dest='start_epoch', default=0, type=int,
	                    help='reload epoch start')
	parser.add_argument("-v", "--verbose", dest='verbose', action="store_true", help="verbosity")
	args = parser.parse_args()
	main(**vars(args))


	"""
	:::::::::::::REGRESSION TESTING::::::::::::
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
	"""