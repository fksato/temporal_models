import tensorflow as tf
import pickle as pk
import numpy as np

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def read_pickle_from_file(filename):
	with tf.gfile.Open(filename, 'rb') as f:
		data_dict = pk.load(f)
	return data_dict['data']


def read_from_tfrecord(filenames):
	tfrecord_file_queue = tf.train.string_input_producer(filenames, name='queue')
	reader = tf.TFRecordReader()
	_, tfrecord_serialized = reader.read(tfrecord_file_queue)

	# label and image are stored as bytes but could be stored as
	# int64 or float64 values in a serialized tf.Example protobuf.
	tfrecord_features = tf.parse_single_example(tfrecord_serialized,
						features={
							'stimulus_path': tf.FixedLenFeature([], tf.string),
							'activations': tf.FixedLenFeature([], tf.float32)
						}, name='features')

	activations = tfrecord_features['activations']
	stimulus_path = tfrecord_features['stimulus_path']
	return stimulus_path, activations


def convert_to_tfrecord(input_files, output_file):
	print('Generating %s' % output_file)
	options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
	with tf.python_io.TFRecordWriter(output_file, options=options) as record_writer:
		for input_file in input_files:
			data = read_pickle_from_file(input_file)
			stimulus_paths = data['stimulus_path'].values
			stimulus_paths = [str.encode(i) for i in stimulus_paths]
			num_entries_in_batch = len(stimulus_paths)
			data_activations = [np.array(i) for i in data.values]
		for i in range(num_entries_in_batch):
			example = tf.train.Example(features=tf.train.Features(
			    feature={
				    'stimulus_path': _bytes_feature(stimulus_paths[i])
				    , 'activations': _float_feature(data_activations[i])
			    }))
			record_writer.write(example.SerializeToString())


def extract_fn(data_record, feature_size):
	features = {
		# Extract features using the keys set during creation
		'stimulus_path': tf.FixedLenFeature([], tf.string),
		'activations': tf.FixedLenFeature([feature_size], tf.float32)
	}
	sample = tf.parse_single_example(data_record, features)
	return sample


if __name__=='__main__':
	# import tqdm
	from utils.mdl_utils import get_tain_test_groups
	from models import HACS_ACTION_CLASSES as actions
	from logistic_regression.logistic_train_predict import check_valid_indices, activations_dataset, LogRegModel
	import time

	from utils.mdl_utils import mask_unused_gpus
	import pickle as pk

	mask_unused_gpus()
	gpu_options = tf.GPUOptions(allow_growth=True)

	# mdl_name = 'alexnet'
	# mdl_id = 'alexnet'
	#
	# data_dir = '/braintree/home/fksato/.result_caching/model_tools.activations.core.VideoActivationsExtractorHelper._from_paths_stored/'

	# in_files = [f'{data_dir}identifier={mdl_id},stimuli_identifier={mdl_name}_full_HACS-200_group_{i}.pkl'
	#                for i in range(1)]
	out_file = '/braintree/home/fksato/temp/group_0_1_compression_test.tfrecords'
	# in_files = [f'{data_dir}/{activations}' for activations in act_fname]

	# print('Running conversion')
	# start = time.clock()
	# convert_to_tfrecord(in_files, out_file)
	# perf = time.clock() - start
	# print(f'Performance on conversion from pickle files: {perf}')
	# print(f'9.8GB total conversion: efficiency = {9.8 / perf}GB/s')

	# act_fname = 'identifier=alexnet,stimuli_identifier=alexnet_full_HACS-200_group_0.pkl'
	# with open(f'{data_dir}/{act_fname}', 'rb') as f:
	# 	test = pk.load(f)['data']
	# test_activations = test[0].values

	graph = tf.Graph()
	feature_size = 4096
	batch_size = 32

	# with graph.as_default():
	# 	dataset = tf.data.TFRecordDataset([out_file], compression_type='GZIP')
	# 	dataset = dataset.map(lambda x: extract_fn(x, feature_size)).batch(batch_size)
	# 	iterator = dataset.make_one_shot_iterator()
	# 	next_element = iterator.get_next()
	#
	# print('Running conversion')
	# it = 0
	# start = time.clock()
	# with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	# 	while True:
	# 		try:
	# 			check = sess.run([next_element])
	# 			it += batch_size
	# 		except tf.errors.OutOfRangeError:
	# 			break
	#
	# perf = time.clock() - start
	# print(f'total time: {perf}, Num iter = {it}: efficiency = {it/perf:.4}it/s')

	# print(check)
	# print('check')

	# mdl_name = 'resnet18'
	# mdl_id = 'resnet18'
	# frames_block_cnt = 60
	# feature_size = 512
	# num_class = 200
	# batch_size = 64
	# num_epoch = 2
	# log_rate = 1
	# TOL = 1e-4
	#
	result_caching = '/braintree/home/fksato/.result_caching/'
	model_act = 'model_tools.activations.core.VideoActivationsExtractorHelper._from_paths_stored/'
	result_cache_dir = f'{result_caching}{model_act}'

	mdl_name = 'alexnet'
	mdl_id = 'alexnet'
	frames_block_cnt = 60

	group_paths = [f'{result_cache_dir}identifier={mdl_id},stimuli_identifier={mdl_name}_full_HACS-200_group_{i}.pkl'
	               for i in range(1)]
	#
	test_size = 0
	validation_size = 0
	max_vid_count = 25
	vid_dir = '/braintree/home/fksato/HACS_total/training'
	train_idx, test_idx, val_idx, train_labels, test_labels, val_labels = get_tain_test_groups(actions
	                                                                                           , validation_size
	                                                                                           , test_size,
	                                                                                           max_vid_count
	                                                                                           , vid_dir)
	#
	# check valid indx:
	val_group_paths, val_idx, val_labels = check_valid_indices(group_paths, val_idx, val_labels)
	test_group_paths, test_idx, test_labels = check_valid_indices(group_paths, test_idx, test_labels)
	invalid_group = [group for group in val_idx.keys() if len(val_idx[group]) == 0]

	train_cnt = sum([len(train_idx[group]) for group in train_idx.keys()])
	val_cnt = sum([len(train_idx[group]) for group in val_idx.keys()])
	test_cnt = sum([len(train_idx[group]) for group in test_idx.keys()])

	if len(invalid_group) > 0:
		test_group_paths = [i for ix, i in enumerate(group_paths) if ix not in invalid_group]
		[(test_idx.pop(i), test_labels.pop(i)) for i in invalid_group]
	else:
		test_group_paths = group_paths

	activations_cnt = train_cnt * frames_block_cnt
	weight_decay = 1. / (1e-3 * activations_cnt)

	with graph.as_default():
		train_dataset = activations_dataset(group_paths, train_idx, train_labels, frames_block_cnt, batch_size
		                                    , train=True)
		# val_dataset = activations_dataset(val_group_paths, val_idx, val_labels, frames_block_cnt, batch_size
		#                                   , train=False)
		# test_dataset = activations_dataset(test_group_paths, test_idx, test_labels, frames_block_cnt, batch_size
		#                                    , train=False)

		# Feedable iterator assigns each iterator a unique string handle it is going to work on
		# handle = tf.placeholder(tf.string, shape=[])
		# iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
		train_iterator = train_dataset.make_one_shot_iterator()
		next_element = train_iterator.get_next()

		# train_val_iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
		# train_iterator = train_val_iterator.make_initializer(train_dataset)
		# val_iterator = train_val_iterator.make_initializer(val_dataset)

		# testing_iterator = test_dataset.make_one_shot_iterator()

		# model = LogRegModel(feature_size, num_class, next_element[1], next_element[2], weight_decay=weight_decay)

	print('Running iter')
	it = 0
	start = time.clock()
	with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		while True:
			try:
				check = sess.run([next_element])
				it += batch_size
			except tf.errors.OutOfRangeError:
				break

	perf = time.clock() - start
	print(f'total time: {perf}, Num iter = {it}: efficiency = {it/perf:.4}it/s')

	# start = time.clock()
	# with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	# 	sess.run([tf.global_variables_initializer()])
	# 	train_val_string = sess.run(train_val_iterator.string_handle())
	# 	testing_string = sess.run(testing_iterator.string_handle())
	#
	# 	# model.saver.restore(sess, f'/braintree/home/fksato/Projects/models/log_regression/check_points/alexnet-20.ckpt')
	# 	for step in tqdm.tqdm(range(num_epoch), unit_scale=1, desc="Epoch Training"):
	# 		sess.run(train_iterator)
	# 		for minibatch_train in tqdm.tqdm(range(0, activations_cnt, batch_size), unit_scale=batch_size
	# 				, desc="Traininng"):
	# 			try:
	# 				_, train_err_loss, train_reg_loss = sess.run(
	# 					[model.train_op, model.classification_error, model.reg_loss]
	# 					, feed_dict={handle: train_val_string})
	# 			except tf.errors.OutOfRangeError:
	# 				break
	#
	# 		if train_err_loss < TOL:
	# 			print(f'Converged with train error loss: {train_err_loss} reg_loss: {train_reg_loss}\n')
	# 			break
	#
	# 		if step % log_rate == 0:
	# 			sess.run(val_iterator)
	# 			for minibatch_val in tqdm.tqdm(range(0, val_cnt, batch_size), unit_scale=batch_size
	# 					, desc="Validation"):
	# 				try:
	# 					while True:
	# 						val_loss, val_reg_loss = sess.run([model.classification_error, model.reg_loss]
	# 						                                  , feed_dict={handle: train_val_string})
	# 				except tf.errors.OutOfRangeError:
	# 					pass
	# 			print(f'\nEpoch: {step + 1}')
	# 			print(f'Training error: {train_err_loss:.4f}, reg loss: {train_reg_loss:.4f}')
	# 			print(f'Val error: {val_loss:.4f}, reg loss: {val_reg_loss:.4f}\n')
	# 	# 		model.saver.save(sess, f'/braintree/home/fksato/Projects/models/'
	# 	# 		f'log_regression/check_points/{mdl_name}.ckpt')
	# 	# model.saver.save(sess, f'/braintree/home/fksato/Projects/models/log_regression/check_points/'
	# 	# f'{mdl_name}_final.ckpt')
	#
	# 	for minibatch_test in tqdm.tqdm(range(0, test_cnt, batch_size), unit_scale=batch_size
	# 			, desc="Testing"):
	# 		try:
	# 			while True:
	# 				test_loss, test_reg_loss = sess.run([model.classification_error, model.reg_loss]
	# 				                                    , feed_dict={handle: testing_string})
	# 		except tf.errors.OutOfRangeError:
	# 			pass
	#
	# print(f'\nTest error: {test_loss:.4f}, loss: {test_reg_loss:.4f}')
	# perf = start - time.clock()
	# print(f'Performance on minibatched pickle files: {perf}')
	#
	# tf.reset_default_graph()
