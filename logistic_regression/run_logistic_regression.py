#! /usr/bin/env python
import sys
sys.path.append("..")

from logistic_regression.logistic_train_predict import activations_dataset
from models import HACS_TOTAL_VID_COUNT as vid_count
from models import HACS_ACTION_CLASSES as actions
from utils.mdl_utils import get_tain_test_groups
from utils.mdl_utils import mask_unused_gpus
import tensorflow as tf
import numpy as np
import tqdm

mdls = {'0': 'alexnet', '1': 'resnet18', '2': 'resnet34', '3': 'r2_1-18_l8-kinetics', '4': 'r2_1-34_l32-kinetics+sports1m'}
mdl_ids = {'0': 'alexnet', '1': 'resnet18', '2': 'resnet34', '3': '<built-in function id>', '4': '<built-in function id>'}
mdl_frames_blocks = {'0': 60, '1': 60, '2': 60, '3': 14, '4': 12}
# groupped_max_vid_cnt = {'0': 25, '1': 25, '2': 25, '3': 25, '4': 15,}
perf_mode = {0: 'training', 1: 'predict'}
input_size = {'0': 4096, '1': 512, '2': 512, '3': 512, '4': 512}
# r2_1-18_l8-kinetics: blocks   = 14
# r2_1-34_l32-kinetics+sports1m = 12

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

# class LogReg:
#
# 	def __init__(self, num_features, num_classes):
# 		self.in_shape = num_features
# 		self.out_shape = num_classes
#
# 	def fc(self):
# 		weights = tf.get_variable('W', [self.in_shape, self.out_shape]
# 		                          , initializer=tf.variance_scaling_initializer(distribution="uniform"))
# 		biases = tf.get_variable('b', [self.out_shape]
# 		                         , initializer=tf.variance_scaling_initializer(distribution="uniform"))
#
# 		logits = tf.add(tf.matmul(next_element[0], weights), biases)
# 		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=next_element[1], logits=logits))
#
# 		opt = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
#
# 		train_prediction = tf.nn.softmax(logits)




def main(mdl_code, batch_size=64, num_epoch=1000, verbose=True, *args, **kwargs):
	mask_unused_gpus()
	gpu_options = tf.GPUOptions(allow_growth=True)

	mdl_name = mdls[mdl_code]
	mdl_id = mdl_ids[mdl_code]
	frames_block_cnt = mdl_frames_blocks[mdl_code]

	feature_size = input_size[mdl_code]
	batch_size = batch_size
	num_epoch = num_epoch

	num_class = 200
	activations_cnt = vid_count * frames_block_cnt

	result_caching = '/braintree/home/fksato/.result_caching/'
	model_act = 'model_tools.activations.core.VideoActivationsExtractorHelper._from_paths_stored/'
	result_cache_dir = f'{result_caching}{model_act}'

	group_paths = [f'{result_cache_dir}identifier={mdl_id},stimuli_identifier={mdl_name}_full_HACS-200_group_{i}.pkl'
	               for i in range(81)]
	assert len(group_paths) == 81

	if verbose:
		print(f'Performance evaluation on {mdl_name}\nmodel id: {mdl_id}\nframe/block count: {frames_block_cnt}')
		print(f'activation directory: {result_cache_dir}')
		print(f'file name pattern: identifier={mdl_id},'
		      f'stimuli_identifier={mdl_name}_full_HACS-200_group_<group number>.pkl')

		if not confirm():
			return -1

	test_size = .1
	validation_size = .1
	max_vid_count = 25
	vid_dir = '/braintree/home/fksato/HACS_total/training'
	train_idx, test_idx, val_idx, train_labels, test_labels, val_labels = get_tain_test_groups(actions, validation_size,
	                                                                                           test_size, max_vid_count,
	                                                                                           vid_dir)
	# check valid indx:
	invalid_group = [group for group in val_idx.keys() if len(val_idx[group]) == 0]
	if len(invalid_group) > 0:
		val_group_paths = [i for ix, i in enumerate(group_paths) if ix not in invalid_group]
		[(val_idx.pop(i), val_labels.pop(i)) for i in invalid_group]
	else:
		val_group_paths = group_paths

	# mdl:
	graph = tf.Graph()

	with graph.as_default():
		train_dataset = activations_dataset(group_paths, train_idx, train_labels, frames_block_cnt, batch_size,
		                                    train=True)
		val_dataset = activations_dataset(val_group_paths, val_idx, val_labels, frames_block_cnt, batch_size
		                                  , train=False)

		# handle constructions. Handle allows us to feed data from different dataset by providing a parameter in feed_dict
		handle = tf.placeholder(tf.string, shape=[])
		iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types,
		                                               train_dataset.output_shapes)
		next_element = iterator.get_next()

		training_iterator = train_dataset.make_initializable_iterator()
		validation_iterator = val_dataset.make_initializable_iterator()

		weights = tf.get_variable('W', [feature_size, num_class]
		                          , initializer=tf.variance_scaling_initializer(distribution="uniform"))
		biases = tf.get_variable('b', [num_class]
		                         , initializer=tf.variance_scaling_initializer(distribution="uniform"))

		logits = tf.add(tf.matmul(next_element[0], weights), biases)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=next_element[1], logits=logits))

		opt = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

		train_prediction = tf.nn.softmax(logits)
		valid_prediction = tf.nn.softmax(tf.add(tf.matmul(next_element[0], weights), biases))

		saver = tf.train.Saver({'W': weights, 'b': biases})

	# utility function to calculate accuracy
	def accuracy(predictions, labels):
		correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
		accu = (100.0 * correctly_predicted) / len(predictions)
		return accu

	with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		sess.run([tf.global_variables_initializer()])
		training_handle = sess.run(training_iterator.string_handle())
		validation_handle = sess.run(validation_iterator.string_handle())
		for step in tqdm.tqdm(range(num_epoch), desc="group logistic"):
			sess.run(training_iterator.initializer)
			# sess.run(train_init)
			for minibatch_train in tqdm.tqdm(range(0, activations_cnt, batch_size), unit_scale=batch_size
					, desc="fit_logistic"):
				try:
					# features, labels = sess.run(next_element, feed_dict={handle: training_handle})
					a, _, l, predictions = sess.run([next_element, opt, loss, train_prediction], feed_dict={handle: training_handle})
					# _, l, predictions = sess.run([opt, loss, train_prediction])
				except tf.errors.OutOfRangeError:
					break

		# 	if step % 10 == 0:
		# 		print(f'loss at step {step}: {l}')
		# 		print(f'accuracy: {accuracy(predictions, labels):.2} ')
		# 		sess.run(validation_iterator.initializer)
		# 		acc = []
		# 		while True:
		# 			try:
		# 				# features, labels = sess.run(next_element, feed_dict={handle: validation_handle})
		# 				sess.run(next_element, feed_dict={handle: validation_handle})
		# 				predictions = sess.run([valid_prediction], feed_dict={placeholder_features: features})
		# 				acc.append(accuracy(predictions, labels))
		# 			except tf.errors.OutOfRangeError:
		# 				break
		# 		print(f'Validation accuracy: {np.mean(acc):.2}')
		# 		saver.save(sess,
		# 		           f'/braintree/home/fksato/Projects/models/log_regression/check_points/{mdl_name}-{step}.ckpt')
		#
		# saver.save(sess, f'/braintree/home/fksato/Projects/models/log_regression/check_points/{mdl_name}-FINAL.ckpt')

if __name__=='__main__':

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("mdl_code", help="0: alexnet, 1: resnet18, 2: resent34, 3: resnet 2.5D 18, 4: resnet 2.5D 34")
	parser.add_argument("-e", "--num_epoch", dest='num_epoch', default=1000, type=int)
	parser.add_argument("-b", "--batch_size", dest='batch_size', default=64, help="set size of batch")
	parser.add_argument("-v", "--verbose", dest='verbose', action="store_true", help="verbosity")
	args = parser.parse_args()
	main(**vars(args))
	# main('1')