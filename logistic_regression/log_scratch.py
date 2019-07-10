#! /usr/bin/env python
import sys
sys.path.append("..")

if __name__=='__main__':
	from logistic_regression.logistic_train_predict import activations_dataset
	from models import HACS_ACTION_CLASSES as actions
	from utils.mdl_utils import get_tain_test_groups
	from utils.mdl_utils import mask_unused_gpus
	import tensorflow as tf
	import numpy as np
	import tqdm

	mask_unused_gpus()

	gpu_options = tf.GPUOptions(allow_growth=True)

	mdl_name = 'resnet18'
	mdl_id = 'resnet18'
	frames_block_cnt = 60
	feature_size = 512
	num_class = 200
	# num_epoch = 1000
	batch_size = 64

	print(mdl_name)

	result_caching = '/braintree/home/fksato/.result_caching/'
	model_act = 'model_tools.activations.core.VideoActivationsExtractorHelper._from_paths_stored/'
	result_cache_dir = f'{result_caching}{model_act}'

	group_paths = [f'{result_cache_dir}identifier={mdl_id},stimuli_identifier={mdl_name}_full_HACS-200_group_{i}.pkl'
	               for i in range(81)]
	assert len(group_paths) == 81

	test_size = .1
	validation_size = .1
	max_vid_count = 25
	vid_dir = '/braintree/home/fksato/HACS_total/training'
	train_idx, test_idx, val_idx, train_labels, test_labels, val_labels = get_tain_test_groups(actions, validation_size, test_size, max_vid_count,
	                                                                      vid_dir)

	# check valid indx:
	invalid_group = [group for group in val_idx.keys() if len(test_idx[group]) == 0]
	cnt_test = sum([len(test_idx[group]) for group in val_idx.keys()])
	if len(invalid_group) > 0:
		test_group_paths = [i for ix, i in enumerate(group_paths) if ix not in invalid_group]
		[(test_idx.pop(i), test_labels.pop(i)) for i in invalid_group]
	else:
		test_group_paths = group_paths

	activations_cnt = cnt_test * frames_block_cnt
	#mdl:
	graph = tf.Graph()

	with graph.as_default():
		test_dataset = activations_dataset(test_group_paths, test_idx, test_labels, frames_block_cnt, batch_size, train=False)
		testing_iterator = test_dataset.make_one_shot_iterator()
		next_element = testing_iterator.get_next()

		weights = tf.get_variable('W', [feature_size, num_class]
		                          , initializer=tf.variance_scaling_initializer(distribution="uniform"))
		biases = tf.get_variable('b', [num_class]
		                          , initializer=tf.variance_scaling_initializer(distribution="uniform"))

		logits = tf.add(tf.matmul(next_element[0], weights), biases)
		# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=training_labels, logits=logits))
		#
		# opt = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

		# train_prediction = tf.nn.softmax(logits)
		# valid_prediction = tf.nn.softmax(tf.add(tf.matmul(val_features, weights), biases))
		test_prediction = tf.nn.softmax(tf.add(tf.matmul(next_element[0], weights), biases))

		saver = tf.train.Saver({'W': weights, 'b': biases})

	with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		saver.restore(sess, f'/braintree/home/fksato/Projects/models/log_regression/check_points/resnet18-80.ckpt')
		sess.run([tf.global_variables_initializer()])
		preds = []
		for minibatch_train in tqdm.tqdm(range(0, activations_cnt, batch_size), unit_scale=batch_size
				, desc="fit_logistic"):
			try:
				data, predictions = sess.run([next_element, tf.nn.softmax(test_prediction)])
				preds.append((np.squeeze(predictions), data[1]))
			except tf.errors.OutOfRangeError:
				break
		proba = np.array(preds)
		np.save('/braintree/home/fksato/Projects/models/log_regression/resnet18_probs', proba)