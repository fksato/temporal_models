import cv2
import numpy as np
import tensorflow as tf

from data_inputs import DataInput


def video_to_frames(src_vid):
	"""
	TODO: Redundancy with PyTorch, needed since src_vid needs to be decoded from byte to string for cv2 to work
	TODO: for 4d models its not necessary to vary frame_count since packaging occurs elsewhere
	Takes a video path, parse into frames by frame_count and return array of images
	:param src_vid: path to video
	:return: array of frames
	"""
	# check type of src_vid
	cap = cv2.VideoCapture(bytes(src_vid).decode("utf-8"))
	if not cap.isOpened():
		raise Exception(f'{bytes(src_vid).decode("utf-8")} file cannot be opened')

	frame_count = 60
	width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	frames_array = np.zeros(shape=(frame_count, height, width, 3), dtype=np.float32)

	cap.set(1, 15)
	f_cnt = 0

	while cap.isOpened() and f_cnt < frame_count:
		success, buff = cap.read()
		if success:
			frames_array[f_cnt] = buff.astype(np.float32)
			f_cnt += 1

	cap.release()
	return height, width, frames_array


def stack_im_blocks(path, t_window, block_starts, frames=60, image_size=[112, 112], image_preprocess=None):
	"""
	stack video frames into blocks of images. 4D models expects input shape = [None, t,h,w,c] == im_blocks
	:param path: path to video
	:param t_window: number of consecutive frames
	:param block_starts: starting index for each block
	:param frames: frames per video
	:param image_size: resize image size for each frame
	:param image_preprocess: image preprocess
	:return: stacked image tensor
	"""

	# tf function wrapper for video_to_frames()
	def tf_video_to_frames(src_vid, frames):
		im_h, im_w, out_put = tf.py_func(video_to_frames, [src_vid], [tf.int64, tf.int64, tf.float32])
		return tf.reshape(out_put, [frames, im_h, im_w, 3])
	# call to tf function wrapper
	frames_array = tf_video_to_frames(path, frames)

	block_idx = [[i+j for j in range(t_window)] for i in block_starts]
	blocks_tensor = None
	for i in block_idx:
		if t_window > 1:
			im_block = tf.expand_dims(tf.stack(tf.map_fn(image_preprocess, tf.gather(frames_array, i, axis=0))), axis=0)
		else:
			im_block = tf.stack(tf.map_fn(image_preprocess, tf.gather(frames_array, i, axis=0)))
		if blocks_tensor is not None:
			blocks_tensor = tf.concat([blocks_tensor, im_block], axis=0)
		else:
			blocks_tensor = im_block
	return tf.data.Dataset.from_tensor_slices(blocks_tensor)


def block_stride_starts(frames_per_video, t_window):
	"""
	check if the parameters for the number of consecutive frames per block, given stride is allowable
	:param frames_per_video: number of frames per video
	:param t_window: number of consecutive frames to group into blocks
	:return: starts for frame blocks
	"""
	depth_block_check = frames_per_video % t_window
	block_stride = t_window

	if depth_block_check != 0:
		block_stride = t_window - depth_block_check
		if (frames_per_video - t_window) % block_stride !=0:
			block_stride = 1
	starts = [i for i in range(0, (frames_per_video + 1 - t_window), block_stride)]

	return starts


def video_data_inputs(video_paths
                       , frames_per_video
                       , block_starts
                       , preprocess_type=None
                       , labels = None
                       , t_window=3
                       , im_height=112, im_width=112
                       , repeat=False, shuffle=False
                       , batch_size=1, prefetch=1, num_procs=4
                       , *args, **kwargs):
	"""
	:param video_paths: list of all paths to frame images for videos
	:param frames_per_video: number of frames found in each video
	:param block_starts: frame indices of where blocks start per video
	:param preprocess_type: image preprocessing Defautls to image resize
	:param labels: labels for inputs
	:param t_window: number of consecutive frames to use in blocks
	:param im_height: height of images after preporcessing
	:param im_width: width of images after preprocessing
	:param repeat: flag to set whether dataset iterator should repeat after all entries in iterator is exhausted
				   or to repeat entries in dataset
	:param shuffle: flat to set whether dataset should be shuffled
	:param batch_size: number of entries in dataset to batch (by number of t_window blocks)
	:param prefetch: number of batches to prefetch (by number of t_window blocks)
	:param num_procs: number of parallel threads
	:return: iterator init op, iterator get_next op
	# :param batch_by_video: batch by video blocks or batch by videos
	"""
	from itertools import product
	num_blocks = len(block_starts)
	batch_size *= num_blocks

	#TODO: remove preprocess_type and allow user to define preprocessing functions
	if preprocess_type is None or not preprocess_type in ['vgg', 'inception']:
		preprocess = lambda img: tf.image.resize_images(img, [im_height, im_width])
	elif preprocess_type == 'vgg':
		from preprocessing import vgg_preprocessing
		preprocess = lambda img: vgg_preprocessing.preprocess_image(img, im_height, im_width, resize_side_min=im_height)
	elif preprocess_type == 'inception':
		from preprocessing import inception_preprocessing
		preprocess = lambda img: inception_preprocessing.preprocess_image(img, im_height, im_width)

	paths_num_block = [
		f'{vid_path_start[0]}:{idx % num_blocks + 1}:{num_blocks}:{vid_path_start[1]}:{vid_path_start[1] + t_window}'
		for idx, vid_path_start in enumerate(list(product(video_paths, block_starts)))]

	input_dataset = tf.data.Dataset. \
		from_tensor_slices(video_paths). \
		interleave(lambda x: stack_im_blocks(x, t_window, block_starts, frames_per_video, [im_height, im_width], preprocess)
	               , cycle_length=num_procs
	               , block_length=num_blocks
	               , num_parallel_calls=num_procs). \
		batch(batch_size=batch_size)

	paths_dataset = tf.data.Dataset.from_tensor_slices(paths_num_block).batch(batch_size=batch_size)
	input_dataset = tf.data.Dataset.zip((input_dataset, paths_dataset))

	if labels:
		stim_path_labels = np.array([(vid_path, label)
		                             for vid_path, label in list(zip(video_paths,labels))
		                             for _ in range(num_blocks)])
		# label dataset:
		stim_path_label_dataset = tf.data.Dataset.from_tensor_slices(stim_path_labels)\
				.batch(batch_size=batch_size)

		input_dataset = tf.data.Dataset.zip((input_dataset, stim_path_label_dataset))

	if repeat:
		input_dataset.repeat()
	if shuffle:
		input_dataset.shuffle(buffer_size=batch_size)

	input_dataset.prefetch(buffer_size=prefetch)
	iterator = tf.data.Iterator.from_structure(input_dataset.output_types, input_dataset.output_shapes)
	iterator_init_op = iterator.make_initializer(input_dataset)
	#TODO: return iterator instead of iterator_init_op, iterator.get_next()
	return iterator_init_op, iterator.get_next()


def paths_iterator(vid_paths, t_window, block_starts, repeat=False, shuffle=False
                       , batch_size=1, prefetch=1):
	"""
	TODO: remove, not used
	:param vid_paths: paths to videos
	:param t_window: time width of blocks
	:param block_starts: a list of indices where blocks of frames start in videos
	:param repeat: flag to set if dataset should repeat (for training)
	:param shuffle: flag to set if dataset should be shuffled (for training)
	:param batch_size: size of the number of videos to batch. ==> batch_size = number of blocks per video * batch_size
	:param prefetch: how many video batches to prefetch
	:return:
	"""
	from itertools import product
	num_blocks = len(block_starts)
	batch_size *= num_blocks
	paths_num_block = [f'{vid_path_start[0]}:{idx%num_blocks + 1}:{num_blocks}:{vid_path_start[1]}:{vid_path_start[1]+t_window}'
                        for idx, vid_path_start in enumerate(list(product(vid_paths, block_starts)))]

	input_dataset = tf.data.Dataset.from_tensor_slices(paths_num_block).batch(batch_size=batch_size)

	if repeat:
		input_dataset.repeat()
	if shuffle:
		input_dataset.shuffle(buffer_size=batch_size)

	input_dataset.prefetch(buffer_size=prefetch)
	iterator = tf.data.Iterator.from_structure(input_dataset.output_types, input_dataset.output_shapes)
	iterator_init_op = iterator.make_initializer(input_dataset)

	return iterator_init_op, iterator.get_next()


def inputs_from_activations(activations, labels, keep_stim_paths=False
                            , repeat=False, shuffle=False, batch_size=64, prefetch=1):
	"""
	TODO: remove, not used
	:param activations: xarray of activations from model-tools
	:param labels: labels for each activation
	:param keep_stim_paths: flag to set whether the column for the paths of the stimulus should be kept
	:param repeat: flag to set whether dataset iterator should repeat once all entries have been consumed
	:param shuffle: flag to set whether the dataset should be shuffled
	:param batch_size: size of batch
	:param prefetch: number of batches to prefetch
	:return: iterator init op, iterator get_next op
	"""
	if isinstance(labels, list):
		labels = np.array(labels)
	act_dataset = tf.data.Dataset.from_tensor_slices(activations)
	label_dataset = tf.data.Dataset.from_tensor_slices(labels)

	if keep_stim_paths:
		stim_path_dataset = tf.data.Dataset.from_tensor_slices(activations['stimulus_path'])
		input_dataset = tf.data.Dataset.zip((act_dataset, label_dataset, stim_path_dataset))
	else:
		input_dataset = tf.data.Dataset.zip((act_dataset, label_dataset))

	if repeat:
		input_dataset.repeat()
	if shuffle:
		input_dataset.shuffle(buffer_size=batch_size)

	input_dataset.batch(batch_size=batch_size).prefetch(buffer_size=prefetch)

	iterator = tf.data.Iterator.from_structure(input_dataset.output_types, input_dataset.output_shapes)
	iterator_init_op = iterator.make_initializer(input_dataset)
	return iterator_init_op, iterator.get_next()


# TODO: remove the need for datainputs object
class TF_di(DataInput):

	def __init__(self, session=None
	                   , frames_per_video=60
	                   , preprocess_type=None
	                   , labels=None
                       , t_window=3
                       , im_height=112, im_width=112
                       , repeat=False, shuffle=False
                       , batch_size=1, prefetch=1, num_procs=4, *args, **kwargs):
		self._session = session
		self.fpv = frames_per_video
		self.preprocess_type = preprocess_type
		self.labels=labels
		self._t_window = t_window
		self._height = im_height
		self._width = im_width
		self.repeat = repeat
		self.shuffle = shuffle
		self.batch_size = batch_size
		self.prefetch = prefetch
		self.num_procs = num_procs
		self.block_starts = block_stride_starts(frames_per_video, t_window)
		self.stim_paths = None


	def make_from_paths(self, paths):
		init_op, self.iterator = video_data_inputs(paths, frames_per_video=self.fpv, block_starts=self.block_starts
		                                                , preprocess_type=self.preprocess_type
		                                                , labels=self.labels
		                                                , t_window=self._t_window
		                                                , im_height=self._height, im_width=self._width
		                                                , repeat=self.repeat, shuffle=self.shuffle
		                                                , batch_size=self.batch_size, prefetch=self.prefetch
		                                                , num_procs=self.num_procs)
		self._session.run([init_op])

	#TODO: remove get_next_stim
	def get_next_stim(self):
		data = self._session.run(self.iterator)
		self.stim_paths = list(data[1])
		#store paths:
		return data[0]

	def get_stim_paths(self):
		return self.stim_paths
