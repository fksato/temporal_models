import cv2
import numpy as np
import tensorflow as tf

from data_inputs import DataInput


def video_to_frames(src_vid, stride=1, offset=15, vid_frames_cnt=60):
	"""
	TODO: Redundancy with PyTorch, needed since src_vid needs to be decoded from byte to string for cv2 to work
	TODO: for 4d models its not necessary to vary frame_count since packaging occurs elsewhere
	Takes a video path, parse into frames by frame_count and return array of images
	:param src_vid: path to video
	:param stride: number of frames to stride over
	:param offset: start frame in video
	:param vid_frames_cnt: total frames in video (less offset)
	:return: array of frames
	"""
	# check type of src_vid
	cap = cv2.VideoCapture(bytes(src_vid).decode("utf-8"))
	if not cap.isOpened():
		raise Exception(f'{bytes(src_vid).decode("utf-8")} file cannot be opened')

	frame_count = int(vid_frames_cnt / stride)
	width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	frames_array = np.zeros(shape=(frame_count, height, width, 3), dtype=np.float32)

	# set video frame start to frame 15 (0.5 seconds into video)
	# 1: "CAP_PROP_POS_FRAMES", 15: starting frame position
	cap.set(cv2.CAP_PROP_POS_FRAMES, offset)
	frame_array_cnt = 0
	f_cnt = 0

	while cap.isOpened() and f_cnt < vid_frames_cnt:
		success, buff = cap.read()
		if success:
			if f_cnt % stride == 0:
				frames_array[frame_array_cnt] = buff.astype(np.float32)
				frame_array_cnt += 1
			f_cnt += 1

	cap.release()
	return height, width, frames_array


def stack_im_blocks(path, stride, offset=15, frames=60, single_frames=False, image_size=[112, 112], image_preprocess=None):
	"""
	stack video frames into blocks of images. 4D models expects input shape = [None, t,h,w,c] == im_blocks
	:param path: path to video
	:param stride: starting index for each block
	:param offset: offset for video start (by frames)
	:param frames: frames per video
	:param single_frames: make 4D tensors or a series of 3D images (for FF models)
	:param image_size: resize image size for each frame
	:param image_preprocess: image preprocess
	:return: stacked image tensor
	"""

	# tf function wrapper for video_to_frames()
	def tf_video_to_frames(_src_vid, _stride, _offset, _frames):
		frames_block_shape = int(_frames/stride)
		im_h, im_w, out_put = tf.py_func(video_to_frames, [_src_vid, _stride, _offset, _frames], [tf.int64, tf.int64, tf.float32])
		return tf.reshape(out_put, [frames_block_shape, im_h, im_w, 3])
	# call to tf function wrapper
	frames_array = tf_video_to_frames(path, stride, offset, frames)

	if single_frames:
		im_block = tf.map_fn(image_preprocess, frames_array)
	else:
		im_block = tf.stack([tf.map_fn(image_preprocess, frames_array)])

	return tf.data.Dataset.from_tensor_slices(im_block)


def video_data_inputs(video_paths
                       , frames_per_video
                       , stride
                       , preprocess_type=None
                       , labels = None
                       , offset = 15
                       , single_frames = False
                       , im_height=112, im_width=112
                       , batch_size=1
                       , prefetch=1
                       , num_procs=4):
	"""
	:param video_paths: list of all paths to frame images for videos
	:param frames_per_video: number of frames found in each video
	:param stride: number of frames to stride over in videos
	:param preprocess_type: image preprocessing Defautls to image resize
	:param labels: labels for inputs
	:param offset: offset from video start in frames
	:param single_frames: 3D or 4D blocks (FF inputs or recursive inputs)
	:param im_height: height of images after preporcessing
	:param im_width: width of images after preprocessing
	:param batch_size: number of entries in dataset to batch (by number of t_window blocks)
	:param prefetch: buffer size to fill with prefetch
	:param num_procs: number of parallel threads
	:return: tf.data.Dataset
	"""

	num_frames_per_video = int(frames_per_video/stride)
	batch_frames = batch_size
	if not single_frames:
		block_width = 1
		paths_num_block = video_paths
	else:
		batch_frames = batch_size * num_frames_per_video
		block_width = num_frames_per_video
		paths_num_block = [
			[f'{vid_path}:{idx % num_frames_per_video + 1}:{num_frames_per_video}'] for idx in
			 range(num_frames_per_video) for vid_path in video_paths]

	#TODO: remove preprocess_type and allow user to define preprocessing functions
	if preprocess_type is None or not preprocess_type in ['vgg', 'inception']:
		preprocess = lambda img: tf.image.resize_images(img, [im_height, im_width])
	elif preprocess_type == 'vgg':
		from preprocessing import vgg_preprocessing
		preprocess = lambda img: vgg_preprocessing.preprocess_image(img, im_height, im_width, resize_side_min=im_height)
	elif preprocess_type == 'inception':
		from preprocessing import inception_preprocessing
		preprocess = lambda img: inception_preprocessing.preprocess_image(img, im_height, im_width)

	input_dataset = tf.data.Dataset. \
		from_tensor_slices(video_paths). \
		interleave(lambda x: stack_im_blocks(x, stride, offset=offset, frames=frames_per_video
	                                         , single_frames=single_frames, image_size=[im_height, im_width]
	                                         , image_preprocess=preprocess)
	               , cycle_length=num_procs
	               , block_length=block_width
	               , num_parallel_calls=num_procs). \
		batch(batch_size=batch_frames)

	paths_dataset = tf.data.Dataset.from_tensor_slices(paths_num_block).batch(batch_size=batch_frames)
	input_dataset = tf.data.Dataset.zip((input_dataset, paths_dataset))

	# if labels:
	# 	stim_path_labels = np.array([(vid_path, label)
	# 	                             for vid_path, label in list(zip(video_paths,labels))
	# 	                             for _ in range(num_blocks)])
	# 	# label dataset:
	# 	stim_path_label_dataset = tf.data.Dataset.from_tensor_slices(stim_path_labels)\
	# 			.batch(batch_size=batch_size)
	#
	# 	input_dataset = tf.data.Dataset.zip((input_dataset, stim_path_label_dataset))

	input_dataset = input_dataset.prefetch(buffer_size=prefetch)

	return input_dataset


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


# TODO: remove the need for datainputs object
class TensorflowVideoDataInput(DataInput):
	def __init__(self, session=None, frames_per_video=60, stride=1, single_frames=False, offset=15, preprocess_type=None
				, labels=None, im_height=112, im_width=112, repeat=False, shuffle=False, batch_size=1
				, prefetch=1, num_procs=4, *args, **kwargs):
		self._session = session
		self.fpv = frames_per_video
		self.stride = stride
		self.single_frames = single_frames
		self.offset = offset
		self.preprocess_type = preprocess_type # make it function
		self.labels=labels
		self._height = im_height
		self._width = im_width
		self.repeat = repeat
		self.shuffle = shuffle
		self.batch_size = batch_size
		self.prefetch = prefetch
		self.num_procs = num_procs
		self.stim_paths = None
		self.units = frames_per_video if single_frames else 1



	def make_from_paths(self, paths):
		# init_op,
		dataset = video_data_inputs(paths, frames_per_video=self.fpv, stride=self.stride
		                                                , preprocess_type=self.preprocess_type
		                                                , labels=self.labels
							                            , offset=self.offset
							                            , single_frames=self.single_frames
		                                                , im_height=self._height, im_width=self._width
		                                                , batch_size=self.batch_size
							                            , prefetch=self.prefetch
		                                                , num_procs=self.num_procs)
		# self._session.run([init_op])
		iterator = dataset.make_one_shot_iterator()
		self.next_elem = iterator.get_next()



if __name__=="__main__":
	from glob import glob
	video_paths = glob('/braintree/home/fksato/Projects/models/tests/testing_vid/*.mp4')
	frames_per_video = 60
	stride = 3
	single_frames = False
	batch_size = 2

	dataset = video_data_inputs(video_paths, frames_per_video, stride
	                  , preprocess_type='vgg'
	                  , single_frames=single_frames
	                  , batch_size=batch_size)

	iterator =  dataset.make_one_shot_iterator()
	next_elem = iterator.get_next()

	with tf.Session() as sess:
		data_check = sess.run(next_elem)
		print(data_check)

