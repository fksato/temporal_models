import cv2
import numpy as np
import tensorflow as tf

from data_inputs import DataInput


def video_to_frames(src_vid, stride=1, offset=15, vid_frames_cnt=75, _frame_count=None):
	"""
	Redundancy with PyTorch, needed since src_vid needs to be decoded from byte to string for cv2 to work
	Takes a video path, parse into frames by frame_count and return array of images
	:param src_vid: path to video
	:param stride: number of frames to stride over
	:param offset: start frame in video
	:param vid_frames_cnt: total frames in video (full video frame count = 75)
	:param _frame_count: number of frames in blocks
	:return: array of frames
	"""
	# check type of src_vid
	cap = cv2.VideoCapture(bytes(src_vid).decode("utf-8"))
	if not cap.isOpened():
		raise Exception(f'{bytes(src_vid).decode("utf-8")} file cannot be opened')

	frame_count =  _frame_count if _frame_count is not None else int( (vid_frames_cnt - offset) / stride)
	width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	frames_array = np.zeros(shape=(frame_count, height, width, 3), dtype=np.float32)

	# set video frame start to frame 15 (0.5 seconds into video)
	# 1: "CAP_PROP_POS_FRAMES", 15: starting frame position
	cap.set(cv2.CAP_PROP_POS_FRAMES, offset)
	frame_array_cnt = 0
	f_cnt = offset

	while cap.isOpened() and (f_cnt < vid_frames_cnt and frame_array_cnt < frame_count):
		success, buff = cap.read()
		if success:
			if f_cnt % stride == 0:
				buff = buff / 255.0
				frames_array[frame_array_cnt] = buff.astype(np.float32)
				frame_array_cnt += 1
			f_cnt += 1

	cap.release()
	return height, width, frames_array


def stack_im_blocks(path, stride, offset=15, frames=75, single_frames=False, image_preprocess=None, frame_count=None):
	"""
	stack video frames into blocks of images. 4D models expects input shape = [None, t,h,w,c] == im_blocks
	:param path: path to video
	:param stride: starting index for each block
	:param offset: offset for video start (by frames)
	:param frames: frames per video
	:param single_frames: make 4D tensors or a series of 3D images (for FF models)
	:param image_preprocess: image preprocess
	:param frame_count: number of frames per block
	:return: stacked image tensor
	"""

	# tf function wrapper for video_to_frames()
	def tf_video_to_frames(_src_vid, _stride, _offset, _frames, _frame_count=None):
		frames_block_shape = _frame_count if _frame_count else int((_frames - offset) /stride)
		if _frame_count:
			func_args = [_src_vid, _stride, _offset, _frames, _frame_count]
		else:
			func_args = [_src_vid, _stride, _offset, _frames]
		im_h, im_w, out_put = tf.py_func(video_to_frames, func_args, [tf.int64, tf.int64, tf.float32])

		return tf.reshape(out_put, [frames_block_shape, im_h, im_w, 3])
	# call to tf function wrapper
	frames_array = tf_video_to_frames(path, stride, offset, frames, frame_count)

	im_block = tf.map_fn(image_preprocess, frames_array)

	if single_frames:
		im_block = tf.data.Dataset.from_tensors_slices(im_block)
	else:
		im_block = tf.data.Dataset.from_tensors(im_block)

	return im_block


def video_data_inputs(video_paths
                       , frames_per_video
                       , stride
                       , preprocess=None
                       , offset = 15
                       , single_frames = False
                       , im_height=112, im_width=112
                       , batch_size=1
                       , prefetch=1
                       , num_procs=4
                       , frame_count=None):
	"""
	:param video_paths: list of all paths to frame images for videos
	:param frames_per_video: number of frames found in each video
	:param stride: number of frames to stride over in videos
	:param preprocess: image preprocessing Defautls to image resize
	:param offset: offset from video start in frames
	:param single_frames: 3D or 4D blocks (FF inputs or recursive inputs)
	:param im_height: height of images after preporcessing
	:param im_width: width of images after preprocessing
	:param batch_size: number of entries in dataset to batch (by number of t_window blocks)
	:param prefetch: buffer size to fill with prefetch
	:param num_procs: number of parallel threads
	:param frame_count: number of frames in blocks
	:return: tf.data.Dataset
	"""

	num_frames_per_video = int( (frames_per_video-offset) / stride)
	batch_frames = batch_size
	# blocks of frames (t, h, w)
	if not single_frames:
		block_width = 1
		paths_num_block = video_paths
	# single frames t * (h, w)
	else:
		batch_frames = batch_size * num_frames_per_video
		block_width = num_frames_per_video
		paths_num_block = [
			[f'{vid_path}:{idx % num_frames_per_video + 1}:{num_frames_per_video}'] for idx in
			 range(num_frames_per_video) for vid_path in video_paths]

	if preprocess is None :
		preprocess = lambda img: tf.image.resize_images(img, [im_height, im_width])

	input_dataset = tf.data.Dataset. \
		from_tensor_slices(video_paths). \
		interleave(lambda x: stack_im_blocks(x, stride, offset=offset, frames=frames_per_video
	                                         , single_frames=single_frames, image_preprocess=preprocess
	                                         , frame_count=frame_count)
	               , cycle_length=num_procs
	               , block_length=block_width
	               , num_parallel_calls=num_procs). \
		batch(batch_size=batch_frames)

	paths_dataset = tf.data.Dataset.from_tensor_slices(paths_num_block).batch(batch_size=batch_frames)
	input_dataset = tf.data.Dataset.zip((input_dataset, paths_dataset))

	input_dataset = input_dataset.prefetch(buffer_size=prefetch)

	return input_dataset


class TensorflowVideoDataInput(DataInput):
	def __init__(self, session=None, frames_per_video=60, stride=1, single_frames=False, offset=15, preprocess=None
				, labels=None, im_height=112, im_width=112, repeat=False, shuffle=False, batch_size=1
				, prefetch=1, num_procs=4, frame_count=None, *args, **kwargs):
		self._session = session
		self.fpv = frames_per_video
		self.stride = stride
		self.single_frames = single_frames
		self.offset = offset
		self.preprocess = preprocess
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
		self.frame_count = frame_count



	def make_from_paths(self, paths):
		# init_op,
		dataset = video_data_inputs(paths, frames_per_video=self.fpv, stride=self.stride
		                                                , preprocess=None
							                            , offset=self.offset
		                                                , frame_count=self.frame_count
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
	import os

	testing_dir = '/braintree/home/fksato/Projects/models/tests/testing_vid/'
	video_paths = glob('/braintree/home/fksato/Projects/models/tests/testing_vid/*.mp4')

	frames_per_video = 75 # total vid size = 2.5 s
	stride = 3
	offset = 27

	single_frames = False
	batch_size = 2

	processed_im_hw = 112
	channels = 3

	preprocess = lambda img: tf.image.per_image_standardization(tf.image.crop_to_bounding_box(
								tf.image.resize_images(img, [processed_im_hw, 200], preserve_aspect_ratio=True)
								, 0, 44, processed_im_hw, processed_im_hw))

	dataset = video_data_inputs(video_paths, frames_per_video, stride=stride, offset=offset
	                            , preprocess=preprocess
								, single_frames=single_frames
								, batch_size=batch_size)

	iterator =  dataset.make_one_shot_iterator()
	next_elem = iterator.get_next()

	with tf.Session() as sess:
		data_check = sess.run(next_elem)
		# print(data_check)

	assert data_check[0].shape == (batch_size, int( (frames_per_video - offset) /stride), processed_im_hw, processed_im_hw, channels)

	for vid_id, vid in enumerate(data_check[0]):
		for frame_id, frames in enumerate(vid):
			vid_frames_dir = f'{testing_dir}/tf_data_frames/vid_{vid_id}'
			if not os.path.isdir(vid_frames_dir):
				os.mkdir(vid_frames_dir)

			cv2.imwrite(f'{vid_frames_dir}/frame_image_{frame_id}.png', frames * 255)

	print('DONE')


