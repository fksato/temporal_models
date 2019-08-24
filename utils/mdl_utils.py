import os
import numpy as np
from glob import glob


def split_train_test(category_size, validation_size, num_vids_per_cat):
	"""
	split videos (per class) into train-validation
	randomly choose: 2 numbers (range 0-9, without replacement)
	:param category_size: number of action classes
	:param validation_size: size of validation set (takes an int or a percentage)
	:param num_vids_per_cat: number of videos per action class
	:return: list of video indices per action class which will be used for the validation set
	"""
	validation_ids = []
	if isinstance(validation_size, float):
		validation_size = int(num_vids_per_cat * validation_size)

	for i in range(category_size):
		validation_choice = np.random.choice(num_vids_per_cat, validation_size, replace=False)
		validation_ids.append(validation_choice)
	return validation_ids


def get_stimulus_train_test(validation_ids, action_categories, video_directory
                            , num_vids_per_cat, num_frames=60, one_hot=True):
	"""
	:param validation_ids: video index which are to be used for validation (local to action-class)
	:param action_categories: list of action classes
	:param video_directory: parent directory where all action videos are stored
	:param num_vids_per_cat: number of videos per action class
	:param num_frames: number of frames to process per video
	:param one_hot: flag to set whether the labels should be one-hot or a scalar
	:return: list of training paths, nd array of training labels, list of validation paths, nd array of validation labels
	"""
	training_im_paths = []  # paths to training images (frames)
	train_labels = []  # labels for training images
	validation_im_paths = []  # ^same for validation
	validation_labels = []  # ^same for validation
	action_class_len = len(action_categories)
	vids_in_cat = num_vids_per_cat

	if isinstance(num_vids_per_cat, list):
		tot_vid_frames = sum(num_vids_per_cat) * num_frames
	else:
		tot_vid_frames = num_vids_per_cat * len(action_categories) * num_frames

	for action_idx, _action in enumerate(action_categories):  # action index (), action class
		vid_path = os.path.join(video_directory, _action)  # create video search path: (*/*/action_class/)
		frame_path = os.path.join(vid_path, 'frames')  # create frames directory: (*/*/action_class/frames)
		if one_hot:
			_labels = [[1 if i == action_idx else 0 for i in range(action_class_len)]]  # make class vectors
		else:
			_labels = [action_idx]
		vid_files = glob(f'{vid_path}/*.mp4')  # get list of videos from search path
		if num_vids_per_cat is None:
			vids_in_cat = len(vid_files)
		elif isinstance(num_vids_per_cat, list):
			vids_in_cat = num_vids_per_cat[action_idx]

		for i in range(vids_in_cat):  # for each vid in search dir/user input
			f_path = os.path.basename(vid_files[i]).replace('.mp4', '')  # remove '.mp4' ext
			im_path = os.path.join(frame_path, f_path)  # use f_path to make dir of frames for vid
			if i in validation_ids[action_idx]:  # if part of the validation set
				im_paths = sorted(glob(f'{im_path}/*.png'))  # store all frame image paths
				assert len(im_paths) == num_frames  # assert len of frames is expected frame counts
				validation_im_paths.extend(im_paths)  # add to validation image path list
				validation_labels.extend(_labels * num_frames)  # add class labels to validation labels list
			else:  # part of training set
				im_paths = sorted(glob(f'{im_path}/*.png'))  # same as validation set
				assert len(im_paths) == num_frames
				training_im_paths.extend(im_paths)
				train_labels.extend(_labels * num_frames)
	assert len(training_im_paths) == len(train_labels)  # asset training set is the same size as label sets
	assert len(validation_im_paths) == len(validation_labels)  # same check for size
	assert len(training_im_paths) + len(
		validation_im_paths) == tot_vid_frames  # check to make sure total sizes agree with splits
	return training_im_paths, np.array(train_labels), validation_im_paths, np.array(validation_labels)


def model_summary(graph):
	with graph.as_default() as g:
		graph_nodes = [n for n in g.get_operations()]
		for i in graph_nodes:
			if len(i.outputs) == 0:
				continue
			print(f'op: {i.name}  output shape: {i.outputs[0].shape}')


def get_vid_paths(actions_list, main_vid_dir):
	"""
	get video paths from main_vid_dir for each action in actions_list
	:param actions_list: list of actions in video dataset
	:param main_vid_dir: location of all videos with path: 'main_vid_dir/<action>/*.mp4'
	:return: video counts per action class, full paths to videos per action
	"""
	action_video_paths = []
	action_vid_cnt = {}
	for action in actions_list:
		vid_dir_path = f'{main_vid_dir}/{action}/*.mp4'
		vid_list = glob(vid_dir_path)
		action_vid_cnt[action] = len(vid_list)
		action_video_paths += vid_list
	return action_vid_cnt, action_video_paths


def get_batched_vid_paths(actions_list, main_vid_dir, start, max_vid_cnt):
	batched_action_video_paths = []
	for action in actions_list:
		vid_dir_path = f'{main_vid_dir}/{action}/*.mp4'
		vid_list = glob(vid_dir_path)
		if len(vid_list) < start:
			continue
		if len(vid_list[start:]) < max_vid_cnt:
			vid_list = vid_list[start:]
		else:
			vid_list = vid_list[start:start+max_vid_cnt]
		batched_action_video_paths += vid_list
	if len(batched_action_video_paths) == 0:
		return False
	return batched_action_video_paths


def _make_group_offsets(actions, max_vid_cnt, global_counts):
	"""
	utility function to get video index offsets in batched activations
	:param actions: actions list in video dataset
	:param max_vid_cnt: max video count per action class per batched activation file (group)
	:param global_counts: total counts of video dataset. type: dictionary: action: video counts
	:return: list of offsets for each batched activation files (group)
	"""
	max_group_nums = [int(global_counts[action]/max_vid_cnt) for action in actions]
	action_group_starts = {}

	for idx, action in enumerate(actions):
		group_offset = []
		for group in range(0, max_group_nums[idx] + 1):
			group_sum = 0
			for j in range(idx):
				if max_group_nums[j] < group:
					continue
				if max_group_nums[j] > group:
					group_sum += max_vid_cnt
				elif max_group_nums[j] == group:
					group_sum += global_counts[actions[j]] % max_vid_cnt
			group_offset.append( (group, group_sum) )
		action_group_starts[action] = group_offset

	return action_group_starts


def _get_group_index(offsets, indices, max_vid_cnt, frames_block_count=0):
	"""
	utility function to get a mapping from global coordinates to batched activation files (group) coordinates
	:param offsets: offsets calculated from above
	:param indices: global index to map
	:param max_vid_cnt: max video count per action class per batched activation files (group)
	:param frames_block_count: time-width size of inputs from activations (8,16 for ResNet2.5D-18,34 respectively)
	:return: remapped global coordinates to batched activation files (group) coordinates
	"""
	group_indices = {group:[] for group, offset in offsets}
	for idx in indices:
		group_num = int(idx/max_vid_cnt)
		local_idx = idx - group_num * max_vid_cnt
		test = offsets[group_num][1] + local_idx
		if frames_block_count > 0:
			test *= frames_block_count
		group_indices[group_num] += [test] if frames_block_count==0 else [test + ix for ix in range(frames_block_count)]
	return group_indices


def get_tain_test_groups(actions, validation_size, testing_size, max_vid_count, vid_dir, frames_block_cnt=0):
	"""
	train, validation, test split per batched activation files (group)
	:param actions: action classes in video dataset
	:param validation_size: size of validation set as a percentage of total training size
	:param testing_size: size of testing set as a percentage of total
	:param max_vid_count: max video count per action per batchced activation files (group)
	:param vid_dir: main video directory where videos are stored: path structure: <vid_dir>/<action>/*.mp4
	:param frames_block_cnt: number of (t,h,w) image blocks per video
	:return: dictionary of training, validation, testing coordinates per batched activation files (group)
	"""
	import operator
	np.random.seed(1234)

	global_counts = {action: len(glob(f'{vid_dir}/{action}/*.mp4')) for action in actions}
	max_group = int(max(global_counts.items(), key=operator.itemgetter(1))[1]/max_vid_count) + 1

	group_offsets = _make_group_offsets(actions, max_vid_count, global_counts)

	testing_idx = {action: np.random.choice(vidset_size, int(vidset_size * testing_size), replace=False)
	                  for action, vidset_size in global_counts.items()}

	training_idx = {action: [i for i in range(vidset_size) if i not in testing_idx[action]]
	                for action, vidset_size in global_counts.items()}

	validation_idx = {action: [indices[i] for i in np.random.choice(len(indices), int(len(indices) * validation_size), replace=False)]
	               for action, indices in training_idx.items()}

	group_testing = {group: [] for group in range(max_group)}
	group_train = {group: [] for group in range(max_group)}
	group_val = {group: [] for group in range(max_group)}

	group_testing_labels = {group: [] for group in range(max_group)}
	group_train_labels = {group: [] for group in range(max_group)}
	group_val_labels = {group: [] for group in range(max_group)}

	for idx, action in enumerate(actions):
		offset = group_offsets[action]
		testing_groups = _get_group_index(offset, testing_idx[action], max_vid_count, frames_block_cnt)
		group_testing.update({group: group_testing[group] + idx
		                  for group, idx in testing_groups.items()})

		train_groups = _get_group_index(offset, training_idx[action], max_vid_count, frames_block_cnt)
		group_train.update({group: group_train[group] + idx
		                    for group, idx in train_groups.items()})

		val_groups = _get_group_index(offset, validation_idx[action], max_vid_count, frames_block_cnt)
		group_val.update({group: group_val[group] + idx
		                    for group, idx in val_groups.items()})

		group_testing_labels.update({group: group_testing_labels[group] + [idx for _ in range(len(testing_groups[group]))]
		                    for group in testing_groups.keys()})

		group_train_labels.update({group: group_train_labels[group] + [idx for _ in range(len(train_groups[group]))]
		                      for group in train_groups.keys()})

		group_val_labels.update({group: group_val_labels[group] + [idx for _ in range(len(val_groups[group]))]
		                           for group in val_groups.keys()})

	return group_train, group_testing, group_val, group_train_labels, group_testing_labels, group_val_labels



def mask_unused_gpus(leave_unmasked=1):
	"""
	utility function to hide gpus that do not meet memory requirements from being recognized in tensorflow/PyTorch
	:param leave_unmasked: minimum number of available gpus
	:return: list of gpu ids sorted by descending max available gpu memory
	"""
	import subprocess as sp

	ACCEPTABLE_AVAILABLE_MEMORY = 1024
	COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"

	try:
		_output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
		memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
		memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
		memory_free_values = sorted( [(e,i) for i,e in enumerate(memory_free_values)], reverse=True)
		available_gpus = [x[1] for x in memory_free_values if x[0] > ACCEPTABLE_AVAILABLE_MEMORY]

		if len(available_gpus) < leave_unmasked:
			raise ValueError('Found only %d usable GPUs in the system' % len(available_gpus))
		os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, available_gpus))
	except Exception as e:
		raise Exception('"nvidia-smi" is probably not installed. GPUs are not masked', e)

	return available_gpus

if __name__=='__main__':
	# main_vid_dir = '/braintree/home/fksato/Projects/aciton_recognition/vid_buckets'
	# action_classes = ['Arm wrestling',
	# 				  'Brushing teeth',
	# 				  'Doing fencing',
	# 				  'Horseback riding',
	# 				  'Hula hoop',
	# 				  'Rope skipping',
	# 				  'Swimming']
	#
	# vid_cnts, vid_paths = get_vid_paths(action_classes, main_vid_dir)
	# print(len(vid_paths))
	# print('check')
	from models import HACS_ACTION_CLASSES as actions
	import time

	vid_dir = '/braintree/home/fksato/HACS_total/training'
	max_vid_count = 25

	frames_block_cnt=60
	validation_size = .1

	start_time = time.clock()
	train, test, train_labels, test_labels = get_tain_test_groups(actions, validation_size, max_vid_count, vid_dir)
	performance = time.clock() - start_time
	print(f'PERFORMANCE: {performance}')
	print('check')

	# check_offset = 35
	# test_action = 'Skateboarding'
	# double_check = glob(f'{vid_dir}/{test_action}/*.mp4')
	# check_indices = len(double_check) - check_offset
	# check_path = double_check[check_indices]
	# print(f'{check_path}: {check_indices}')

	# g_num = int(check_indices/max_vid_count)
	# global_counts = {action: len(glob(f'{vid_dir}/{action}/*.mp4')) for action in actions}
	# test_offset = _make_group_offsets(actions, max_vid_count, global_counts)
	# idx = _get_group_index(test_offset[test_action], [check_indices], max_vid_count)[g_num][0]
	#
	# batched_paths = get_batched_vid_paths(actions_list=actions, main_vid_dir=vid_dir
	#                                       , start=g_num*25, max_vid_cnt=max_vid_count)
	#
	# assert batched_paths[idx] == check_path
	# print(f'check_group = {g_num}\ncheck_idx = {idx*frames_block_cnt}')
	#
	# print('check')

