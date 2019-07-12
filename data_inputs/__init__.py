import os
import subprocess
import logging

import numpy as np
from glob import glob

logging.root.setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.CRITICAL)
# create logger
split_frame_logger = logging.getLogger('split_frame_logger')
split_frame_logger.setLevel(logging.WARNING)

#TODO: remove datainputs dependencies
class DataInput:

	def make_from_paths(self, paths):
		pass

	# def get_next_stim(self):
	# 	pass
	#
	# def get_stim_paths(self):
	# 	pass


def split_frames(src_vid, path, start, num_frames):
	"""
	:param src_vid: path to video to split into frames
	:param path: path to save frames
	:param start: where to begin splitting into frames (in seconds)
	:param num_frames: number of frames to split
	:return: 1 on success (True)

	executes: ffmpeg -y -v quiet -i <src_vid> -vf fps=30 <path>/test%04d.png
	"""
	#
	cmd = ["ffmpeg", "-y"
		, "-v", "quiet"
		, "-i", src_vid
		, "-vf", f"select=gte(n\\,{start})"
		, "-vframes", str(num_frames)
		, f'{path}/frame_%05d.png'
	       ]
	return 1 - subprocess.run(cmd).returncode


def make_action_frames(action_video_directory, vid_start_offset=0.5, number_of_frames=60, verbose=False):
	"""
	:param action_video_directory: directory to the videos associated to an action
	:param vid_start_offset: offset of where to begin parsing into frames (in seconds)
	:param number_of_frames: number of frames to parse per video
	:param verbose: flag to set whether print logger to screen
	:return: returns True on success
	"""
	split_frame_logger.info(f'make_action_frames for {action_video_directory}')
	if verbose:
		split_frame_logger.setLevel(logging.DEBUG)

	video_files = glob(f'{action_video_directory}/*.mp4')
	num_vids_to_process = len(video_files)
	split_frame_logger.debug(f'splitting into frames {num_vids_to_process} videos')
	frames_dir = os.path.join(action_video_directory, 'frames')
	_make_frames_dir(frames_dir)
	split_frame_logger.debug(f'saving frame images to {frames_dir}')
	total_frames_cnt = 0
	for src_vid in video_files:
		split_frame_logger.debug(f'processing {src_vid}')
		video_frames_path = os.path.join(frames_dir, os.path.basename(src_vid).replace('.mp4', ''))
		_make_frames_dir(video_frames_path)
		split_frame_logger.debug(f'saving {src_vid} frame images into {video_frames_path}')
		success = split_frames(src_vid, video_frames_path, vid_start_offset, number_of_frames)
		if not success:
			split_frame_logger.error(f'Unable to split into frame images for {src_vid}\n'
			                         f'input printout:\nvideo frame image path: {video_frames_path}\n'
			                         f'video start offset: {vid_start_offset}\nnumber of frames: {number_of_frames}')
			raise Exception()
		split_frame_logger.debug(f'split frames was successful for {src_vid}')
		total_frames_cnt += len(glob(os.path.join(video_frames_path, '*.png')))
	num_frame_dirs = len(next(os.walk(frames_dir))[1])
	assert num_frame_dirs == num_vids_to_process
	assert total_frames_cnt == number_of_frames * num_vids_to_process
	return True


def _make_frames_dir(frames_path):
	"""
	:param frames_path: frames directory to make
	:return: None
	"""
	if os.path.isdir(frames_path):
		return
	os.mkdir(frames_path)