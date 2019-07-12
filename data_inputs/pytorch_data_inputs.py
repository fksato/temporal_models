import  cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from model_tools.activations.pytorch import torchvision_preprocess

from data_inputs import DataInput


def vid_frames(vid_path):
	"""
	Takes a video path, parse into frames by frame_count and return array of images
	:param vid_path: paths to videos (type: string)
	:return: frames of videos as an array
	TODO: allow for frame count to be set by user
	"""
	cap = cv2.VideoCapture(vid_path)
	if not cap.isOpened():
		raise Exception(f'{vid_path} file cannot be opened')

	frame_count = 60
	width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	frames_array = np.zeros(shape=(frame_count, height, width, 3), dtype=np.uint8)

	cap.set(1, 15)
	f_cnt = 0

	while cap.isOpened() and f_cnt < frame_count:
		success, buff = cap.read()
		if success:
			frames_array[f_cnt] = buff.astype(np.uint8)
			f_cnt += 1

	cap.release()
	return frames_array

class VideoDataSet(Dataset):

	def __init__(self, vid_paths, fpv=60, image_size=227
				 , transform=None): #labels
		"""
		dataset object for videos
		:param vid_paths: paths to videos
		:param fpv: frames per video
		:param image_size: size of image to feed into models
		:param transform: any preprocessing transforms for images
		"""
		self.vid_paths = vid_paths
		self.resize_imgs = image_size
		self._fpv = fpv
		self.transform = transform

	def __len__(self):
		return len(self.vid_paths)

	def __getitem__(self, item):
		"""
		Getter function for dataset
		:param item: indices for vid_paths
		:return: images and the path ID for image
		path ID format: video_path+frame_number+total_frame_count
		"""
		frames = vid_frames(self.vid_paths[item]) # (60, h, w, 3)
		paths = [f'{self.vid_paths[item]}:{f_idx%self._fpv + 1}:{self._fpv}' for f_idx in range(self._fpv)]
		imgs = [Image.fromarray(im, 'RGB') for im in frames]

		if self.transform:
			imgs = [self.transform(img) for img in imgs]

		return imgs, paths


def pack_frames(batch):
	"""
	pack frames from videos into contiguous groups and batch
	:param batch: size of batch
	:return: stacked frames with their corresponding paths
	"""
	# if len(batch[0][0]) > 1:
	# 	input_stack = [frame for vid in batch for frame in vid[0]]
	# 	labels_stack = np.array([label for vid in batch for label in vid[1]])
	# 	input_stack = np.array(input_stack)
	# 	return torch.from_numpy(input_stack), torch.from_numpy(labels_stack)
	# else:
	img_stack = [frame for vid in batch for frame in vid[0]]
	img_stack = np.array(img_stack)
	path_stack = np.array([path for vid in batch for path in vid[1]])
	return torch.from_numpy(img_stack), path_stack


# TODO: remove datainputs dependencies
class Torch_di(DataInput):

	def __init__(self, batch_size, shuffle, num_workers, frames_per_video=60, image_size=227, collate_fn=pack_frames):

		self.batch_size = batch_size
		self.shuffle = shuffle
		self.num_workers = num_workers
		self.image_size = image_size
		self.fpv = frames_per_video
		self.collate_fn = collate_fn
		self.stim_paths = None
		self.iterator = None
		self.units = frames_per_video

	def make_from_paths(self, paths):
		from torchvision import transforms
		self.paths = paths
		normalize_mean = (0.485, 0.456, 0.406)
		normalize_std = (0.229, 0.224, 0.225)
		preprocess = transforms.Compose([transforms.CenterCrop(self.image_size)
		                                 , transforms.ToTensor()
		                                 , transforms.Normalize(mean=normalize_mean, std=normalize_std)
		                                 , transforms.Lambda(lambda x: x.numpy())])

		vid_dataset = VideoDataSet(paths, image_size=self.image_size, transform=preprocess)
		self.dataloader = DataLoader(vid_dataset, batch_size=self.batch_size, shuffle=self.shuffle
									 , num_workers=self.num_workers, collate_fn=self.collate_fn)

		self.iterator = iter(self.dataloader)

	# def get_next_stim(self):
	# 	data = next(self.iterator)
	# 	self.
	# 	return data[0]
	#
	# def get_stim_paths(self):
	# 	return self.stim_paths


if __name__=='__main__':

	import torchvision.models as models
	alexnet = models.alexnet(pretrained=True)
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:1" if use_cuda else "cpu")
	alexnet.cuda()
	alexnet.eval()
	alexnet.to(device)

	vid_paths = ['/braintree/home/fksato/Projects/aciton_recognition/vid_buckets/Swimming/video_00000.mp4'
			 , '/braintree/home/fksato/Projects/aciton_recognition/vid_buckets/Swimming/video_00002.mp4'
			 , '/braintree/home/fksato/Projects/aciton_recognition/vid_buckets/Swimming/video_00003.mp4']

	labels = [1,1,1]

	# vid_dataset = VideoDataSet(vid_paths, labels)
	# dataloader = DataLoader(vid_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=pack_frames)

	di_test = Torch_di(batch_size=1, shuffle=False, num_workers=4, image_size=227, collate_fn=pack_frames)
	di_test.make_from_paths(vid_paths)
	x = di_test.get_next_stim()
	s = di_test.get_stim_paths()
	print(s)
	in_p = x.to(device)
	out = alexnet(in_p)
	print(out)


