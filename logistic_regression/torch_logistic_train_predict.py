'''

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

'''
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from logistic_regression.logistic_train_predict import load_act_from_group
from tqdm import tqdm
from glob import glob
import os

import numpy as np


DATASET_TRAIN_VID_CNT = 90427
DATASET_TEST_VID_CNT = 9945


class ActivationsDataset(Dataset):
	def __init__(self, activation_paths, fpv, shuffle=False):
		"""
		dataset object for videos
		:param activation_paths: paths to activations
		:param fpv: frames per video
		"""
		self._activation_paths = activation_paths
		self._fpv = fpv
		self._shuffle = shuffle

	def __len__(self):
		return len(self._activation_paths)

	def __getitem__(self, item):
		"""
		Getter function for dataset
		:param item: indices for activation_paths
		:return: paths, activations, and labels from file
		path ID format: video_path+frame_number+total_frame_count
		"""
		stim_paths, activations, labels = load_act_from_group(self._activation_paths[item], self._fpv, self._shuffle)
		return stim_paths, activations, labels


def pack_activations(batch):

	"""
	img_stack = [frame for vid in batch for frame in vid[0]]
	img_stack = np.array(img_stack)
	path_stack = np.array([path for vid in batch for path in vid[1]])

	:param batch:
	:return:
	"""

	stims = np.array([stimulus for tups in batch for stimulus in tups[0]])
	act = np.array([activations for tups in batch for activations in tups[1]])
	labels = np.array([lab for tups in batch for lab in tups[2]])
	return stims, torch.from_numpy(act), torch.from_numpy(labels)


def _limiter(check_path, limit):
	check_pt_files = glob(check_path)
	if len(check_pt_files) < limit or check_pt_files is None:
		return
	else:
		sorted(check_pt_files, key= lambda x: int(''.join(filter(str.isdigit, x))))
		os.remove(check_pt_files[0])
	return


def _get_latest_checkpoint(search_path):
	check_pt_files = glob(search_path)
	if len(check_pt_files) == 0:
		return False
	elif len(check_pt_files) == 1:
		return check_pt_files[0]
	else:
		sorted(check_pt_files, key=lambda x: int(''.join(filter(str.isdigit, x))))
		return check_pt_files[-1]


def checkpoint_step(epoch, model, optimizer, path, limit=5):
	# limiter
	_limiter(f'{path}*.pth', limit)
	path = f'{path}_{epoch}.pth'
	torch.save({
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict()
	}, path)


def load_checkpoint(path, model, optimizer=None):
	if not path[-3:] == 'pth':
		path = _get_latest_checkpoint(path)

	if not path:
		raise Exception('no model to checkpoint')

	checkpoint = torch.load(path)

	model.load_state_dict(checkpoint['model_state_dict'])

	if optimizer is not None:
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	# return model, optimizer


class TorchLogReg(torch.nn.Module):
	def __init__(self, input_dim, output_dim):
		super(TorchLogReg, self).__init__()
		self.linear = torch.nn.Linear(input_dim, output_dim)

	def forward(self, x):
		outputs = self.linear(x)
		return outputs


def fit(model, mdl_name, training_act_paths, validation_act_paths, frames_block_cnt, num_procs, epochs, batch_size, TOL, lr
        , log_rate, weight_decay, device):

	l2_weight = 0.01

	training_dataset = ActivationsDataset(training_act_paths, frames_block_cnt, shuffle=True)
	val_dataset = ActivationsDataset(validation_act_paths, frames_block_cnt, shuffle=False)

	train_dataloader = DataLoader(training_dataset, batch_size, shuffle=True, num_workers=num_procs, collate_fn=pack_activations)
	val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_procs, collate_fn=pack_activations)

	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	cross_entropy_loss = torch.nn.CrossEntropyLoss()

	iter = 0
	training_total = DATASET_TRAIN_VID_CNT * frames_block_cnt

	writer = SummaryWriter(logdir=f'/braintree/home/fksato/Projects/models/log_regression/model_summaries'
	                                     f'/{mdl_name}_train')

	checkpoint_path = f'/braintree/home/fksato/Projects/models/log_regression/check_points/{mdl_name}'

	for epoch in tqdm(range(int(epochs)), desc='Train step'):

		train_correct = 0
		train_err_loss = 0
		train_reg_loss = 0

		for i, (stim_paths, activations, labels) in tqdm(enumerate(train_dataloader), desc='Data train'):
			"""
			input_ids_tensor = input_ids_tensor.to(self.device)
			segment_ids_tensor = segment_ids_tensor.to(self.device)
			input_mask_tensor = input_mask_tensor.to(self.device)
			"""
			activations_input = Variable(activations).to(device)
			labels = Variable(labels).to(device)

			optimizer.zero_grad()
			outputs = model(activations_input)

			l2_params = torch.cat([x.view(-1) for x in model.linear2.parameters()])
			l2_regularization = l2_weight * torch.norm(l2_params, 2)

			loss = cross_entropy_loss(outputs, labels)  + l2_regularization
			loss.backward()
			optimizer.step()

			scores, predictions = torch.max(F.softmax(outputs.data, dim=1), 1)
			train_correct += torch.sum(predictions == labels).item()  # labels.size(0) returns int
			train_err_loss += loss.item()
			train_reg_loss += l2_regularization.item()


		training_acc = train_correct / training_total
		training_loss = train_err_loss / training_total
		training_reg_loss = train_reg_loss / training_total

		writer.add_scalar('data/training_acc', training_acc, epoch)
		writer.add_scalar('data/training_loss', training_loss, epoch)
		writer.add_scalar('data/training_reg_loss', training_reg_loss, epoch)

		if training_loss < TOL:
			print(f'Converged with train accuracy: {training_acc:.4f} train error loss: {training_loss:.4f} '
			      f'reg_loss: {training_reg_loss:.4f}\n')
			break

		iter += 1
		if iter % log_rate == 0:
			print(f'Train accuracy: {training_acc:.4f} train error loss: {training_loss:.4f} '
			      f'reg_loss: {training_reg_loss:.4f}\n')

			checkpoint_step(epoch, model, optimizer, checkpoint_path)
			# calculate Accuracy
			val_correct = 0
			val_err_loss = 0
			val_reg_loss = 0
			val_total = 0

			for val_stim_paths, val_activations, val_labels in val_dataloader:
				activations_input = Variable(val_activations).to(device)
				val_labels.to(device)

				outputs = model(activations_input)

				val_l2_params = torch.cat([x.view(-1) for x in model.linear2.parameters()])
				val_l2_regularization = l2_weight * torch.norm(val_l2_params, 2)

				val_loss = cross_entropy_loss(outputs, val_labels) + val_l2_regularization


				_, predicted = torch.max(F.softmax(outputs.data, dim=1), 1)
				val_total += val_labels.size(0)

				val_correct += torch.sum(predicted == val_labels).item()  # labels.size(0) returns int
				val_err_loss += val_loss.item()
				val_reg_loss += val_l2_regularization.item()

			val_acc = val_correct / val_total
			val_loss = val_err_loss / val_total
			val_reg = val_reg_loss / val_total
			print(f'Validation accuracy: {val_acc:.4f} Validation loss: {val_loss:.4f}'
			      f', regularization loss: {val_reg:.4f}\n')

	# export scalar data to JSON for external processing
	writer.export_scalars_to_json("./all_scalars.json")
	writer.close()


def predict(model, mdl_name, testing_act_paths, frames_block_cnt, num_procs, batch_size, device):

	testing_dataset = ActivationsDataset(testing_act_paths, frames_block_cnt, shuffle=False)
	testing_dataloader = DataLoader(testing_dataset, batch_size, shuffle=False, num_workers=num_procs
	                                , collate_fn=pack_activations)

	# test_cnt = DATASET_TEST_VID_CNT * frames_block_cnt
	# preds = np.zeros((test_cnt, 200), dtype=np.float)
	# stim_paths = []
	# labels = []

	for i, (stim_paths, activations, labels) in tqdm(enumerate(testing_dataloader), desc='Data train'):
		activations_input = Variable(activations).to(device)

		outputs = model(activations_input)
		softmax = F.softmax(outputs.data, dim=1)
		_, predicted = torch.max(softmax, 1)

		# preds[batch_start:batch_start + predictions.shape[0]] = predictions
		# stim_paths += list(frame_paths)
		# labels += list(frame_labels)
		# batch_start += predictions.shape[0]




def main(train, mdl_code, weight_decay, batch_size=64, lr=1e-4, num_epoch=1000, keep_prob=1.0, log_rate=10, TOL=1e-4
         , verbose=True, save_load=False, save_name='DEBUG', num_procs=8, checkpoint_name='', *args, **kwargs):

	from math import ceil
	from logistic_regression.logistic_train_predict import confirm
	from utils.mdl_utils import mask_unused_gpus

	gpus_list = mask_unused_gpus()
	use_cuda = torch.cuda.is_available()
	device = torch.device(f"cuda:{gpus_list[0]}" if use_cuda else "cpu")

	model_names = {'0': 'alexnet', '1': 'resnet18', '2': 'resnet34', '3': 'resnet25-18', '4': 'resnet25-34'}
	mdl_frames_blocks = {'0': 60, '1': 60, '2': 60, '3': 1, '4': 1}
	input_size = {'0': 4096, '1': 512, '2': 512, '3': 512, '4': 512}
	num_class = 200

	mdl_name = model_names[mdl_code]
	frames_block_cnt = mdl_frames_blocks[mdl_code]
	feature_size = input_size[mdl_code]
	num_epoch = num_epoch
	batch_size = ceil(
		batch_size / frames_block_cnt) * frames_block_cnt  # guarantee full videos are represented per batch
	log_rate = log_rate
	TOL = TOL

	# activation_paths:
	model_act = f'/braintree/home/fksato/Projects/models/model_data/{mdl_name}'

	print(f'::::::::LOGISTIC REGRESSION::::::::::::')
	print(f':::::::::MODEL INFORMATION:::::::::::::')
	print(f'MODE: {"train" if train else "predict"}')
	print(f'Performance evaluation on {mdl_name}\nframe/block count: {frames_block_cnt}')
	print(f'activation directory: {model_act}')
	print(f'{save_load}')
	if verbose and not confirm():
		return -1

	# /home/deer_meat/mnt/braintree/Projects/models/model_data/alexnet/training_activations/alexnet_training_act_group_0.pkl
	train_sort_start = len(f'{model_act}/training_activations/{mdl_name}_training_act_group_')
	train_group_paths = glob(f'{model_act}/training_activations/*.pkl')
	sorted(train_group_paths, key=lambda name: int(name[train_sort_start:-4]))

	test_sort_start = len(f'{model_act}/testing_activations/{mdl_name}_testing_act_group_')
	test_group_paths = glob(f'{model_act}/testing_activations/*.pkl')
	sorted(test_group_paths, key=lambda name: int(name[test_sort_start:-4]))

	# training_dataset = ActivationsDataset(train_group_paths, frames_block_cnt, shuffle=True)
	# testing_dataset = ActivationsDataset(test_group_paths, frames_block_cnt, shuffle=False)

	model = TorchLogReg(feature_size, num_class)

	model.cuda()
	model.eval()
	model.to(device)

	mdl_save_name = f'{mdl_name}_{save_name}'

	if save_load:
		checkpoint_path = f'/braintree/home/fksato/Projects/models/log_regression/check_points/{mdl_save_name}'
		load_checkpoint(checkpoint_path)

	if train:
		fit(model, mdl_save_name, train_group_paths, test_group_paths, frames_block_cnt, num_procs, num_epoch, batch_size,
		    TOL, lr, log_rate, weight_decay, device)
	else:
		predict(model, mdl_save_name, test_group_paths, frames_block_cnt, num_procs, batch_size, device)


if __name__=='__main__':
	# import argparse
	#
	# parser = argparse.ArgumentParser()
	# parser.add_argument("train", type=lambda x: (str(x).lower() in ['true', '1', 'yes'])
	#                     , help="train model: 1, predict: 0")
	# parser.add_argument("mdl_code", help="0: alexnet, 1: resnet18, 2: resent34, 3: resnet 2.5D 18, 4: resnet 2.5D 34")
	# parser.add_argument("save_load", type=lambda x: (str(x).lower() in ['true', '1', 'yes'])
	#                     , help="1: save/load model for training/predict 0: dont save/load")
	# parser.add_argument("-w", "--weight_decay", dest='weight_decay', default=1e-6
	#                     , help="weight decay for the FC weights. suggested values: 5e-6 for alexnet, 1e-6 for ResNets"
	#                     , type=float)
	# parser.add_argument("-s", "--save_name", dest='save_name', default='DEBUG',
	#                     help='save name of model training/testing')
	# parser.add_argument("-c", "--checkpoint_name", dest='checkpoint_name', default='final',
	#                     help='check point name to load for model for predictions')
	# parser.add_argument("-n", "--num_proc", dest='num_proc', default=8, help="set number of processors", type=int)
	# parser.add_argument("-b", "--batch_size", dest='batch_size', default=64, help="set size of batch", type=int)
	# parser.add_argument("-r", "--lr", dest='lr', default=1e-4, help="learning rate", type=float)
	# parser.add_argument("-e", "--num_epoch", dest='num_epoch', default=1000, type=int)
	# parser.add_argument("-k", "--keep_prob", dest='keep_prob', default=1.0,
	#                     help="keep probability. suggested to be set to 1.0", type=float)
	# parser.add_argument("-l", "--log_rate", dest='log_rate', default=10, type=int)
	# parser.add_argument("-t", "--tolerance", dest='TOL', default=1e-4, type=float)
	# parser.add_argument("-v", "--verbose", dest='verbose', action="store_true", help="verbosity")
	# args = parser.parse_args()
	# main(**vars(args))

	main(True, '1', weight_decay=1e-6, batch_size=4, lr=1e-1, num_epoch=1000, log_rate=10, TOL=1e-4, verbose=False,
			     save_load=False, save_name="_DATA_CHECK_DEBUG", num_procs=4)

	# main(False, '1', weight_decay=1e-6, batch_size=128, lr=1e-1, num_epoch=1000, log_rate=10, TOL=1e-4, verbose=False,
	#      		     save_load=False, save_name="DEBUG", num_procs=4)
"""
vid_dataset = ActivationsDataset(paths, image_size=self.image_size, transform=preprocess)
dataloader = DataLoader(vid_dataset, batch_size=self.batch_size, shuffle=self.shuffle
							 , num_workers=self.num_workers
"""