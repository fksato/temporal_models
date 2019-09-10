import copy
from collections import OrderedDict

import numpy as np
from tqdm import tqdm

from model_tools.activations.core import ActivationsExtractorHelper

from brainio_base.assemblies import NeuroidAssembly, walk_coords, merge_data_arrays
from model_tools.activations.core import flatten

class VideoActivationsExtractorHelper(ActivationsExtractorHelper):

	def __init__(self, data_inputs=None, batch_size=64, *args, **kwargs):

		super(VideoActivationsExtractorHelper, self).__init__(batch_size=batch_size, *args, **kwargs)
		# TODO: remove datainputs dependencies
		self.data_inputs = data_inputs

	def _from_paths(self, layers, stimuli_paths):
		"""
		:param layers: layers to stimulate (type: list)
		:param stimuli_paths: paths to stimulus (type: list)
		:return:
		"""
		# create inputs from paths:
		self.data_inputs.make_from_paths(stimuli_paths)
		total_data_len = len(stimuli_paths) * self.data_inputs.units
		return self._get_activations_batched(total_data_len, layers, self._batch_size)

	def _get_activations_batched(self, total_data_len, layers, batch_size):
		"""
		Same as in model-tools, overrides call to self._get_batch_activations
		self.get_activations batched activations automatically
		if we can get the size, we can allocate Layer_activations before
		=> no need for merge data
		:param path_len:
		:param layers:
		:param batch_size:
		:return:
		"""
		layer_activations = None
		stimulus_paths = []
		for batch_start in tqdm(range(0, total_data_len, batch_size), unit_scale=batch_size, desc="activations"):
			try:
				stim_paths, batch_activations = self.get_activations(layer_names=layers)
			except Exception as e:
				print(e)
				break

			assert isinstance(batch_activations, OrderedDict)
			for hook in self._batch_activations_hooks.copy().values():  # copy to avoid handle re-enabling messing with the loop
				batch_activations = hook(batch_activations)

			stimulus_paths += list(stim_paths)
			if layer_activations is None:
				layer_activations = OrderedDict({layer: np.empty((total_data_len, *batch_activations[layer].shape[1:]))*np.nan
				                                 for layer in layers})
				for layer_name, layer_output in batch_activations.items():
					subset_idx = np.arange(batch_start, batch_start+layer_output.shape[0], 1)
					layer_activations[layer_name][subset_idx] = layer_output
			else:
				for layer_name, layer_output in batch_activations.items():
					subset_idx = np.arange(batch_start, batch_start + layer_output.shape[0], 1)
					layer_activations[layer_name][subset_idx] = layer_output

		# drop any empty rows (useful for when videos are not all consistent sizes)
		for key, val in layer_activations.items():
			val = np.squeeze(val)
			layer_activations[key] = val[~np.isnan(val).any(axis=1)]

		return self._package(layer_activations, stimulus_paths)


	def _package_layer(self, layer_activations, layer, stimuli_paths):
		"""
		copy of _package_layer, added '4' as an allowed shape to account for temporal models in assert
		:param layer_activations:
		:param layer:
		:param stimuli_paths:
		:return:
		"""
		print(layer_activations.shape[0], len(stimuli_paths))
		assert layer_activations.shape[0] == len(stimuli_paths)
		activations, flatten_indices = flatten(layer_activations, return_index=True)  # collapse for single neuroid dim
		assert flatten_indices.shape[1] in [1, 3, 4]  # either convolutional or fully-connected
		flatten_coord_names = ['channel', 'channel_x', 'channel_y']
		flatten_coords = {flatten_coord_names[i]: [sample_index[i] if i < flatten_indices.shape[1] else np.nan
		                                           for sample_index in flatten_indices]
		                  for i in range(len(flatten_coord_names))}
		layer_assembly = NeuroidAssembly(
			activations,
			coords={**{'stimulus_path': stimuli_paths,
			           'neuroid_num': ('neuroid', list(range(activations.shape[1]))),
			           'model': ('neuroid', [self.identifier] * activations.shape[1]),
			           'layer': ('neuroid', [layer] * activations.shape[1]),
			           },
			        **{coord: ('neuroid', values) for coord, values in flatten_coords.items()}},
			dims=['stimulus_path', 'neuroid']
		)
		neuroid_id = [".".join([f"{value}" for value in values]) for values in zip(*[
			layer_assembly[coord].values for coord in ['model', 'layer', 'neuroid_num']])]
		layer_assembly['neuroid_id'] = 'neuroid', neuroid_id
		return layer_assembly


if __name__=="__main__":
	from models.pytorch_model import pytorch_model
	from models import HACS_ACTION_CLASSES as action_classes

	batch = 2
	num_procs = 4

	#### MDL-specific
	mdl_name = 'resnet18'
	layers = ['avgpool']
	imsize = 224
	frames_per_video = 75
	video_offset = 0
	####

	vid_paths = ['/braintree/home/fksato/temp/test/testing/action/_6QHqZr734U_1.mp4'
		, '/braintree/home/fksato/temp/test/testing/action/-3dWoak69HM_167.mp4'
		, '/braintree/home/fksato/temp/test/testing/action/-90Qh6fmV-w_326.mp4'
		, '/braintree/home/fksato/temp/test/testing/action/-90Qh6fmV-w_326.mp4']

	# vid_cnts, vid_paths = get_vid_paths(actions_list=action_classes, main_vid_dir=main_vid_dir)
	di_args = {'batch_size': batch, 'shuffle': False, 'num_workers': num_procs, 'image_size': imsize
	           , 'frames_per_video': frames_per_video, 'video_offset': video_offset}
	test = pytorch_model(mdl_name, **di_args)

	a = test(vid_paths, layers)

