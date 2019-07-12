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
		:param path_len:
		:param layers:
		:param batch_size:
		:return:
		"""
		layer_activations = None
		for batch_start in tqdm(range(0, total_data_len, batch_size), unit_scale=batch_size, desc="activations"):
			try:
				stim_paths, batch_activations = self.get_activations(layer_names=layers)
			except:
				break
			assert isinstance(batch_activations, OrderedDict)
			for hook in self._batch_activations_hooks.copy().values():  # copy to avoid handle re-enabling messing with the loop
				batch_activations = hook(batch_activations)

			if layer_activations is None:
				layer_activations = copy.copy(batch_activations)
				for layer_name, layer_output in batch_activations.items():
					layer_activations[layer_name] = self._package_layer(layer_output, layer_name, stim_paths)
			else:
				for layer_name, layer_output in batch_activations.items():
					layer_output_pkg = self._package_layer(layer_output, layer_name, stim_paths)
					layer_activations[layer_name] = merge_data_arrays((layer_activations[layer_name], layer_output_pkg))

		# return stim_paths, layer_activations
		return self._package(layer_activations, stimuli_paths)

	def _package(self, layer_activations, stimuli_paths):
		layer_assemblies = [layer_activations_assemblies for layer, layer_activations_assemblies
		                    in layer_activations.items()]

		# merge manually instead of using merge_data_arrays since `xarray.merge` is very slow with these large arrays
		self._logger.debug("Merging layer assemblies")
		model_assembly = np.concatenate([a.values for a in layer_assemblies],
		                                axis=layer_assemblies[0].dims.index('neuroid'))
		nonneuroid_coords = {coord: (dims, values) for coord, dims, values in walk_coords(layer_assemblies[0])
		                     if set(dims) != {'neuroid'}}
		neuroid_coords = {coord: [dims, values] for coord, dims, values in walk_coords(layer_assemblies[0])
		                  if set(dims) == {'neuroid'}}
		for layer_assembly in layer_assemblies[1:]:
			for coord in neuroid_coords:
				neuroid_coords[coord][1] = np.concatenate((neuroid_coords[coord][1], layer_assembly[coord].values))
			assert layer_assemblies[0].dims == layer_assembly.dims
			for dim in set(layer_assembly.dims) - {'neuroid'}:
				for coord in layer_assembly[dim].coords:
					assert (layer_assembly[coord].values == nonneuroid_coords[coord][1]).all()
		neuroid_coords = {coord: (dims_values[0], dims_values[1])  # re-package as tuple instead of list for xarray
		                  for coord, dims_values in neuroid_coords.items()}
		model_assembly = type(layer_assemblies[0])(model_assembly, coords={**nonneuroid_coords, **neuroid_coords},
		                                           dims=layer_assemblies[0].dims)
		return model_assembly

	def _package_layer(self, layer_activations, layer, stimuli_paths):
		"""
		copy of _package_layer, added '4' as an allowed shape to account for temporal models in assert
		:param layer_activations:
		:param layer:
		:param stimuli_paths:
		:return:
		"""
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

	#TODO: def _package(self, layer_activations, stimuli_paths) may be a performance issue for very large datasets
	#TODO: Idea: merge activations after each batch in get_activations_batch

