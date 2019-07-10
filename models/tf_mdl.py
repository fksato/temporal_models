from r3Dconv import R25DNet_V1
from data_inputs.tf_data_inputs import TF_di
from activations.tensorflow import TensorflowVideowrapper
from brainscore.utils import fullname

class Default3DArgs:
	depth = 18  # 18, 34
	t_depth = 8  # 8, 16, 32
	dataset = 'kinetics'  # 'kinetics', 'kinetics+sports1m'
	model_name = f'r2_1-{depth}_l{t_depth}-{dataset}'
	batch_size = 1
	weight_decay_rate = 0.05


class DefaultGPUConfig:
	import tensorflow as tf
	gpu_config = tf.GPUOptions(allow_growth=True)

#TODO: replace "placeholder" with iterator.get_next() to avoid using feed_dict
class TensorflowR3DModel:
	@staticmethod
	def init(identifier, net_name, model_ctr_kwargs=None, data_input_kwargs=None):
		import tensorflow as tf
		placeholder = tf.placeholder(dtype=tf.float32
		                             , shape=[None, data_input_kwargs['t_window']
									 , data_input_kwargs['im_height']
		                             , data_input_kwargs['im_width']
		                             , 3])

		model = R25DNet_V1(**model_ctr_kwargs)
		_, endpoints = model.build_graph(placeholder, train_mode=False)

		net_name = net_name or identifier

		session = tf.Session(config=tf.ConfigProto(gpu_options=DefaultGPUConfig.gpu_config))
		session.run(tf.global_variables_initializer())
		data_input_kwargs['session'] = session
		#TODO: remove datainputs dependencies
		data_inputs = TF_di(**data_input_kwargs)
		TensorflowR3DModel._restore_weights(net_name, session)
		wrapper = TensorflowVideowrapper(data_inputs=data_inputs, identifier=identifier
		                                 , inputs=placeholder, endpoints=endpoints, session=session)
		return wrapper

	@staticmethod
	def _restore_weights(name, session):
		import tensorflow as tf
		restorer = tf.train.Saver()
		restorer.restore(session, f'/braintree/home/fksato/Projects/models/mdl_restore/{name}.ckpt')


#TODO: replace "placeholder" with iterator.get_next() to avoid using feed_dict
#TODO: inherit from TensorflowSlimModel since the only difference is init()
class TensorflowSlimVidModel:
	@staticmethod
	def init(identifier, net_name=None, labels_offset=1, model_ctr_kwargs=None, data_input_kwargs=None):
		import tensorflow as tf
		from nets import nets_factory

		placeholder = tf.placeholder(dtype=tf.float32
		                             , shape=[None
									 , data_input_kwargs['im_height']
		                             , data_input_kwargs['im_width']
		                             , 3])

		net_name = net_name or identifier
		model_ctr = nets_factory.get_network_fn(net_name, num_classes=labels_offset + 1000, is_training=False)
		logits, endpoints = model_ctr(placeholder, **(model_ctr_kwargs or {}))
		if 'Logits' in endpoints:  # unify capitalization
			endpoints['logits'] = endpoints['Logits']
			del endpoints['Logits']

		session = tf.Session(config=tf.ConfigProto(gpu_options=DefaultGPUConfig.gpu_config))
		# session = tf.Session()
		session.run(tf.global_variables_initializer())
		data_input_kwargs['session'] = session
		# TODO: remove datainputs dependencies
		data_inputs = TF_di(**data_input_kwargs)
		TensorflowSlimVidModel._restore_imagenet_weights(identifier, session)
		wrapper = TensorflowVideowrapper(data_inputs=data_inputs, identifier=identifier
		                                 , inputs=placeholder, endpoints=endpoints, session=session)
		return wrapper

	@staticmethod
	def _restore_imagenet_weights(name, session):
		import tensorflow as tf
		var_list = None
		if name.startswith('mobilenet'):
			# Restore using exponential moving average since it produces (1.5-2%) higher accuracy according to
			# https://github.com/tensorflow/models/blob/a6494752575fad4d95e92698dbfb88eb086d8526/research/slim/nets/mobilenet/mobilenet_example.ipynb
			ema = tf.train.ExponentialMovingAverage(0.999)
			var_list = ema.variables_to_restore()
		restorer = tf.train.Saver(var_list)

		restore_path = TensorflowSlimVidModel._find_model_weights(name)
		restorer.restore(session, restore_path)

	@staticmethod
	def _find_model_weights(model_name):
		import os
		import glob
		import logging
		from candidate_models import s3
		_logger = logging.getLogger(fullname(TensorflowSlimVidModel._find_model_weights))
		framework_home = os.path.expanduser(os.getenv('CM_HOME', '~/.candidate_models'))
		weights_path = os.getenv('CM_TSLIM_WEIGHTS_DIR', os.path.join(framework_home, 'model-weights', 'slim'))
		model_path = os.path.join(weights_path, model_name)
		if not os.path.isdir(model_path):
			_logger.debug(f"Downloading weights for {model_name} to {model_path}")
			os.makedirs(model_path)
			s3.download_folder(f"slim/{model_name}", model_path)
		fnames = glob.glob(os.path.join(model_path, '*.ckpt*'))
		assert len(fnames) > 0, f"no checkpoint found in {model_path}"
		restore_path = fnames[0].split('.ckpt')[0] + '.ckpt'
		return restore_path
