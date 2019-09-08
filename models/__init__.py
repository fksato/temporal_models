import pickle as pk

# load action class files once:

## Moments
class MomentsActionClasses:
	__instance = None
	@staticmethod
	def getInstance():
		if MomentsActionClasses.__instance is None:
			MomentsActionClasses()
		return MomentsActionClasses.__instance

	def __init__(self):
		if MomentsActionClasses.__instance is not None:
			raise Exception("Error: should not happen")
		else:
			MomentsActionClasses.__instance = load_aciton('moments')

## HACS
class HACSActionClasses:
	__instance = None
	@staticmethod
	def getInstance():
		if HACSActionClasses.__instance is None:
			HACSActionClasses()
		return HACSActionClasses.__instance
	def __init__(self):
		if HACSActionClasses.__instance is not None:
			raise Exception("Error: should not happen")
		else:
			HACSActionClasses.__instance = load_aciton('hacs')

# Loader
def load_aciton(dataset):
	with open(f'../{dataset}_action_classes.pk', 'rb') as f:
		_action_class = pk.load(f)
		_action_class = list(_action_class.keys())

	return _action_class


MOMENTS_ACTION_CLASSES = MomentsActionClasses.getInstance()

HACS_ACTION_CLASSES = HACSActionClasses.getInstance()

# Hard-coded video counts
HACS_TOTAL_VID_COUNT = 100372

if __name__=='__main__':
	print(MOMENTS_ACTION_CLASSES)
	print(HACS_ACTION_CLASSES)
	print('Done')