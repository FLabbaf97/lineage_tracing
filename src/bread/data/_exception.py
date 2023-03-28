"""Define our own exceptions to be able to filter them out easily"""

__all__ = ['BreadException', 'BreadWarning']

class BreadException(UserWarning):
	# inherit from UserWarning, which itself is a subclass of Exception
	# so we can write
	# try:
	# 	...
	# 	raise LineageException(...)
	# except LineageException as e:
	# 	warnings.warn(e)
	pass

class BreadWarning(UserWarning):
	pass