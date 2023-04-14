from typing import Any
import time

__all__ = ['concat_dict', 'Timer']

def concat_dict(d1: 'dict[Any, list]', d2: 'dict[Any, list]'):
	for key in d1:
		d1[key] += d2[key]

class Timer:
	def __init__(self, name = ''):
		self.name = name

	def __enter__(self):
		self.time_start = time.perf_counter()
		return self

	def __exit__(self, *exc_info):
		print(f'"{self.name}" finished in {time.perf_counter()-self.time_start:.2f}s')