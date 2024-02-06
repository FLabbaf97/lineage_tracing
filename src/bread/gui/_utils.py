def clamp(x, a, b):
	return max(a, min(x, b))

def lerp(x, a1, b1, a2, b2):
	return (x - a1)/(b1 - a1)*(b2 - a2) + a2