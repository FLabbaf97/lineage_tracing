from matplotlib import cm, colors, pyplot as plt, ticker
import numpy as np

__all__ = ['plot_parallel']

def plot_parallel(df, shift: float = 0.05, alpha=0.5):
	data = df.to_numpy()
	labels = df.columns
	cmap = cm.get_cmap('RdYlGn')
	cnorm = colors.LogNorm(vmin=data[:, -1].min(), vmax=data[:, -1].max())

	fig, host = plt.subplots(figsize=(8, 4))
	ax = [host] + [host.twinx() for i in range(data.shape[1]-1)]
	scale = data.max(axis=0) - data.min(axis=0)

	for i, a in enumerate(ax):
		a.set_ylim(data[:, i].min()-scale[i]*shift, data[:, i].max()+scale[i]*shift)
		a.spines['top'].set_visible(False)
		a.spines['bottom'].set_visible(False)
		if ax != host:
			a.spines['left'].set_visible(False)
			a.yaxis.set_ticks_position('right')
			a.spines["right"].set_position(("axes", i / (data.shape[1]-1)))

	host.set_xlim(0, data.shape[1]-1)
	host.set_xticks(range(data.shape[1]))
	host.set_xticklabels(labels)
	host.tick_params(axis='x', which='major', pad=7)

	argsort = np.argsort(data[:, -1])[::-1]
	data = data[argsort]
	# add some noise to the data to we can better see what's going on
	for i in range(len(scale)):
		# print(scale)
		data[:, i] += np.linspace(-shift, shift, len(data[:, i]))*scale[i]

	Z = np.zeros_like(data)
	Z[:, 0] = data[:, 0]
	Z[:, 1:] = (data[:, 1:] - data[:, 1:].min(axis=0)) / (data[:, 1:].max(axis=0) - data[:, 1:].min(axis=0)) * (data[:, 0].max() - data[:, 0].min()) + data[:, 0].min()

	np.random.seed(42)
	for idx in range(data.shape[0]):

		host.plot(range(data.shape[1]), Z[idx, :],
			color=cmap(1-cnorm(data[idx, -1])),
			zorder=-1, linewidth=1, alpha=alpha,
		)
		
	return ax