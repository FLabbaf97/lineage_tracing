from copy import deepcopy
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.lines, matplotlib.patches
import numpy as np
from bread.data import Segmentation, Microscopy, Features, Ellipse, Contour

__all__ = ['plot_segmentation', 'plot_visible', 'plot_cellids', 'plot_graph', 'plot_debug_pair', 'plot_debug_cell']

FigAx = Tuple[Figure, Axes]

def _unwrap_figax(figax: Optional[FigAx] = None) -> FigAx:
	if figax is None:
		fig, ax = plt.subplots(figsize=(4, 4))
	else:
		fig, ax = figax
	return fig, ax


def plot_cellids(seg: Segmentation, time_id: int, figax: Optional[FigAx] = None, **kwargs) -> FigAx:
	fig, ax = _unwrap_figax(figax)

	cell_ids = seg.cell_ids(time_id)
	cms = seg.cms(time_id)
	for cm, cell_id in zip(cms, cell_ids):
		ax.text(cm[1], cm[0], f'{int(cell_id)}', fontsize=8, ha='center', va='center')

	return fig, ax


def plot_segmentation(seg: Segmentation, time_id: int, figax: Optional[FigAx] = None, cellids: bool = True, **kwargs) -> FigAx:
	fig, ax = _unwrap_figax(figax)
	
	img = seg.data[time_id].astype(float)
	img[img == 0] = np.nan
	ax.imshow(img, cmap='gist_rainbow', **kwargs)

	if cellids:
		plot_cellids(seg, time_id, (fig, ax))
	
	return fig, ax


def plot_visible(mic: Microscopy, time_id: int, figax: Optional[FigAx] = None, **kwargs) -> FigAx:
	fig, ax = _unwrap_figax(figax)

	kwargs_ = dict(cmap='binary')
	kwargs_.update(kwargs)

	ax.imshow(mic.data[time_id], **kwargs_)

	return fig, ax


def plot_graph(seg: Segmentation, time_id: int, edges, figax=None):
	fig, ax = _unwrap_figax(figax)
	
	cell_id_to_cm = { cell_id: cm for cell_id, cm in zip(seg.cell_ids(time_id), seg.cms(time_id)) }
	
	for cell_id1, cell_id2 in edges:
		cm1 = cell_id_to_cm[cell_id1]
		cm2 = cell_id_to_cm[cell_id2]
		ax.add_artist(matplotlib.lines.Line2D(
			(cm1[1], cm2[1]), (cm1[0], cm2[0]),
			color='black', linewidth=1, alpha=0.7
		))
		
	return fig, ax


def _plot_ellipse(el: Ellipse, draw_axes: bool = True, figax: Optional[FigAx] = None, **kwargs) -> FigAx:
	fig, ax = _unwrap_figax(figax)

	rot = np.array(((np.sin(el.angle), -np.cos(el.angle)), (np.cos(el.angle), np.sin(el.angle))))
	yx = np.array([ el.y, el.x ])
	vec_maj = yx + rot @ np.array((el.r_maj, 0))
	vec_min = yx + rot @ np.array((0, el.r_min))
	if draw_axes:
		ax.add_artist(matplotlib.lines.Line2D(
			(yx[1], vec_maj[1]), (yx[0], vec_maj[0]),
			**kwargs
		))
		ax.add_artist(matplotlib.lines.Line2D(
			(yx[1], vec_min[1]), (yx[0], vec_min[0]),
			**kwargs
		))
	ax.add_artist(matplotlib.patches.Ellipse(
		(yx[1], yx[0]), 2*el.r_maj, 2*el.r_min, np.rad2deg(el.angle),
		fill=False, **kwargs
	))

	return fig, ax


def _plot_contour(contour: Contour, figax: Optional[FigAx] = None, **kwargs) -> FigAx:
	fig, ax = _unwrap_figax(figax)

	ax.plot(*contour.data.T, **kwargs)

	return fig, ax


def plot_debug_cell(time_id: int, cell_id: int, feat: Features, figax: Optional[FigAx] = None, **kwargs) -> FigAx:
	fig, ax = _unwrap_figax(figax)

	kwargs = deepcopy(kwargs)
	kwargs['linewidth'] = kwargs.get('linewidth', 1)
	kwargs['color'] = kwargs.get('color', 'black')

	cm = feat._cm(cell_id, time_id)
	contour = feat._contour(cell_id, time_id)
	ellipse = feat._ellipse(cell_id, time_id)

	_plot_ellipse(ellipse, figax=(fig, ax), **kwargs)
	_plot_contour(contour, figax=(fig, ax), **kwargs)
	ax.plot(*cm[[1,0]], '+', **kwargs)

	return fig, ax


def plot_debug_pair(time_id: int, bud_id: int, candidate_id: int, feat: Features, figax: Optional[FigAx] = None, **kwargs) -> FigAx:
	fig, ax = _unwrap_figax(figax)

	cm_bud, cm_candidate = feat._cm(bud_id, time_id)[[1,0]], feat._cm(candidate_id, time_id)[[1,0]]
	contour_bud, contour_candidate = feat._contour(bud_id, time_id), feat._contour(candidate_id, time_id)
	budding_point = feat._budding_point(bud_id, candidate_id, time_id)
	
	kwargs = deepcopy(kwargs)
	kwargs['linewidth'] = kwargs.get('linewidth', 0.5)
	kwargs['color'] = kwargs.get('color', 'black')

	ax.arrow(
		*cm_candidate, *feat.pair_cmtocm(time_id, bud_id, candidate_id),
		length_includes_head=True, head_width=1,
		**kwargs
	)

	ax.arrow(
		*cm_candidate, *feat.pair_budpt(time_id, bud_id, candidate_id),
		length_includes_head=True, head_width=1,
		**kwargs
	)

	ax.plot(*budding_point, marker='*', **kwargs)
	ax.plot(*feat._farthest_point_on_contour(budding_point, contour_bud), marker='x', **kwargs)

	return fig, ax