from bread.data import Features, Segmentation, Lineage
import networkx as nx
import pandas as pd
import numpy as np

__all__ = ['build_cellgraph']

def build_cellgraph(feat: Features, time_id: int, return_df: bool = False) -> nx.DiGraph:
	"""Build a cellgraph given a ``Features`` object for a given frame

	The returned cellgraph has sorted node ids corresponding to cell ids in the segmentation (see ``Segmentation.cell_ids``).
	Edges are placed between nodes according to ``Features._nearest_neighbours_of``, which can be configured from ``Features.nn_threshold``

	Parameters
	----------
	feat : Features
	time_id : int
		Frame index in the segmentation
	return_df : bool, optional
		Return the node and edge feature dataframes, by default False

	Returns
	-------
	nx.DiGraph
		Cellgraph
	"""

	# Extract node features
	cell_ids = feat.segmentation.cell_ids(time_id)
	areas, r_equivs, r_mins, r_majs, angles, eccs = [], [], [], [], [], []
	for cell_id in cell_ids:
		areas.append(feat.cell_area(time_id, cell_id))
		r_equivs.append(feat.cell_r_equiv(time_id, cell_id))
		r_majs.append(feat.cell_r_maj(time_id, cell_id))
		r_mins.append(feat.cell_r_min(time_id, cell_id))
		angles.append(feat.cell_alpha(time_id, cell_id))
		eccs.append(feat.cell_ecc(time_id, cell_id))
	
	df_x = pd.DataFrame(dict(
		cell_id=cell_ids,
		area=areas,
		r_equiv=r_equivs,
		r_maj=r_majs,
		r_min=r_mins,
		angle=angles,
		ecc=eccs
	))

	# Extract edge features
	# The edges are bidirectional, i.e. we have (i, j) and (j, i)
	pairs = set()  # using a set prevents duplicates
	for cell_id in cell_ids:
		nn_ids = feat._nearest_neighbours_of(time_id, cell_id)
		for nn_id in nn_ids:
			pairs.add((cell_id, nn_id))

	df_e = pd.DataFrame(pairs, columns=['cell_id1', 'cell_id2'])\
		.sort_values(['cell_id1', 'cell_id2']).reset_index(drop=True)
	cmtocm_xs, cmtocm_ys, cmtocm_rhos, cmtocm_angles, contour_dists, majmaj_angle = [], [], [], [], [], []
	for _, row in df_e.iterrows():
		cmtocm = feat.pair_cmtocm(time_id, row.cell_id1, row.cell_id2)
		x, y = cmtocm[0], cmtocm[1]
		rho = np.sqrt(x**2 + y**2)
		theta = np.arctan2(y, x)
		cmtocm_xs.append(x)
		cmtocm_ys.append(y)
		cmtocm_rhos.append(rho)
		cmtocm_angles.append(theta)
		contour_dists.append(feat.pair_dist(time_id, row.cell_id1, row.cell_id2))
		majmaj_angle.append(feat.pair_majmaj_angle(time_id, row.cell_id1, row.cell_id2))

	df_e['cmtocm_x'] = cmtocm_xs
	df_e['cmtocm_y'] = cmtocm_ys
	df_e['cmtocm_len'] = cmtocm_rhos
	df_e['cmtocm_angle'] = cmtocm_angles
	df_e['contour_dist'] = contour_dists

	# WARNING : this is incorrect, as the nodes (cell_ids) are not sorted !
	# graph = nx.from_pandas_edgelist(df_e,
	# 	source='cell_id1', target='cell_id2', edge_attr=True,
	# 	create_using=nx.DiGraph
	# )
	graph = nx.DiGraph()
	graph.add_nodes_from(cell_ids)
	graph.add_edges_from(df_e[['cell_id1', 'cell_id2']].to_numpy())
	nx.set_node_attributes(graph, df_x.set_index('cell_id').to_dict(orient='index'))
	nx.set_edge_attributes(graph, df_e.set_index(['cell_id1', 'cell_id2']).to_dict(orient='index'))

	if return_df:
		return graph, df_x, df_e
	else:
		return graph