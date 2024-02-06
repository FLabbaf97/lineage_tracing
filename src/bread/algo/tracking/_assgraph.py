from typing import List, Tuple
import networkx as nx
import numpy as np
import itertools
from torch_geometric.data import Data as TGData
import torch_geometric as tg

__all__ = ['build_assgraph', 'to_node_ids', 'to_data']

TARGET_FEATURE_NAME = 'y'

def build_assgraph(g1: nx.DiGraph, g2: nx.DiGraph, include_target_feature: bool = False) -> nx.DiGraph:
	"""Build assignment graph from cellgraphs g1 and g2

	The assignment graph nodes are a product of g1.nodes and g2.nodes. See ``build_cellgraph``.

	Parameters
	----------
	g1 : nx.DiGraph
		cellgraph 1
	g2 : nx.DiGraph
		cellgraph 2
	include_target_feature : bool
		If True, nodes have an attribute `y = (i == a)` for the training target

	Returns
	-------
	nx.DiGraph
		Assignment graph
	"""
	nodes = list(itertools.product(g1.nodes, g2.nodes))
	nodes_attr = {}
	for i, a in nodes:
		i_attr = { f'{k}_1': v for k, v in g1.nodes[i].items() }
		a_attr = { f'{k}_2': v for k, v in g2.nodes[a].items() }
		d = {**i_attr, **a_attr}
		if include_target_feature:
			d[TARGET_FEATURE_NAME] = int(i == a)
		nodes_attr[(i, a)] = d

	edges = []
	edges_attr = {}
	for ij, ab in itertools.product(g1.edges, g2.edges):
		i, j = ij
		a, b = ab
		edge = ((i, a), (j, b))
		ij_attr = { f'{k}_1': v for k, v in g1.edges[ij].items() }
		ab_attr = { f'{k}_2': v for k, v in g2.edges[ab].items() }
		edges.append(edge)
		edges_attr[edge] = {**ij_attr, **ab_attr}

	graph = nx.DiGraph()
	# ensure all nodes, even isolated, are added IN ORDER to the graph
	# this is needed so I can do unravel_multi_index later
	graph.add_nodes_from(nodes)
	graph.add_edges_from(edges)
	nx.set_node_attributes(graph, nodes_attr)
	nx.set_edge_attributes(graph, edges_attr)
	
	return graph


def to_data(ga: nx.DiGraph, include_target_feature: bool = True, complex_graph: bool = False) -> Tuple[TGData, List[str], List[str]]:
	"""Convert assignment graph generated from ``build_assgraph`` to ``torch_geometric.data.Data``

	Parameters
	----------
	ga : nx.DiGraph
		Assignment graph
	include_target_feature : bool
		If True, include the target as a graph attribute `tga.y`. This assumes the assignment graph has been generated with `include_target_feature = True`

	Returns
	-------
	Tuple[TGData, List[str], List[str]]
		Data object, node attribute names, edge attribute names
	"""

	node_attrs = list(ga.nodes[next(iter(ga.nodes))].keys())

	# do not include the target in the training data !
	if TARGET_FEATURE_NAME in node_attrs:
		node_attrs.remove(TARGET_FEATURE_NAME)

	if len(ga.edges) > 0:
		edge_attrs = list(ga.edges[next(iter(ga.edges))].keys())
		tga = tg.utils.convert.from_networkx(ga, group_node_attrs=node_attrs, group_edge_attrs=edge_attrs)
	else:
		edge_attrs = []
		tga = tg.utils.convert.from_networkx(ga, group_node_attrs=node_attrs)

	if include_target_feature:
		gt = tg.utils.convert.from_networkx(ga, group_node_attrs=[TARGET_FEATURE_NAME])
		tga.y = gt.x

	# include the cell ids in the graph, so we can conveniently reshape afterwards
	if (complex_graph):
		tga.cell_ids1 = set([ i for i, a in ga.nodes ])
		tga.cell_ids2 = set([ a for i, a in ga.nodes ])
	else:
		tga.cell_ids = set([i for i in ga.nodes])
	

	return tga, node_attrs, edge_attrs


def to_node_ids(assnode_idx: int, g1: nx.DiGraph, g2: nx.DiGraph) -> Tuple[int, int]:
	"""Convert a node index in the assignment graph to (cell_id1, cell_id2)

	Parameters
	----------
	assnode_idx : int
		Node index in the pytorch assignment graph
	g1 : nx.DiGraph
		Cellgraph 1
	g2 : nx.DiGraph
		Cellgraph 2

	Returns
	-------
	Tuple[int, int]
		(cell_id1, cell_id2)
	"""
	cell_idx1, cell_idx2 = np.unravel_index(assnode_idx, (g1.number_of_nodes(), g2.number_of_nodes()))
	return (list(g1.nodes)[cell_idx1], list(g2.nodes)[cell_idx2])