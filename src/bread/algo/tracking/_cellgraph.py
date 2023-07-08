from bread.data import Features, Segmentation, Lineage
import networkx as nx
import pandas as pd
import numpy as np

__all__ = ['build_cellgraph']

def build_cellgraph(feat: Features, time_id: int, cell_features, edge_features, return_df: bool = False) -> nx.DiGraph:
    # Extract node features
    cell_ids = feat.segmentation.cell_ids(time_id)
    areas, r_equivs, r_mins, r_majs, angles, eccs, x_coordinates, y_coordinates = [], [], [], [], [], [], [], []
    for cell_id in cell_ids:
        areas.append(feat.cell_area(time_id, cell_id))
        r_equivs.append(feat.cell_r_equiv(time_id, cell_id))
        r_majs.append(feat.cell_r_maj(time_id, cell_id))
        r_mins.append(feat.cell_r_min(time_id, cell_id))
        angles.append(feat.cell_alpha(time_id, cell_id))
        eccs.append(feat.cell_ecc(time_id, cell_id))
        x_coordinates.append(feat._cm(cell_id, time_id)[0])
        y_coordinates.append(feat._cm(cell_id, time_id)[1])

    df_x = pd.DataFrame(dict(
        cell_id=cell_ids,
        area=areas,
        r_equiv=r_equivs,
        r_maj=r_majs,
        r_min=r_mins,
        angle=angles,
        ecc=eccs,
        x=x_coordinates,
        y=y_coordinates,
    ))

    # Extract edge features
    pairs = set()
    for cell_id in cell_ids:
        nn_ids = feat._nearest_neighbours_of(time_id, cell_id)
        for nn_id in nn_ids:
            pairs.add((cell_id, nn_id))

    df_e = pd.DataFrame(pairs, columns=['cell_id1', 'cell_id2']).sort_values(['cell_id1', 'cell_id2']).reset_index(drop=True)
    cmtocm_xs, cmtocm_ys, cmtocm_rhos, cmtocm_angles, majmaj_angle, contour_dists = [], [], [], [], [], []
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
    df_e['majmaj_angle'] = majmaj_angle


    # Create the graph
    graph = nx.DiGraph()
    graph.add_nodes_from(cell_ids)
    graph.add_edges_from(df_e[['cell_id1', 'cell_id2']].to_numpy())

    # Set node attributes
    if 'cell_id' not in cell_features:
    	cell_features.append('cell_id')
    node_attributes = df_x[cell_features].set_index('cell_id').to_dict(orient='index')
    nx.set_node_attributes(graph, node_attributes)

    # Set edge attributes
    if 'cell_id1' not in edge_features:
    	edge_features.append('cell_id1')
    if 'cell_id2' not in edge_features:
        edge_features.append('cell_id2')
    edge_attributes = df_e[edge_features].set_index(['cell_id1', 'cell_id2']).to_dict(orient='index')
    nx.set_edge_attributes(graph, edge_attributes)

    if return_df:
        return graph, df_x, df_e
    else:
        return graph
