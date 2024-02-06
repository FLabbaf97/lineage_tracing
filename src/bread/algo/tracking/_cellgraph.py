from bread.data import Features, Segmentation, Lineage
import networkx as nx
import pandas as pd
import numpy as np

__all__ = ['build_cellgraph']

def build_cellgraph(feat: Features, time_id: int, cell_features, edge_features, return_df: bool = False, seg: Segmentation = None) -> nx.DiGraph:
    # Extract node features
    cell_ids = feat.segmentation.cell_ids(time_id)
    is_fourier = False # Default
    is_features = True
    if cell_features[0] == 'fourier':
        is_fourier = True
        if(len(cell_features)>3):
            is_featurs = True
            feature_list = cell_features[3]
        fourier_features = []
        num_points = cell_features[1] # Convert to int if necessary
        locality = cell_features[2]
        f_features = ['fourier_' + str(i) for i in range(2 * num_points)]
        fourier_list = f_features
        
        for cell_id in cell_ids:
            area = (feat.segmentation[time_id] == cell_id).astype(np.uint8)
            fourier_features.append(get_fourier_descriptors(area, num_points))
        
        # Convert the list of lists into a NumPy array
        fourier_features = np.array(fourier_features)

        # Create a DataFrame with proper column names
        fourier_df_x = pd.DataFrame({fourier_list[i]: fourier_features[:, i] for i in range(len(fourier_features[0]))})
        fourier_df_x['cell_id'] = cell_ids

    else:
        is_features = True
    if is_features:
        areas, r_equivs, r_mins, r_majs, angles, eccs, x_coordinates, y_coordinates, majs_x, majs_y, mins_x, mins_y = [], [], [], [], [], [], [], [], [], [], [], []
        mean_intensity, std_intensity, correlations , contrasts, energys, homogeneitys = [], [], [], [], [], []
        for cell_id in cell_ids:
            areas.append(feat.cell_area(time_id, cell_id))
            r_equivs.append(feat.cell_r_equiv(time_id, cell_id))
            r_majs.append(feat.cell_r_maj(time_id, cell_id))
            r_mins.append(feat.cell_r_min(time_id, cell_id))
            angles.append(feat.cell_alpha(time_id, cell_id))
            eccs.append(feat.cell_ecc(time_id, cell_id))
            x_coordinates.append(feat._cm(cell_id, time_id)[0])
            y_coordinates.append(feat._cm(cell_id, time_id)[1])
            cell_maj_x, cell_maj_y = feat.cell_maj(cell_id, time_id)
            cell_min_x, cell_min_y = feat.cell_min(cell_id, time_id)
            majs_x.append(cell_maj_x)
            majs_y.append(cell_maj_y)
            mins_x.append(cell_min_x)
            mins_y.append(cell_min_y)
            mean_intensity.append(feat.cell_mean_intensity(time_id, cell_id))
            std_intensity.append(feat.cell_std_intensity(time_id, cell_id))
            correlation , contrast, energy, homogeneity = feat.cell_texture(time_id, cell_id)
            correlations.append(correlation)
            contrasts.append(contrast)
            energys.append(energy)
            homogeneitys.append(homogeneity)

        feature_df_x = pd.DataFrame(dict(
            cell_id=cell_ids,
            area=areas,
            r_equiv=r_equivs,
            r_maj=r_majs,
            r_min=r_mins,
            angel=angles,
            ecc=eccs,
            maj_x=majs_x,
            maj_y=majs_y,
            min_x=mins_x,
            min_y=mins_y,
            x=x_coordinates,
            y=y_coordinates,
            mean_intensity=mean_intensity,
            std_intensity=std_intensity,
            correlation=correlations,
            contrast=contrasts,
            energy=energys,
            homogeneity=homogeneitys,
        ))
    if (is_fourier and is_features) :
        df_x = pd.merge(fourier_df_x, feature_df_x, on='cell_id', how='inner')
        cell_features = fourier_list + feature_list
        assert len(df_x) == len(fourier_df_x) == len(feature_df_x)
    elif is_fourier:
        df_x = fourier_df_x
        cell_features = fourier_list
    elif is_features:
        df_x = feature_df_x
        cell_features = ['cell_id'] + cell_features
    else:
        raise ValueError('No features were selected')

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



import numpy as np
import cv2
import matplotlib.pyplot as plt

# import required libraries
from numpy.linalg import norm


def get_fourier_descriptors(area, n_points, locality=False):
    # Compute the contour of the area
    contours, _ = cv2.findContours(
        area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0]

    normalized_contour = normalize_contour(contour, n_points)
    # Convert normalized_contour points to polar coordinates
    normalized_contour_shape = normalized_contour.shape
    normalized_contour_complex = np.empty(
        (normalized_contour_shape[0],), dtype=complex)
    normalized_contour_complex.real = normalized_contour[:, 0, 0]
    normalized_contour_complex.imag = normalized_contour[:, 0, 1]

    # Compute the Fourier descriptors
    fourier_descriptors = np.fft.fft(normalized_contour_complex)
    if locality == False:
        fourier_descriptors[0] = 0
    
    x_magnitude = abs(fourier_descriptors)
    x_phase = np.angle(fourier_descriptors)        
    # Append the real-valued features to your lists
    x_simple = np.concatenate([x_magnitude, x_phase])

    return x_simple

def normalize_contour(contour, n_points):
    contour_length = contour.shape[0]
    contour_indices = np.linspace(0, contour_length - 1, n_points).astype(int)
    normalized_contour = contour[contour_indices]
    return normalized_contour