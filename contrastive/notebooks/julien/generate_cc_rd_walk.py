import numpy as np
from scipy import ndimage
import os
import pandas as pd
import networkx as nx
from collections import defaultdict
import random
from soma import aims
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

walk_length = 2000
alpha = 0.5
structure = np.ones((3,3,3))

n_cpus= 46
side = 'R'
datasets = ['ACCpatterns']
sulcus_list = ['LARGE_CINGULATE.']

## UTILS

def find_connected_components(volume, structure):
    """Label connected components in a 3D binary volume."""
    labeled_volume, num_components = ndimage.label(volume, structure=structure)  # Label each connected component
    return labeled_volume, num_components

def voxel_graph(component, label_value):
    """Create a graph from a single connected component."""
    G = nx.Graph()
    dims = component.shape
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                if component[i, j, k] == label_value:
                    G.add_node((i, j, k))
                    #for di, dj, dk in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]: ## not the right connectivity ?
                    for di in range(-1,2,1):
                        for dj in range(-1,2,1):
                            for dk in range(-1,2,1):
                                ni, nj, nk = i + di, j + dj, k + dk
                                if 0 <= ni < dims[0] and 0 <= nj < dims[1] and 0 <= nk < dims[2] and not (ni==i and nj==j and nk==k): # and not (np.linalg.norm((ni-i, nj-j, nk-k))>1.5) # use connectivity 26
                                    if component[ni, nj, nk] == label_value:
                                        G.add_edge((i, j, k), (ni, nj, nk), weight=np.linalg.norm((ni-i, nj-j, nk-k)))
    return G

def random_walk_all_points(graph, walk_length=20, alpha=0.5):
    """Simulates random walks from every point in the graph to measure accessibility."""
    visit_counts = defaultdict(int)
    
    for start_point in graph.nodes:
        current_point = start_point
        
        for _ in range(walk_length):
            visit_counts[current_point] += 1  # Count visit

            neighbors = list(graph.neighbors(current_point))
            if not neighbors:
                break  # Dead end

            # Move to a random neighbor
            current_point = random.choice(neighbors)
    
    # Normalize visit counts
    max_visits = max(visit_counts.values()) if visit_counts else 1
    for node in graph.nodes:
        #visit_counts[node] = (1-visit_counts[node]/max_visits)*alpha + alpha if node in visit_counts else alpha ## minimum on edges
        visit_counts[node] = (1-alpha)*(1-visit_counts[node]/max_visits) + alpha if node in visit_counts else 1 ## maximum on edges

    return visit_counts


## Processing

def process_element(subject_dir, root_dir, save_dir, walk_length=1000, alpha=0.5, structure=np.ones((3,3,3))):
    skel_volume = aims.read(os.path.join(root_dir, subject_dir))
    skel_volume.np[skel_volume.np!=0]=1
    skel = skel_volume.np
    volume = skel[:,:,:,0]!=0

    # Find connected components
    labeled_volume, num_components = find_connected_components(volume, structure=structure)

    # random walks on each connected component
    results = defaultdict(dict)
    for label_value in range(1, num_components + 1):  # Labels start from 1
        G = voxel_graph(labeled_volume, label_value)
        visit_frequencies = random_walk_all_points(G, walk_length=walk_length, alpha=alpha)
        results = {**results, **visit_frequencies}

    ccrdwalk_skel = np.zeros(skel.shape, dtype=float)
    for point, freq in results.items():
        ccrdwalk_skel[point] = freq
    
    vol = ccrdwalk_skel.astype(np.float32)
    vol = aims.Volume(vol)

    aims.write(vol, os.path.join(save_dir, subject_dir))
    
    return vol.np, subject_dir


def process_list(root_dir, save_dir, walk_length, alpha, structure, list_sub_dirs, n_cpus):
    # Create a partial function with y and z fixed
    partial_func = partial(process_element, root_dir=root_dir, save_dir=save_dir, walk_length=walk_length, alpha=alpha, structure=structure)
    
    # Create a pool of workers with n_cpus
    with mp.Pool(processes=n_cpus) as pool:
        # Map the partial function to the elements list
        results = list(tqdm(pool.imap(partial_func, list_sub_dirs), total=len(list_sub_dirs)))
    return results


for sulcus in sulcus_list:
    for dataset in datasets:

        print(f'Treating {dataset} {sulcus} {side}')
        print('Build niftis')

        root_dir = f'/neurospin/dico/data/deep_folding/current/datasets/{dataset}/crops/2mm/{sulcus}/mask/'
        save_dir = os.path.join(root_dir, f'{side}ccrdwalks')
        crops_dir = os.path.join(root_dir, f'{side}crops')
        subjects_list = os.listdir(crops_dir)
        subjects_list = [elem for elem in subjects_list if elem[-1]!='f'] # remove minf files from list

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        results = process_list(crops_dir, save_dir, walk_length=walk_length, alpha=alpha, structure=structure, list_sub_dirs=subjects_list, n_cpus=n_cpus)
        subjects = [results[k][1].split('_cropped_skeleton.nii.gz')[0] for k in range(len(results))]
        print(f'Build Numpy array')
        results = np.stack([results[k][0] for k in range(len(results))])
        subjects = pd.DataFrame(data=subjects, columns=['Subject'])
        ## sort subjects by name
        subjects = subjects.sort_values(by='Subject')
        idxs = subjects.index.tolist()
        # reorder crops
        results = results[idxs]
        np.save(os.path.join(root_dir, f'{side}ccrdwalks.npy'), results)
        subjects.to_csv(os.path.join(root_dir, f'{side}ccrdwalks_subjects.csv'), index=False)
