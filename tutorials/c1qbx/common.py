from typing import List
import numpy as np
import scipy.linalg
import scipy.spatial
import scipy.interpolate
import sympy as sp
import matplotlib.pyplot as plt
from dataclasses import dataclass


def qbx_center_setup(src_surfs, directions=None, mult=0.5, singularities=None):
    """
    Determine the ideal locations for QBX expansion centers for several
    surfaces.

    src_surfs: The list of source surfaces.

    directions: A list equal in length to src_surfs specifying whether
        to expand on the positive (1.0) or negative (-1.0) side of the surface. The
        positive side is the side in the direction of the normal vector.
        If you want to expand on both sides, simply pass the source surface
        twice and specify 1.0 once and -1.0 once.

    mult: The default panel length multiplier for how far from the surface to offset
        the expansion centers.

    p: The order of the QBX expansions.
    """
    if directions is None:
        directions = [1.0 for i in range(len(src_surfs))]

    proc_directions = []
    proc_src_surfs = []
    for i in range(len(src_surfs)):
        s = src_surfs[i]
        d = directions[i]
        if d == 0:
            proc_directions += [-1, 1]
            proc_src_surfs += [s, s]
        else:
            proc_directions.append(d)
            proc_src_surfs.append(s)
        
    src_trees = []
    for surf in proc_src_surfs:
        src_trees.append(scipy.spatial.KDTree(surf.pts))
    if singularities is not None:
        singularity_tree = scipy.spatial.KDTree(singularities)
        
    all_centers = []
    all_rs = []
    for i, surf in enumerate(proc_src_surfs):
        r = mult * np.repeat(surf.panel_length, surf.panel_order)
        offset = proc_directions[i] * r

        max_iter = 40
        for j in range(max_iter):
            centers = surf.pts + offset[:, None] * surf.normals
            which_violations = np.zeros(centers.shape[0], dtype=bool)
            for t in src_trees:
                dist_to_nearest_panel = t.query(centers)[0]
                # The fudge factor helps avoid numerical precision issues. For example,
                # when we offset an expansion center 1.0 away from a surface node,
                # without the fudge factor this test will be checking 1.0 < 1.0, but
                # that is fragile in the face of small 1e-15 sized numerical errors.
                # By simply multiplying by 1.0001, we avoid this issue without
                # introducing any other problems.
                fudge_factor = 1.0001
                which_violations |= dist_to_nearest_panel * fudge_factor < np.abs(
                    offset
                )
            
            if singularities is not None:
                dist_to_singularity, which_singularity = singularity_tree.query(centers)
                which_violations |= dist_to_singularity <= 4 * offset
                #import ipdb;ipdb.set_trace()

            if not which_violations.any():
                break
            if j + 1 != max_iter:
                offset[which_violations] *= 0.75
        all_centers.append(centers)
        all_rs.append(np.abs(offset))
    
    out = []
    s_idx = 0
    for i in range(len(src_surfs)):
        if directions[i] == 0:
            C = np.concatenate((all_centers[s_idx], all_centers[s_idx+1]))
            R = np.concatenate((all_rs[s_idx], all_rs[s_idx+1]))
            out.append(QBXExpansions(C, R))
            s_idx += 2
        else:
            out.append(QBXExpansions(all_centers[s_idx], all_rs[s_idx]))
            s_idx += 1
    
    return out