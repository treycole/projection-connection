# from modules.model import Model
from pythtb import TBModel
from collections import defaultdict
import numpy as np
from numpy import sqrt

def fu_kane_mele(t, soc, m, beta):
    # set up Fu-Kane-Mele model
    lat = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    # lat = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    orb = [[0, 0, 0], [0.25, 0.25, 0.25]]
    model = TBModel(3, 3, lat, orb, nspin=2)

    h = m*np.sin(beta)*np.array([1,1,1])
    dt = m*np.cos(beta)

    h0 = [0] + list(h)
    h1 = [0] + list(-h)

    model.set_onsite(h0, 0)
    model.set_onsite(h1, 1)

    # spin-independent first-neighbor hops
    for lvec in ([-1, 0, 0], [0, -1, 0], [0, 0, -1]):
        model.set_hop(t, 0, 1, lvec)

    model.set_hop(3*t + dt, 0, 1, [0, 0, 0], mode="add")

    # spin-dependent second-neighbor hops
    lvec_list = ([1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 1, 0], [0, -1, 1], [1, 0, -1])
    dir_list = ([0, 1, -1], [-1, 0, 1], [1, -1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1])
    for j in range(6):
        spin = np.array([0.]+dir_list[j])
        model.set_hop( 1j*soc*spin, 0, 0, lvec_list[j])
        model.set_hop(-1j*soc*spin, 1, 1, lvec_list[j])

    return model


def get_neighbors(orb_vecs, lat_vecs):
    lat_trans = np.array([[0, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ lat_vecs
    nbrs = {}
    for i, orb_i in enumerate(orb_vecs):
        for j, orb_j in enumerate(orb_vecs):
            for k, trans in enumerate(lat_trans):
                orb_j_trans = np.array(orb_j) + trans
                diff = orb_j_trans - orb_i
                dist = np.linalg.norm(diff)
                nbrs[(i, j, k)] = {
                    'key': (i, j, k),
                    'orb_i': orb_i,
                    'orb_j': orb_j,
                    'lat translation': trans,
                    'pos_i': orb_i,
                    'pos_j': orb_j_trans,
                    'diff': diff,
                    'dist': dist,
                }

    grouped = defaultdict(list)
    for key, val in nbrs.items():
        grouped[key[0]].append((key, val))

    for _, items in grouped.items():
        sorted_items = sorted(items, key=lambda x: x[1]['dist'])

        rank_groups = defaultdict(list)
        for key, val in sorted_items:
            rank_groups[val['dist']].append((key, val))

        for rank, (dist, group) in enumerate(sorted(rank_groups.items()), start=0):
            for key, val in group:
                val['neighbor_rank'] = rank


    return grouped


def axion_ins(Ei, phix, phiy, phiz, t=1):
    lat = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    orb = [[0, 0, 0]]

    model = TBModel(3, 3, lat, orb)

    # 2x2x2 supercell
    model = model.make_supercell([[2, 0, 0], [0, 2, 0], [0, 0, 2]])

    model.set_onsite(Ei, mode='reset')

    nbrs = get_neighbors(model.get_orb_vecs(Cartesian=True), model.get_lat_vecs())
    nn_vecs = [[0, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

    for i, nbrs in nbrs.items():
        for nbr in nbrs:
            data = nbr[1]
            key = nbr[0]
            if data['neighbor_rank'] == 1:
                j = key[1]
                lvec = nn_vecs[key[2]]
                if np.array_equal(data['diff'], np.array([1, 0, 0])):
                    model.set_hop(t * np.exp(1j * phix[i]), i, j, lvec, mode='set')
                elif np.array_equal(data['diff'], np.array([0, 1, 0])):
                    model.set_hop(t * np.exp(1j * phiy[i]), i, j, lvec, mode='set')
                elif np.array_equal(data['diff'], np.array([0, 0, 1])):
                    model.set_hop(t * np.exp(1j * phiz[i]), i, j, lvec, mode='set')

    return model