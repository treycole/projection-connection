from wanpy.wpythtb import *
import numpy as np
from modules import tb_model, wf_array
import matplotlib.pyplot as plt
from calculations.models import *
from scipy.linalg import polar


def Berry_phase(wfs, last_phase=1, dir=0):
    dim_param = len(wfs.shape[:-2]) # dimensionality of parameter space
    param_axes = np.arange(0, dim_param) # parameter axes
    param_axes = np.setdiff1d(param_axes, dir) # remove direction from axes to loop
    lens = [wfs.shape[ax] for ax in param_axes] # sizes of loop directions
    idxs = np.ndindex(*lens) # index mesh

    phase = np.zeros((*lens, wfs.shape[-2]))

    for idx_set in idxs:
        # take wfs along loop axis at given idex
        sliced_wf = wfs.copy()
        for ax, idx in enumerate(idx_set):
            sliced_wf = np.take(sliced_wf, idx, axis=param_axes[ax])

        # wf now has 3 indices: [phase ax, eigval idx, orb amp]
        for n in range(sliced_wf.shape[-2]): # loop over eigval idxs
            prod = np.prod(
                [ np.vdot(sliced_wf[i, n], sliced_wf[i+1, n])
                for i in range(sliced_wf.shape[0]-1) ] )
            prod *= np.vdot(sliced_wf[-1, n], sliced_wf[0, n]*last_phase)
            phase[idx_set][n] = -np.angle(prod)

    return phase


t = -1.3
delta = 2.0
lmbd = 0.3
spd_model = chain(t, delta, lmbd)

orb_vecs = spd_model.get_orb_vecs()

nks = np.arange(10, 1000)
diffs = np.zeros(nks.shape)

for idx, nk in enumerate(nks):
    print(nk)

    t = -1.3
    delta = 2.0
    lmbd = 0.3
    spd_model = chain(t, delta, lmbd)

    dk = 1/nk
    kpts = np.linspace(0, 1, nk, endpoint=False)[:, None]

    # Velocity matrix
    v_mu = spd_model.gen_velocity(kpts)[0]

    # Bloch class
    bloch_wfs = Bloch(spd_model, nk)
    bloch_wfs.solve_model()
    u_nk_sp = bloch_wfs.get_states()["Cell periodic"]
    psi_nk_sp = bloch_wfs.get_states()["Bloch"]
    energies = bloch_wfs.energies

    # Occupied (lower two) and conduction (last) bands
    u_occ, psi_occ = u_nk_sp[:, :2, :], psi_nk_sp[:, :2, :]
    u_con, psi_con = u_nk_sp[:, -1, :][:, None, :], psi_nk_sp[:, -1, :][:, None, :]
    n_occ, n_con = u_occ.shape[-2], u_con.shape[-2]

    E_occ, E_con = energies[:, :2], energies[:, -1][:, None]

    # Bloch class for occupied bands
    bloch_occ = Bloch(spd_model, nk)
    bloch_occ.set_wfs(psi_occ, cell_periodic=True)

    # Bloch class for conduction bands
    bloch_con = Bloch(spd_model, nk)
    bloch_con.set_wfs(psi_con, cell_periodic=True)

    # # For tilde (projection) gauge states
    bloch_tilde = Bloch(spd_model, nk)
    WF = Wannier(spd_model, [nk])

    # Trial wavefunctions as delta on first and second orbital
    tf_list = [[(0,0), (1,1), (2,0)], [(0,0), (1,0), (2,1)]]
    twfs = WF.get_trial_wfs(tf_list)

    # Overlap function S_nm = <psi_nk| g_m> with occupied bands
    S_occ = np.einsum("...nj, mj -> ...nm", psi_occ.conj(), twfs)
    S_con = np.einsum("...nj, mj -> ...nm", psi_con.conj(), twfs)

    # SVD
    W, Sig, Vh = np.linalg.svd(S_occ, full_matrices=True)
    Sig_mat = np.einsum("...i, ij -> ...ij", Sig, np.eye(Sig.shape[-1]))

    # Polar
    pol =  np.array([polar(S_occ[k]) for k in range(S_occ.shape[0]) ])
    U_p = pol[:, 0]
    P = pol[:, 1]

    # Unitary part
    U_rot = W @ Vh

    ##### Berry connections from finite differences (links)
    psi_tilde = np.einsum("...mn, ...mj -> ...nj", U_rot, psi_occ) # shape: (*nks, states, orbs*n_spin])
    bloch_tilde.set_wfs(psi_tilde, cell_periodic=False)
    u_tilde = bloch_tilde.get_states()["Cell periodic"]

    U_links_tilde = bloch_tilde.get_links(None)
    U_links_en = bloch_occ.get_links(None)

    # A_mu
    eigvals, eigvecs = np.linalg.eig(U_links_en)
    angles = -np.angle(eigvals)
    angles_diag = np.einsum("...i, ij -> ...ij", angles, np.eye(angles.shape[-1]))
    eigvecs_inv = np.linalg.inv(eigvecs)
    berry_conn_en = np.matmul(np.matmul(eigvecs, angles_diag), eigvecs_inv)

    # tilde{A}_mu
    eigvals, eigvecs = np.linalg.eig(U_links_tilde)
    angles = -np.angle(eigvals)
    angles_diag = np.einsum("...i, ij -> ...ij", angles, np.eye(angles.shape[-1]))
    eigvecs_inv = np.linalg.inv(eigvecs)
    # NOTE: This is what is being compared to (finite differences)
    berry_conn_tilde = np.matmul(np.matmul(eigvecs, angles_diag), eigvecs_inv)

    ##### Parmu S
    parmu_S = np.zeros_like(S_occ, dtype=complex)
    for n in range(S_occ.shape[-1]):
        for m in range(n_occ):
            for c in range(n_con):
                u_m_occ, u_c_con = u_occ[:, m, :], u_con[:, c, :]
                v_mu_mc = np.einsum("...j, ...jk, ...k", u_m_occ.conj(), v_mu, u_c_con)
                parmu_S[:, m, n] += ( v_mu_mc / (E_occ[:, m] - E_con[:, c]) ) * S_con[:, c, n]

    ## NOTE: This term was missing previously
    parmu_S += 1j* berry_conn_en[0]*nk @ S_occ

    #### P parmu P
    mid_mat = Vh @ (parmu_S.conj().swapaxes(-1,-2) @ S_occ + S_occ.conj().swapaxes(-1,-2) @ parmu_S) @ Vh.conj().swapaxes(-1,-2)
    for a in range(mid_mat.shape[-2]):
        for b in range(mid_mat.shape[-1]):
            mid_mat[:, a, b] *= (Sig[:, a] / (Sig[:, a] + Sig[:, b]))

    PparmuP = Vh.conj().swapaxes(-1,-2) @ mid_mat @ Vh

    #### U^dag parmu U
    UdagparU = np.linalg.inv(P) @ ( S_occ.conj().swapaxes(-1,-2) @ parmu_S - PparmuP ) @ np.linalg.inv(P)

    ##### NOTE: This is the derived equation ######
    A_tilde_ad =  U_rot.conj().swapaxes(-1,-2) @ berry_conn_en[0]*nk @ U_rot + 1j*UdagparU

    diff = A_tilde_ad - berry_conn_tilde[0]*nk
    diffs[idx] = np.amax(diff).real

plt.plot(nks, diffs)
plt.xlabel(r"$N_k$")
plt.ylabel(r"$\text{max} |\tilde{A}_{\mu}^{fd} - \tilde{A}_{\mu}^{note}|$")
plt.show()
