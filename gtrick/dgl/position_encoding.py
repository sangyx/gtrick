import dgl
import torch
import torch.nn.functional as F

import numpy as np
from scipy import sparse as sp


def position_encoding(g, max_freqs):
    n = g.number_of_nodes()
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(
        g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    # Keep up to the maximum desired number of frequencies
    EigVal, EigVec = EigVal[: max_freqs], EigVec[:, :max_freqs]

    # Normalize and pad EigenVectors
    EigVecs = torch.from_numpy(EigVec).float()
    EigVecs = F.normalize(EigVecs, p=2, dim=1, eps=1e-12, out=None)

    if n < max_freqs:
        return F.pad(EigVecs, (0, max_freqs-n), value=0)
    else:
        return EigVecs
