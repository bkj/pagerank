#!/usr/bin/env python

"""
  pr.py
"""

import numpy as np
from time import time
from scipy.io import mmread
from numba import njit, prange

@njit(parallel=True)
def _pr_numba(n_nodes, n_edges, indptr, indices, data, alpha, max_iter, tol):
    x = np.ones(n_nodes) / n_nodes
    p = np.ones(n_nodes) / n_nodes
    
    for it in range(max_iter):
        xlast = x
        x     = (1.0 - alpha) * p
        
        for dst in prange(n_nodes):
          for offset in range(indptr[dst], indptr[dst + 1]):
            src = indices[offset]
            val = data[offset]
            x[dst] += alpha * xlast[src] * val # would be better to use csc, but conversion time...
        
        err = np.abs(x - xlast).max()
        if err < tol:
            return x
      
    return x

def pr_numba(adj, alpha=0.85, max_iter=100, tol=1e-6):
  n_nodes = adj.shape[0]
  n_edges = adj.nnz
  return _pr_numba(n_nodes, n_edges, adj.indptr, adj.indices, adj.data, alpha=alpha, max_iter=max_iter, tol=tol)

# --
# Run

csr = mmread('rmat20.mtx')

adj = csr.multiply(1 / (csr.sum(axis=-1) + 1e-10))
adj = adj.tocsc()

_ = pr_numba(adj[:100,:100])

for _ in range(10):
  t       = time()
  x       = pr_numba(adj)
  elapsed = time() - t
  
  print(elapsed)

