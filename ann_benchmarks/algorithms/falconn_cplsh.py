from __future__ import absolute_import
import numpy as np
from ann_benchmarks.algorithms.base import BaseANN
import falconn
# from pdb import set_trace as bp


class FalconnCPLSH(BaseANN):
    def __init__(self, metric, n_hashes=20, n_buckets=50, n_pseudo_rotation=3):
        assert metric == "angular"
        self._metric = metric
        self.name = 'FalconnCPLSH(n_hash=%d, n_bucket=%d, n_pseudo_rotation=%d)' % (n_hashes, n_buckets, n_pseudo_rotation)

        # we haven't set vector dim yet
        params_cp = falconn.LSHConstructionParameters()
        params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
        params_cp.distance_function = falconn.DistanceFunction.NegativeInnerProduct
        params_cp.l = n_hashes
        # params_cp.k = n_buckets
        # params_cp.last_cp_dimension = 1 # may need to tune
        params_cp.num_rotations = n_pseudo_rotation
        params_cp.seed = 5721840
        params_cp.num_setup_threads = 1
        params_cp.storage_hash_table = falconn.StorageHashTable.FlatHashTable
        self.params_cp = params_cp

        self._n_hashes = n_hashes
        self._n_buckets = n_buckets
        self._n_pseudo_rotation = n_pseudo_rotation

        self.table = None
        self.query_object = None

    def fit(self, X):
        _, d = X.shape
        self.params_cp.dimension = d
        falconn.compute_number_of_hash_functions(18, self.params_cp)

        assert X.dtype == np.float32
        norm_X = X / np.linalg.norm(X, axis=-1, keepdims=True)
        self.norm_X = norm_X
        table = falconn.LSHIndex(self.params_cp)
        # table.setup(deepcopy(norm_X))
        table.setup(self.norm_X)
        self.table = table
        self.query_object = table.construct_query_object()

    def query(self, v, n):
        v = np.array([v]).astype(np.float32)
        v = v / np.linalg.norm(v, axis=-1, keepdims=True)
        v = v[0]
        topk_index = self.query_object.find_k_nearest_neighbors(v, k=n)
        return topk_index