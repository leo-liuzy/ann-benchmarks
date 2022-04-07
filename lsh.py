# from __future__ import absolute_import
import numpy as np
from copy import deepcopy
import falconn
# from ann_benchmarks.algorithms.base import BaseANN
from pdb import set_trace as bp


class MyCPLSH():
    def __init__(self, metric, n_hashes=20, n_buckets=50):
        assert metric == "angular"
        self.name = 'MyCPLSH(n_hash=%d, n_bucket=%d)' % (n_hashes, n_buckets)
        self._metric = metric
        self._n_hashes = n_hashes
        self._n_buckets = n_buckets

    def fit(self, X):
        _, d = X.shape
        rotations_shape = (d, self._n_hashes, self._n_buckets // 2)
        norm_X = X / np.linalg.norm(X, axis=1, keepdims=True)
        self.R = np.random.randn(*rotations_shape)
        rotated_vectors = np.einsum("bd,dhr->bhr", norm_X, self.R)
        rotated_vectors = rotated_vectors / np.linalg.norm(rotated_vectors, axis=-1, keepdims=True)
        self.X_hash = np.argmax(np.concatenate([rotated_vectors, -rotated_vectors], axis=-1), axis=-1)        

    def query(self, v, n):    
        v = np.array([v])
        v = v / np.linalg.norm(v, axis=-1, keepdims=True)
        v_rotated = np.einsum("bd,dhr->bhr", v, self.R)
        v_hash = np.argmax(np.concatenate([v_rotated, -v_rotated], axis=-1), axis=-1)
        bucket_match_count = np.sum(np.expand_dims(v_hash, 1) == np.expand_dims(self.X_hash, 0), axis=-1)
        k = min(n, np.count_nonzero(bucket_match_count[0])) # it's possible that we didn't have enough hit
        topk_idx = np.argsort(bucket_match_count, axis=-1)[..., ::-1][..., :k]
        return topk_idx[0]

class FalconnCPLSH():
    # [8,32,64,128,256]
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
        # self.query_object.set_num_probes(self.params_cp.l)

    def query(self, v, n):
        v = np.array([v]).astype(np.float32)
        v = v / np.linalg.norm(v, axis=-1, keepdims=True)
        v = v[0]
        topk_index = self.query_object.find_k_nearest_neighbors(v, k=n)
        return topk_index


if __name__ == "__main__":
    search = MyCPLSH('angular')
    X = np.random.randn(1000,100,).astype(np.float32)
    search.fit(X)
    query = np.random.randn(100)
    n = 10
    search.query(query, n)
