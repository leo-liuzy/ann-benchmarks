from __future__ import absolute_import
import numpy as np
from ann_benchmarks.algorithms.base import BaseANN
# from pdb import set_trace as bp


class MyCPLSH(BaseANN):
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
        topk_idx = np.argsort(bucket_match_count, axis=-1)[..., ::-1][..., :n]
        return topk_idx[0]
