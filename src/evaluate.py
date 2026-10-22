import numpy as np
from official_metrics import group_means

def normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
