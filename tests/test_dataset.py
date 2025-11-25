import numpy as np
from src.minesweeper.dataset import generate_examples


def test_generate_examples_shapes_and_labels():
    X, M, Y, V = generate_examples(n_examples=8, rows=6, cols=6, mines=4, strategy='random', seed=1)
    assert X.shape[0] == 8
    # channels-first: (N, C, H, W)
    assert X.ndim == 4 and X.shape[1] == 10 and X.shape[2] == 6 and X.shape[3] == 6
    # masks shape (N, H*W)
    assert M.shape == (8, 6*6)
    # labels one-hot
    assert Y.shape == (8, 6*6)
    # each label must be one-hot and correspond to a legal action in mask
    for i in range(8):
        lab = Y[i]
        assert np.isin(lab.sum(), [1.0])
        idx = int(np.argmax(lab))
        assert M[i, idx] == 1
