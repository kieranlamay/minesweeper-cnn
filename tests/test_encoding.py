import numpy as np
from src.encoding.full_encoding import encode_full, encode_full_batch, make_action_mask


def test_single_encoding_shape_and_values():
    H, W = 4, 5
    counts = np.zeros((H, W), dtype=int)
    revealed = np.zeros((H, W), dtype=bool)

    # set some revealed numbered cells
    counts[0, 0] = 3
    counts[1, 2] = 0
    revealed[0, 0] = True
    revealed[1, 2] = True

    enc = encode_full(counts, revealed)
    assert enc.shape == (H, W, 10)
    # check channel for number 3 (channel index 3) has a 1 at (0,0)
    assert enc[0, 0, 3] == 1.0
    # channel for number 0 at (1,2)
    assert enc[1, 2, 0] == 1.0
    # unknown channel (index 9) should be 1 where unrevealed
    assert enc[0, 1, 9] == 1.0
    # revealed cell should have unknown=0
    assert enc[0, 0, 9] == 0.0


def test_batch_encoding_and_mask():
    N = 2; H = 3; W = 3
    counts = np.zeros((N, H, W), dtype=int)
    revealed = np.zeros((N, H, W), dtype=bool)
    # reveal different cell per example
    counts[0, 0, 0] = 1; revealed[0, 0, 0] = True
    counts[1, 2, 2] = 2; revealed[1, 2, 2] = True

    enc_batch = encode_full_batch(counts, revealed)
    assert enc_batch.shape == (N, H, W, 10)

    mask0 = make_action_mask(revealed[0])
    assert mask0.shape == (H*W,)
    assert mask0[0] == False  # (0,0) is revealed -> not clickable for example 0
