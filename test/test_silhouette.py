
import numpy as np
import pytest

from cluster.silhouette import Silhouette


def test_score_inputCheck():
    #x Y mismatch
    X = np.array([[0.0, 0.0],
                  [0.0, 1.1],
                  [4.0, 5.3],
                  [4.9, 2.2]])
    y = np.array([0, 1])
    init_1 = Silhouette()
    with pytest.raises(ValueError):
        score_1 = init_1.score(X, y)


def test_score_outputShape():
    #x Y mismatch
    X = np.array([[0.0, 0.0],
                  [0.0, 1.1],
                  [4.0, 5.3],
                  [4.9, 2.2]])
    y = np.array([0, 0, 1, 1])
    init_1 = Silhouette()
    score_1 = init_1.score(X, y)
    assert score_1.shape[0] == 4