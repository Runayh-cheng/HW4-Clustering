
# Write your k-means unit tests here

import numpy as np
import pytest

from cluster.kmeans import KMeans

def test_input_type_check():
    #wrong input
    inputArray = np.array([[1,2],
    [1,2]])
    with pytest.raises(ValueError):
        testInit = KMeans(k="wrong input", tol = 1e-6, max_iter=50)

def test_input_val_check_1():
    #wrong input
    inputArray = np.array([[1,2],
    [1,2]])
    with pytest.raises(ValueError):
        testInit = KMeans(k=0, tol = 1e-6, max_iter=50)

def test_input_val_check_2():
    #wrong input
    inputArray = np.array([[1,2],
    [1,2]])
    with pytest.raises(ValueError):
        testInit = KMeans(k=1, tol = 1e-6, max_iter=0)

def test_fit_then_call_1():
    inputArray = np.array([[1,2],
    [1,2]])
    testInit = KMeans(k=1, tol = 1e-6, max_iter=50)
    with pytest.raises(ValueError):
        testInit.get_error()

def test_fit_then_call_2():
    inputArray = np.array([[1,2],
    [1,2]])
    testInit = KMeans(k=1, tol = 1e-6, max_iter=50)
    with pytest.raises(ValueError):
        testInit.get_centroids()

def test_error():
    inputArray = np.array([[1,2],
    [1,2]])
    testInit = KMeans(k=1, tol = 1e-6, max_iter=50)
    testInit.fit(inputArray)
    assert testInit.get_error() < 1e-6


def test_centroids_shape():
    inputArray = np.array([[1,2],
    [1,2]])
    testInit = KMeans(k=1, tol = 1e-6, max_iter=50)
    testInit.fit(inputArray)
    assert testInit.get_centroids().shape == (1, 2)

def test_centroids_calc():
    inputArray = np.array([[1,2],
    [1,2]])
    testInit = KMeans(k=1, tol = 1e-6, max_iter=50)
    testInit.fit(inputArray)
    finalCentroids = testInit.get_centroids()
    print(finalCentroids)
    assert 0 <= finalCentroids[0][0] <= 2
    assert 1 <= finalCentroids[0][1] <= 3



    


    