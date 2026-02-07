import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        #check type and values 

        if type(k) != int:
            raise ValueError("Input k need to be an integer")

        if type(tol) != int and type(tol) != float:
            raise ValueError("tol need to be a float or int")

        if type(max_iter) != int:
            raise ValueError("max_iter need to be an int")

        if k < 1 or max_iter < 1 :
            raise ValueError("k and max_iter need to be greater than 0.")


        self.k = k
        self.tol = tol
        self.max_iter = max_iter

        self.centroids = []    
        self.ErrorTracking = [] 

            

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        if type(mat) != np.ndarray:
            raise ValueError("mat need to be a np array.")
        
        
        cellNum = mat.shape[0]
        geneNum = mat.shape[1]

        
        self.centroids = np.random.randn(self.k, geneNum)

        self.ErrorTracking = []

        i = 0
        while i < self.max_iter:
            
            # (cellNum, k); distance of each point to each centroid....
            distanceMatrix = cdist(mat, self.centroids) 

            # (cellNum,)
            labels = np.argmin(distanceMatrix, axis=1)  

            # sum of square of dist diff = error; calculate and update
            assigned = distanceMatrix[np.arange(cellNum), labels]
            sse = float(np.sum(assigned * assigned))
            self.ErrorTracking.append(sse)

            # initiate new centroid array 
            newCentroids = np.zeros((self.k, geneNum), dtype=float)

            #loop through all the clusters 
            cluster = 0
            while cluster < self.k:
                pointsInCluster = mat[labels == cluster]

                if pointsInCluster.shape[0] == 0:
                    # if no points assigned, bad centroid, rand init a new one 
                    rand_index = np.random.randint(0, cellNum)
                    newCentroids[cluster] = mat[rand_index]
                else:
                    newCentroids[cluster] = np.mean(pointsInCluster, axis=0)

                cluster = cluster + 1

            # how much did the centroid change 
            diff = newCentroids - self.centroids
            shift_each = np.sqrt(np.sum(diff * diff, axis=1))
            max_shift = float(np.max(shift_each))

            self.centroids = newCentroids

            #when under the tol value, can stop
            if max_shift < self.tol:
                break

            i = i + 1

        return None

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        distanceMatrix = cdist(mat, self.centroids)
        #return index of smallest distance in each row
        labels = np.argmin(distanceMatrix, axis=1)
        return labels

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        if len(self.ErrorTracking) == 0:
            raise ValueError("Cannot call get error before fitting the points.")

        return float(self.ErrorTracking[-1])


    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        if len(self.centroids) == 0:
            raise ValueError("Cannot get centroids before fitting the points.")

        return self.centroids

