import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        if len(X.shape) != 2:
            raise ValueError ("X needs to be a 2D matrix.")
        if type(y) != np.ndarray:
            raise ValueError ("y needs to be a 1D label array for each point.")
        if X.shape[0]!=y.shape[0]:
            riase ValueError("X y dim mismatch.")


        NumPoints = X.shape[0]

        UniqueLabels = np.unique(y)
        if UniqueLabels.shape[0] < 2:
            raise ValueError("Need at least 2 clusters")

        PairwiseDist = cdist(X, X)
        Scores = np.zeros(NumPoints, dtype=float)

        PointNum = 0
        while PointNum < NumPoints:
            CurrentLabel = y[PointNum]

            SameLabelRows = np.where(y == CurrentLabel)[0]

            if SameLabelRows.shape[0] <= 1:
                Scores[PointNum] = 0.0
                PointNum = PointNum + 1
                continue

            SameLabelRowsNoSelf = SameLabelRows[SameLabelRows != PointNum]
            AvgDistSame = float(np.mean(PairwiseDist[PointNum, SameLabelRowsNoSelf]))

            BestAvgDistOther = None
            LabelNum = 0

            while LabelNum < UniqueLabels.shape[0]:
                OtherLabel = UniqueLabels[LabelNum]

                if OtherLabel != CurrentLabel:
                    OtherLabelRows = np.where(y == OtherLabel)[0]
                    AvgDistOther = float(np.mean(PairwiseDist[PointNum, OtherLabelRows]))

                    if (BestAvgDistOther is None) or (AvgDistOther < BestAvgDistOther):
                        BestAvgDistOther = AvgDistOther

                LabelNum = LabelNum + 1

            #bottom of the S fx
            Denom = AvgDistSame
            if BestAvgDistOther > Denom:
                Denom = BestAvgDistOther
            
            # !!! denominator cannot be 0 or else cannot divide, just mark 0 
            if Denom == 0:
                Scores[PointNum] = 0.0
            else:
                Scores[PointNum] = (BestAvgDistOther - AvgDistSame) / Denom

            PointNum = PointNum + 1

        return Scores
