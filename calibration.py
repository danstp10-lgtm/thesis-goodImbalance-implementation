import numpy as np
from scipy.spatial.transform import Rotation as R

def get_Xsens2Vive_transforms(xsens_samples, vive_samples):
    # Compute centroids
    centroid_xsens = np.mean(xsens_samples, axis=0)
    centroid_vive = np.mean(vive_samples, axis=0)

    # Bring both to origin
    X = xsens_samples - centroid_xsens
    V = vive_samples - centroid_vive

    # Compute the covariance matrix
    H = np.dot(X.T, V)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Validate right-handed coordinate system
    if np.linalg.det(np.dot(Vt.T, U.T)) < 0.0:
        Vt[-1, :] *= -1.0

    # Optimal rotation
    R = np.dot(Vt.T, U.T)

    # Optimal translation (depends on R, so computed after it)
    t = centroid_vive - np.dot(R, centroid_xsens)

    # RMSD
    rmsd = np.sqrt(np.sum(np.square(np.dot(X, R.T) - V)) / xsens_samples.shape[0])

    return R, t, rmsd


def transform_Xsens2TSP(P_xsens, R_xv, t_xv, P_TSP, R_TSP):
    """
    Transforms a 3D point from Xsens space directly to TSP space.
    P_xsense - point in Xsens space
    R_xv - rotation matrix Xsens to Vive
    t_xv - trainslation Xsens to Vive
    P_TSP - origin of TSP, marked by third Vive controller
    R_TSP - rotation of TSP, also get from Vive controller
    """
    P_vive = (R_xv @ P_xsens) + t_xv # Xsens point to Vive Space
    P_grid = R_TSP.T @ (P_vive - P_TSP) # Vive point to TSP Space
    return P_grid

