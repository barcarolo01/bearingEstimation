import numpy as np

ERR_TH = 0.99      # Accept a new estimation if (new_error < 0.99 * old_error)

def MDS_estimate(self_pos, self_err, dist, distance_weights):
    # Replace all NaN distances with 0
    dist = np.nan_to_num(np.asarray(dist, dtype=float), nan=0.0)

    # Compute the weights of the anchors
    pos_w = compute_MDS_pos_w(self_err)

    # Corredt floater positions through MDS
    estimated_pos = mds_algo(self_pos, dist, distance_weights, pos_w)

    return update_decision(self_pos, estimated_pos, dist, self_err, distance_weights)

def compute_MDS_pos_w(err):
    """
    Compute the weights of the positions (anchors) for the MDS algorithm
    
    Input:
        err: self-estimated error of floaters, shape (N,)
    
    Returns:
        tuple: (distance_weights matrice N x N, pos_w matrice N x 1) di pesi normalizzati
    """ 

    # Compute the inverse of the self-estimated positioning error
    pos_w = 1 / (err + 10e-9)

    # Weights normalization
    pos_w = pos_w / np.sum(pos_w) * np.size(pos_w)
    
    # Reshape to a matrix
    pos_w = np.reshape(pos_w, [np.size(pos_w), 1])

    return pos_w

def mds_algo(pos, distance_matrix, distance_weights, position_weights):
    """
    Multidimensional scaling with 2D points with known depth + Procrustes alignment.

    Input:
        pos: anchor positions, shape (N,3)
        distance_matrix: distance matrix, shape (N,N)
        distance_weights distance weights (binary matrix), shape (N,N)
        position_weights: anchor weights, shape (N,)

    Output:
        estimated_positions: coordinates corrected with MDS (N,3)
    """

    pos = np.asarray(pos, dtype=float)
    z   = pos[:, 2]

    # Projection on the horizontal plane
    dz     = z[:, None] - z[None, :]
    dist_h = np.sqrt(np.maximum(distance_matrix**2 - dz**2, 0.0))

    # 2D SMACOF algorithm
    xy_MDS = weighted_smacof(dist_h, distance_weights, init=pos[:, :2], n_iter = 500)

    # Compute rigid transformation (rotation matrix and translation vector)
    R, t = roto_trans(pos[:, :2], xy_MDS, position_weights)
    xy   = (R.dot(np.transpose(xy_MDS)) + np.reshape(t, (-1, 1))).T

    estimated_pos = np.column_stack([xy, z])

    # If a node does not have any distance contraint: return the anchor positions
    isolated = (distance_weights.sum(axis=1) == 0)
    if isolated.any():
        estimated_pos[isolated] = pos[isolated]

    return estimated_pos

def weighted_smacof(D, W, init, n_iter=500, eps=1e-9):
    """
    Applies MDS SMACOF to estimated the cloud of points that best-fits the constraints provided

    Input:
        D: distance_matrix, shape (N,N)
        W: distance_weights (binary mask, 0 = missing distance), shape (N,N)
        init: anchor positions, shape (N,2)
        n_inter: number of iterations to perform to minimize the stress
        eps: threshold under which a distance is considered zero
    
    Output:
        X: floater corrected positions, shape (N,3)
    """

    # Initalization
    X = np.array(init)
    D = np.nan_to_num(np.array(D), nan=0.0)

    # Build the V matrix: (i,i) indicates the number of distances measured for i-th node
    V  = -W.copy()
    np.fill_diagonal(V, W.sum(axis=1))

    # Compute the pseudo-inverse of V
    Vp = np.linalg.pinv(V)

    # SMACOF iterations
    for i in range(n_iter):
        # Compute the distance between every pair of points
        dX = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)

        # Ratio: (measured distance / corrected distance)
        ratio = np.divide(D, dX, out=np.zeros_like(D), where=dX > eps)

        # Weights the ratio with the distance weights to exclude non-measured distances
        B = -W * ratio

        # Force diagonal to be 0 and make every element of each row sum up to 0
        np.fill_diagonal(B, 0.0)
        np.fill_diagonal(B, -B.sum(axis=1))

        # Apply corrections
        X = Vp @ (B @ X)

    return X
   
def roto_trans(pos, pos_MDS, position_weights):
    """
    Computes rototranslation parameters that best matches two cloud points (pos and pos_MDS)
    
    Inputs:
        pos: matrix (N,D) of the position estimated by floaters
        pos_MDS: matrix (N,D) returned by MDS
        position_weights: array (N,) of weights

    Returns:
            R: rotation matrix (D,D)
            t: translation vector (D,)
    """

    # Compute the center of the two clouds of points
    center_MDS = np.sum(pos_MDS * position_weights, axis=0) / np.sum(position_weights)
    center_anchor = np.sum(pos * position_weights, axis=0) / np.sum(position_weights)

    # Shift the two clouds of points to the center
    directions_MDS = (pos_MDS - center_MDS)
    directions = (pos - center_anchor)

    # Positions weights
    W = np.diag(np.reshape(position_weights, np.size(position_weights)))

    # S = pos_MDS^T @ W @ pos (Weighted covariance matrix)
    S = np.transpose(directions_MDS).dot(W).dot(directions)

    # SVD DECOMPOSITION 
    U, _, V = np.linalg.svd(S)

    # Rotation matrix
    R = np.dot(V.T, U.T)
    
    # Check the determinant
    d = np.linalg.det(R)
    if d < 0:
        V[-1, :] *= -1  # Reflect last row of V
        R = np.dot(V.T, U.T)
    
    # Translational vector
    t = np.transpose(center_anchor) - R.dot(np.transpose(center_MDS))

    return R, t

def update_decision(anchor_pos, estimated_pos, dist, err, distance_weights):
    """
    Updates the positions only for floaters whose corrected position in more coherent with the 
    measured distance than the non-corrected position

    Input:
        anchor_pos: floater initial positions (anchors), shape (N,3)
        estimated_pos: floater corrected positions, shape (N,3)
        dist: distance matrix, shape (N,N)
        err: self-estimated positioning error for each floater, shape (N,)
        distance_weights: distance weights (binary mask), shape (N,N)

    Output:
        corrected_positions: corrected positions for each floater, shape (N,3)
    """

    # Error factor for each node: ratio between the new and the old error
    estimated_errors = np.zeros(np.shape(err)[0])

    # Store the error estimated for each node
    for r in range(np.shape(err)[0]):
        estimated_errors[r] = estimate_pos_error(anchor_pos, estimated_pos, dist, r, distance_weights)

    # Mask: array of "True" values only for floaters that improved their position estimation
    accept = estimated_errors < ERR_TH

    # Copy the anchor positions and errors and replace old values with their corrected version, if available
    corrected_positions, corrected_errors = anchor_pos.copy(), err.copy()
    corrected_positions[accept] = estimated_pos[accept]
    corrected_errors[accept] = err[accept] * estimated_errors[accept]

    return corrected_positions, corrected_errors

def estimate_pos_error(pos_anchor, pos_new, dist, ref, distance_weights):
    """
    For each node 'ref', computes the improvement of the correction with respect to its position used as anchor:

    Input:
        pos_anchor: Starting positions for floaters (anchors), shape (N,3)
        pos_new: Corrected positions, shape (N,3)
        dist: distance matrix, shape (N,N)
        ref: number of floater under analysis
        distance_weights: distance weights (binary mask), shape (N,N)

    Output:
        R (int):
            *   = 1: The correction had no effect
            *   > 1: The correction worsen the error compared to the anchor position
            *   < 1: The correction improved the error compared to the anchor position
    """

    # Distance between 'ref' and all other nodes in the new configuration
    new_dist = np.linalg.norm(pos_new - pos_new[ref, :], axis=1)
    new_dist[ref] = 0

    # Distance between 'ref' and all other nodes in the old configuration
    old_dist = np.linalg.norm(pos_anchor - pos_anchor[ref, :], axis=1)
    old_dist[ref] = 0

    # Difference computation: (new - measured) and (old - measured)
    new_diff = abs(new_dist - dist[ref, :])
    old_diff = abs(old_dist - dist[ref, :])

    # Apply the mask to consider only pairs with a measured distance
    m = distance_weights[ref, :].astype(bool)
    new_diff, old_diff = new_diff[m], old_diff[m]

    # Return 1 if the old configuration was already perfect or if the node does not have any distance constraint
    den = np.sum(old_diff)
    if den == 0 or new_diff.size == 0:
        return 1

    R = np.sum(new_diff) / den
    return R