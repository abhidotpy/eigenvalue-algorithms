import numpy as np

def Householder(A):
    row, col = np.shape(A)
    P = np.eye(col)     # For storing the projectors
    I = np.eye(col)     # Just an identity matrix

    for i in range(col-1):
        W = A[i:, i:]                                               # Principal minor of A_ii
        T = np.eye(col)
        v = W[:, 0] - np.linalg.norm(W[:, 0]) * I[:, i][i:]         # v = x - ||x||e_i
        if np.allclose(v, 0):
            T1 = I[i:, i:]                                          # To avoid divide by zero error
        else:
            T1 = I[i:, i:] - (2 * np.outer(v, v)) / np.vdot(v, v)   # Projector matrix
        T[i:, i:] = T1                                              # Projector matrix as minor of identity
        P = T @ P                                                  # Reverse multiplication of projectors
        A = T @ A                                                   # Transform A to upper triangular form

    return P


def QR(A, eps=1E-3, limit=1000, method=None):
    row, col = A.shape
    B = A.copy()
    count = 0
    V = np.eye(row)

    while not np.allclose(np.tril(B, -1), 0, atol=eps) and count < limit:
        
        Q = Householder(B)
        R = Q.T @ B
        B = R @ Q
        V = V @ Q
        count += 1

    if np.allclose(A, A.T):
        return np.diag(B), V, count
    else:
        return np.diag(B), count


def QRShift(A, eps=1E-3, limit=1000, method=None):
    row, col = A.shape
    B = A.copy()
    count = 0
    reduced = False
    V = np.eye(A.shape[0])
    I = np.identity(B.shape[0])

    if np.allclose(np.tril(B, -2), 0, atol=eps) and method is None:  # If matrix is already in a reduced form
        reduced = True

    while not np.allclose(np.tril(B, -1), 0, atol=eps) and count < limit:
        mu = A[-1, -1]

        Q = Householder(B - mu * I)
        R = Q.T @ (B - mu * I)
        B = R @ Q + mu * I
        V = V @ Q
        count += 1

    if np.allclose(A, A.T):
        return np.diag(B), V, count
    else:
        return np.diag(B), count
    
M = np.random.randint(-9, 9, (3, 3), dtype='int64')
M = M + M.T
eig, ev, k = QR(M)

print("Original Matrix :-\n", M, '\n')
print("Eigenvalues :- \n", eig, '\n')
print("Eigenvectors :- \n", ev, '\n')