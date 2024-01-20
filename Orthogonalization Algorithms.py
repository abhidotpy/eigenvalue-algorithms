import numpy as np

def GramSchmidt(A):
    row, col = A.shape
    B = np.zeros(col).reshape(-1, 1)
    for i in range(col):
        u = A[:, i] / np.linalg.norm(A[:, i])
        v = np.zeros_like(u)
        for j in range(np.shape(B)[1]):
            v = v + np.vdot(u, B[:, j]) * B[:, j]
        u = (u - v)
        u = u / np.linalg.norm(u)
        B = np.hstack((B, u.reshape(-1, 1)))
    return B[:, 1:].T


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


def Givens(A):
    B = A.copy()
    row, col = np.shape(A)
    R = np.eye(col)

    for j in range(col-1):
        I = np.eye(col)                       # Identity initialization
        hyp = np.hypot(A[j+1, j], A[j, j])    # Hypotenuse
        C = A[j, j] / hyp                     # Cosine of rotation
        S = A[j+1, j] / hyp                   # Sine of rotation

        I[j, j] = I[j+1, j+1] = C               # Changing the minor of identity
        I[j+1, j] = -S
        I[j, j+1] = S
        R = I @ R                           # Reverse multiplication of projectors
        A = I @ A                           # Transform A to upper triangular form

    return R


def Tridiagonal(A):
    row, col = np.shape(A)
    B = A.copy()
    P = np.eye(col)
    I = np.eye(col)

    for i in range(col - 2):
        W = B[i+1:, i:]  # Principal minor of A_ii
        T = np.eye(col)
        v = W[:, 0] - np.linalg.norm(W[:, 0]) * I[:, 0][:-i-1]  # Projection hyperplane vector
        if np.allclose(v, 0):
            T1 = I[i+1:, i+1:]
        else:
            T1 = I[i+1:, i+1:] - (2 * np.outer(v, v)) / np.vdot(v, v)  # Projector matrix
        T[i+1:, i+1:] = T1  # Projector matrix as minor of identity
        P = P @ T       # Reverse multiplication of projectors
        B = T.T @ B @ T

    return P.T @ A @ P