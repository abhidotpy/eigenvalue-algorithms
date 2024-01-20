import numpy as np

def Jacobi(A, eps=1E-3, limit=100000, mu=None):

    if not np.allclose(A, A.T):
        print('Asymmetric Matrix')
        return None

    row, col = np.shape(A)
    B = A.copy()
    R = np.eye(col)  # Storing the sequential projectors
    I = np.eye(col)  # Identity Matrix
    count = 0        # Iteration counter


    while not np.allclose(np.tril(B, -1), 0, atol=eps) and count < limit:
        index = np.argmax(np.abs(np.triu(B, 1)))
        i, j = index // col, index % col

        S = np.eye(col)    # Identity initialization
        a, b = 2 * B[i, j], (B[j, j] - B[i, i])
        hyp = np.hypot(a, b)      # Hypotenuse
        c = np.sqrt(0.5 * (1 + b / hyp))    # Cosine of rotation
        s = a / (hyp * np.sqrt(2 * (1 + b / hyp)))  # Sine of rotation

        S[i, i] = S[j, j] = c  # Changing the minor of identity
        S[i, j] = s
        S[j, i] = -s

        R = R @ S
        B = S.T @ B @ S
        count += 1

    return np.diag(B), R, count


M = np.random.randint(-9, 9, (3, 3), dtype='int64')
M = M + M.T
eig, ev, k = Jacobi(M)

print("Original Matrix :-\n", M, '\n')
print("Eigenvalues :- \n", eig, '\n')
print("Eigenvectors :- \n", ev, '\n')