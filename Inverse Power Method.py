import numpy as np

def InversePower(A, mu=0, init_guess=None, eps=0.001, limit=1000):
    L = A.shape[0]
    eig_old = np.zeros(L)
    eig_new = np.ones(L) if init_guess is None else np.array(init_guess)
    count = 0

    while not np.allclose(np.fabs(eig_new), np.fabs(eig_old), atol=eps) and count < limit:
        eig_old = eig_new
        A_inv = np.linalg.inv(A - mu * np.identity(L))
        C = A_inv @ eig_old
        eig_new = C / np.linalg.norm(C)
        count += 1

    eig_val = eig_new @ A @ eig_new
    return eig_val, eig_new, count

M = np.random.randint(-9, 9, (3, 3), dtype='int64')
M = M + M.T
eig, ev, k = InversePower(M)

print("Original Matrix :-\n", M, '\n')
print("Smallest eigenvalue :- \n", eig, '\n')
print("Corresponding eigenvector :- \n", ev, '\n')