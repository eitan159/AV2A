import numpy as np

def idx2onehot(a, k):
    a = a.astype(int)
    b = np.zeros((a.size, k))
    b[np.arange(a.size), a] = 1
    return b


def confusion_matrix(ytrue, ypred, k):
    # C[i,j] denotes the frequency of ypred = i, ytrue = j.
    n = ytrue.size
    C = np.dot(idx2onehot(ypred,k).T,idx2onehot(ytrue,k))
    return C/n

def confusion_matrix_probabilistic(ytrue, ypred, k):
    # Input is probabilistic classifiers in forms of n by k matrices
    n, d = np.shape(ypred)
    C = np.dot(ypred.T, ytrue)
    return C/n


def calculate_marginal(y,k):
    mu = np.zeros(shape=(k,1))
    for i in range(k):
        mu[i] = np.count_nonzero(y == i)
    return mu/np.size(y)

def calculate_marginal_probabilistic(y, k):
    return np.mean(y, axis=0)

def estimate_labelshift_ratio(ytrue_s, ypred_s, ypred_t, k): # (1-hot vector of frame_i, props of frame_i, probs of frame_i+1)
    #if ypred_s.ndim == 2: # this indicates that it is probabilistic
    C = confusion_matrix_probabilistic(ytrue_s,ypred_s,k)
    mu_t = calculate_marginal_probabilistic(ypred_t, k)
    lamb = (1/min(len(ypred_s),len(ypred_t)))
    wt = np.linalg.solve(np.dot(C.T, C)+lamb*np.eye(k), np.dot(C.T, mu_t))

    return wt * mu_t