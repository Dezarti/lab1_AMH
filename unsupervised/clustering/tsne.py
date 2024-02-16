import numpy as np

def Hbeta(D=np.array([]), beta=1.0):
    P = np.exp(-D.copy() * beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P

def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), axis=1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)
    
    for i in range(n):
        betamin = -np.inf
        betamax = np.inf
        H, thisP = Hbeta(D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))], beta[i])
        
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.
            
            H, thisP = Hbeta(D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))], beta[i])
            Hdiff = H - logU
            tries += 1
        
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
        
    return P

def pca(X=np.array([]), no_dims=50):
    (n, d) = X.shape
    X = X - np.mean(X, axis=0)
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:,0:no_dims])
    return Y

class TSNE:
    def __init__(self, n_components=2, initial_dims=50, perplexity=30.0):
        self.n_components = n_components
        self.initial_dims = initial_dims
        self.perplexity = perplexity
    
    def fit_transform(self, X):
        X = pca(X, self.initial_dims).real
        (n, d) = X.shape
        P = x2p(X, 1e-5, self.perplexity)
        P = P + P.T
        P = P / np.sum(P)
        P = P * 4.
        P = np.maximum(P, 1e-12)
        
        Y = np.random.randn(n, self.n_components)
        dY = np.zeros((n, self.n_components))
        iY = np.zeros((n, self.n_components))
        gains = np.ones((n, self.n_components))
        
        for iter in range(1000):
            sum_Y = np.sum(np.square(Y), axis=1)
            num = -2. * np.dot(Y, Y.T)
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
            num[range(n), range(n)] = 0.
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)
            
            PQ = P - Q
            for i in range(n):
                dY[i,:] = np.sum(np.tile(PQ[:,i] * num[:,i], (self.n_components, 1)).T * (Y[i,:] - Y), 0)
            
            Y = Y + 0.5 * dY
            Y = Y - np.tile(np.mean(Y, 0), (n, 1))
            
            if iter == 100:
                P = P / 4.
                
        return Y