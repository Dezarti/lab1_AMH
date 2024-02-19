import numpy as np

import numpy as np

import numpy as np

class TSNE:
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=150.0, n_iter=100, momentum=0.9, random_state=None, plot_every=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.momentum = momentum
        self.random_state = random_state
        self.plot_every = plot_every
        self.rng = np.random.RandomState(random_state)

    def _neg_squared_euc_dists(self, X):
        """Compute matrix containing negative squared euclidean
        distance for all pairs of points in input matrix X."""
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        return -D

    def _softmax(self, X, diag_zero=True, zero_index=None):
        """Compute softmax values for each row of matrix X."""
        e_x = np.exp(X - np.max(X, axis=1).reshape([-1, 1]))
        if zero_index is None:
            if diag_zero:
                np.fill_diagonal(e_x, 0.)
        else:
            e_x[:, zero_index] = 0.
        e_x = e_x + 1e-8  # numerical stability
        return e_x / e_x.sum(axis=1).reshape([-1, 1])

    def _calc_prob_matrix(self, distances, sigmas=None, zero_index=None):
        if sigmas is not None:
            two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
            return self._softmax(distances / two_sig_sq, zero_index=zero_index)
        else:
            return self._softmax(distances, zero_index=zero_index)

    @staticmethod
    def _binary_search(eval_fn, target, tol=1e-10, max_iter=10000, lower=1e-20, upper=1000.):
        for i in range(max_iter):
            guess = (lower + upper) / 2.
            val = eval_fn(guess)
            if val > target:
                upper = guess
            else:
                lower = guess
            if np.abs(val - target) <= tol:
                break
        return guess

    def _calc_perplexity(self, prob_matrix):
        entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1)
        perplexity = 2 ** entropy
        return perplexity

    def _perplexity(self, distances, sigmas, zero_index):
        return self._calc_perplexity(
            self._calc_prob_matrix(distances, sigmas, zero_index))
    
    def _find_optimal_sigmas(self, distances, target_perplexity):
        sigmas = []
        for i in range(distances.shape[0]):
            # Aquí usamos un lambda que referencia _perplexity, ajustado para ser un método de instancia
            eval_fn = lambda sigma: self._perplexity(distances[i:i+1, :], np.array(sigma), i)
            correct_sigma = self._binary_search(eval_fn, target_perplexity)
            sigmas.append(correct_sigma)
        return np.array(sigmas)
    
    def _p_conditional_to_joint(self, P):
        return (P + P.T) / (2. * P.shape[0])
    
    def _q_joint(self, Y):
        distances = self._neg_squared_euc_dists(Y)
        exp_distances = np.exp(distances)
        np.fill_diagonal(exp_distances, 0.)
        return exp_distances / np.sum(exp_distances)

    def _symmetric_sne_grad(self, P, Q, Y):
        pq_diff = P - Q
        pq_expanded = np.expand_dims(pq_diff, 2)
        y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
        grad = 4. * (pq_expanded * y_diffs).sum(1)
        return grad

    def _q_tsne(self, Y):
        distances = self._neg_squared_euc_dists(Y)
        inv_distances = np.power(1. - distances, -1)
        np.fill_diagonal(inv_distances, 0.)
        return inv_distances / np.sum(inv_distances), inv_distances
    
    def _tsne_grad(self, P, Q, Y, distances):
        pq_diff = P - Q
        pq_expanded = np.expand_dims(pq_diff, 2)
        y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
        distances_expanded = np.expand_dims(distances, 2)
        y_diffs_wt = y_diffs * distances_expanded
        grad = 4. * (pq_expanded * y_diffs_wt).sum(1)
        return grad

    def _p_joint(self, X, target_perplexity):
        distances = self._neg_squared_euc_dists(X)
        sigmas = self._find_optimal_sigmas(distances, target_perplexity)
        p_conditional = self._calc_prob_matrix(distances, sigmas)
        P = self._p_conditional_to_joint(p_conditional)
        return P

    def fit(self, X, y=None):
        P = self._p_joint(X, self.perplexity)

        Y = self.rng.normal(0., 0.0001, [X.shape[0], self.n_components])
        Y_m2 = Y.copy()
        Y_m1 = Y.copy()

        for i in range(self.n_iter):
            Q, distances = self._q_tsne(Y)  # Usar _q_joint si es SNE
            grads = self._tsne_grad(P, Q, Y, distances)

            Y = Y - self.learning_rate * grads
            if self.momentum:
                Y += self.momentum * (Y_m1 - Y_m2)
                Y_m2, Y_m1 = Y_m1, Y.copy()
        self.Y_ = Y
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.Y_
