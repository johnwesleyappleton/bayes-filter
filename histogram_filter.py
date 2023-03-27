import numpy as np


class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """

    def histogram_filter(self, cmap, belief, action, observation):
        """
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        """

        # flip coordinates
        cmap = np.flip(cmap, 0)
        belief = np.flip(belief, 0)

        # get dimensions of the cmap
        n, m = cmap.shape
        N = n * m

        # define observation matrix
        M = np.zeros((N, 2))
        for row in range(n):
            for col in range(m):
                color = cmap[row, col]
                M[row * m + col, 0] = .1 * color + .9 * (1 - color)
                M[row * m + col, 1] = .1 * (1 - color) + .9 * color

        # define transition matrix
        T = np.zeros((N, N))
        for row in range(n):
            for col in range(m):
                row_new = row + action[1]
                col_new = col + action[0]
                i = row * m + col
                i_new = row_new * m + col_new
                if 0 <= row_new < n and 0 <= col_new < m:
                    T[i, i] = .1
                    T[i, i_new] = .9
                else:
                    T[i, i] = 1

        # define and update forward variables
        forward = belief.reshape(-1)
        forward = (M[:, observation] * (forward.T @ T)).T

        # update posterior belief
        belief = forward.reshape((n, m)) / np.sum(forward)

        # flip coordinates
        cmap = np.flip(cmap, 0)
        belief = np.flip(belief, 0)

        return belief
