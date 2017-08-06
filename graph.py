import numpy as np


class DataGraph:

    def __init__(self, distance_matrix, epsilon, dim):
        # self.vertices will be just range(self.matrix.shape[0])
        # self.edges will be able to be read off the self.matrix

        self.matrix = np.logical_and(distance_matrix <= epsilon, distance_matrix != 0)
        # take only upper triangle of the matrix for information not to repeat itself
        self.matrix = np.triu(self.matrix)
        self.vr = incremental_vr(self.matrix, dim)
        self.clusters = []

    def _get_edges(self, vertex):
        relevant_column = self.matrix[:, vertex]
        relevant_row = self.matrix[vertex, :]
        column_nbrs = np.argwhere(relevant_column == True).flatten()
        row_nbrs = np.argwhere(relevant_row == True).flatten()
        return [{vertex, nbr} for nbr in column_nbrs + row_nbrs]

    def _get_relevant_subgraph(self, simplex):
        return {vr_simplex for vr_simplex in self.vr if simplex <= vr_simplex}

    def get_isomorphism_dict(self):
        # TODO: write get_localhom function
        # TODO: write check_isomorphism function
        isomorphism_dict = {}
        for vertex in range(self.matrix.shape[0]):
            edges = self._get_edges(self, vertex)
            for edge in edges:
                localhom_v = get_localhom(vertex)
                localhom_e = get_localhom(edge)
                isomorphism_dict[(vertex, edge)] = check_isomorphism(localhom_v, localhom_e)


# The incremental algorithm from https://pdfs.semanticscholar.org/e503/c24dcc7a8110a001ae653913ccd064c1044b.pdf
# TODO: rename variables to make it readable
def lower_nbrs(g_matrix, u):
    # look at the column with index u and return row nums with non-zeros
    relevant_column = g_matrix[:, u]
    return np.argwhere(relevant_column == True).flatten()


def add_cofaces(g_matrix, k, t, N, V):
    V.add(t)
    if len(t) >= k:
        return V
    for v in N:
        s = t | frozenset({v})
        M = np.intersect1d(N, lower_nbrs(g_matrix, v))
        V = add_cofaces(g_matrix, k, s, M, V)
    return V


def incremental_vr(g_matrix, k):
    V = set([])
    for u in range(g_matrix.shape[0]):
        N = lower_nbrs(g_matrix, u)
        V = add_cofaces(g_matrix, k, frozenset({u}), N, V)
    return V