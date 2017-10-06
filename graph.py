import numpy as np

from helpers import get_boundary_operator, check_isomorphism, get_betti_numbers


class DataGraph:

    def __init__(self, distance_matrix, epsilon, dim):
        # self.vertices will be just range(self.matrix.shape[0])
        # self.edges will be able to be read off the self.matrix

        self.matrix = np.logical_and(distance_matrix <= epsilon, distance_matrix != 0)
        # take only upper triangle of the matrix for information not to repeat itself
        self.matrix = np.triu(self.matrix)
        self.epsilon = epsilon
        self.dim = dim
        self._get_vr_complex(dim)
        self._isomorphism_dict = {}
        self.clusters = []
        
    def _get_vr_complex(self, dim):
        # The incremental algorithm from https://pdfs.semanticscholar.org/e503/c24dcc7a8110a001ae653913ccd064c1044b.pdf
        self.vr = []
        for vertex in range(self.matrix.shape[0]):
            nbrs = self._lower_nbrs(vertex)
            self._add_cofaces(dim, {vertex}, nbrs)

    def _lower_nbrs(self, vertex):
        # look at the column with index `vertex` and return row nums with non-zeros
        relevant_column = self.matrix[:, vertex]
        return np.argwhere(relevant_column == True).flatten()

    def _add_cofaces(self, dim, simplex, nbrs):
        self.vr.append(simplex)
        if len(simplex) >= dim:
            return
        for vertex in nbrs:
            new_simplex = simplex | {vertex}
            intersect = np.intersect1d(nbrs, self._lower_nbrs(vertex))
            self._add_cofaces(dim, new_simplex, intersect)

    def _get_edges(self, vertex):
        relevant_column = self.matrix[:, vertex]
        relevant_row = self.matrix[vertex, :]
        column_nbrs = np.argwhere(relevant_column == True).flatten()
        row_nbrs = np.argwhere(relevant_row == True).flatten()
        return [{vertex, nbr} for nbr in column_nbrs + row_nbrs]

    def _get_relevant_subcomplex(self, simplex):
        return [vr_simplex for vr_simplex in self.vr if simplex <= vr_simplex]

    def _get_isomorphism_dict(self):
        isomorphism_dict = {}
        for vertex in range(self.matrix.shape[0]):
            edges = self._get_edges(vertex)
            for edge in edges:
                localhom_v = self._get_localhom(vertex)
                localhom_e = self._get_localhom(edge)
                isomorphism_dict[(vertex, edge)] = check_isomorphism(localhom_v, localhom_e)
        return isomorphism_dict

    def _get_localhom(self, simplex):
        relevant_subcomplex = self._get_relevant_subcomplex(simplex)
        operators = [get_boundary_operator(relevant_subcomplex, dim) for dim in range(self.dim)]
        betti_numbers = get_betti_numbers(operators)
        return betti_numbers

    def cluster(self):
        pass
