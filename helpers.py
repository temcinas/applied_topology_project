import numpy as np

from collections import Counter
from itertools import combinations
from snf import Smith


def get_boundary_operator(simplicial_complex, k):
    domain_basis = [simplex for simplex in simplicial_complex if len(simplex) == k + 1]
    codomain_basis = [simplex for simplex in simplicial_complex if len(simplex) == k]
    if not k:
        return np.array([[0 for simplex in domain_basis]])
    if not domain_basis:
        # return np.array([[0] for simplex in codomain_basis])
        return np.array([])
    operator = []
    for cod_simplex in codomain_basis:
        row = []
        for d_simplex in domain_basis:
            if cod_simplex <= d_simplex:
                row.append(1)
            else:
                row.append(0)
        operator.append(row)
    return np.array(operator)


def swap_rows(matrix, i, j):
    temp = np.copy(matrix[i, :])
    matrix[i, :] = matrix[j, :]
    matrix[j, :] = temp


def check_isomorphism(homology_groups1, homology_groups2):
    # homology_groups will be a list of betti numbers
    # betti numbers completely pin down homology groups because we are working over Z_2 - a field.
    return homology_groups1 == homology_groups2


def get_betti_numbers(boundary_operators):
    betti_numbers = []
    prev_dim_kernel = 0
    for i, operator in enumerate(boundary_operators):
        # print(i, operator)
        snf = operator.tolist()
        if not snf:
            dim_kernel = 0
            dim_image = 0
        else:
            Smith(snf)
            snf = np.array(snf) % 2  # for it to be over Z_2
            snf = np.transpose(snf)  # now we can iterate over columns
            zero_counter = Counter([column.any() for column in snf])
            dim_image = zero_counter.get(True, 0)
            dim_kernel = zero_counter.get(False, 0)
        # print("current kernel: {0}, current image: {1}".format(dim_kernel, dim_image))
        # print("prev kernel: {0}".format(prev_dim_kernel))
        prev_betti = prev_dim_kernel - dim_image
        if prev_betti < 0:
            prev_betti = 0
        prev_dim_kernel = dim_kernel
        if i == 0:
            continue
        # print('betti number {0}'.format(prev_betti))
        betti_numbers.append(prev_betti)
    betti_numbers.append(prev_dim_kernel) # last betti number is just last_kernel_dim - 0
    return betti_numbers


def get_powerset(given_set):
    # we don't include the empty set
    pset = []
    for n in range(len(given_set) + 1):
        for sset in combinations(given_set, n):
            if sset:
                pset.append(set(sset))
    return pset


def fill_in_complex(simplex_list):
    complex = []
    for simplex in simplex_list:
        powerset = get_powerset(simplex)
        for sset in powerset:
            if sset not in complex:
                complex.append(sset)
    return complex


def get_node_nbrs(node, graph_matrix):
    column = graph_matrix[:, node]
    row = graph_matrix[node, :]
    column_nbrs = np.argwhere(column == True).flatten()
    row_nbrs = np.argwhere(row == True).flatten()
    return np.concatenate((column_nbrs, row_nbrs))


def visit_nbrs(node, graph_matrix, visited, cluster):
    nbrs = get_node_nbrs(node, graph_matrix)
    for nbr in nbrs:
        if_visited = visited.get(nbr, False)
        if not if_visited:
            visited[nbr] = True
            cluster.append(nbr)
            visit_nbrs(nbr, graph_matrix, visited, cluster)


def get_snf(matrix):
    raise NotImplementedError # not in use at the moment
    # we follow the Wikipedia's algorithm from https://en.wikipedia.org/wiki/Smith_normal_form#Algorithm
    rows = matrix.shape[1]
    prev = -1
    for t in range(rows):
        # Step I: Choosing a pivot
        non_zero_indicies = np.transpose(np.argwhere(matrix != 0))
        non_zero_clms = np.transpose(np.argwhere(matrix != 0))[1]
        # TODO: What if we run out and have an empty array? Need to deal with that
        j = min(np.argwhere(prev < non_zero_clms).flatten())
        if not matrix[t, j]:
            place = np.argwhere(non_zero_clms == j)[0][0]
            relevant_rownum = non_zero_indicies[0][place]
            swap_rows(matrix, j, relevant_rownum)

        # Step II: Improving the pivot
        # Can skip step II because our element at t,j is non-zero and hence will never have a new element which is
        # Not divisible by the element at t,j (since we are working over Z_2)
        # Step III: Eliminating entries

