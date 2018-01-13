import numpy as np

from collections import Counter
from snf import Smith


def get_boundary_operator(simplicial_complex, k):
    domain_basis = [simplex for simplex in simplicial_complex if len(simplex) == k + 1]
    codomain_basis = [simplex for simplex in simplicial_complex if len(simplex) == k]
    if not k:
        return np.array([[0 for simplex in domain_basis]])
    if not domain_basis:
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


def get_betti_numbers(boundary_operators):
    betti_numbers = []
    prev_dim_kernel = 0
    for i, operator in enumerate(boundary_operators):
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

        prev_betti = prev_dim_kernel - dim_image
        if prev_betti < 0:
            prev_betti = 0
        prev_dim_kernel = dim_kernel
        if i == 0:
            continue

        betti_numbers.append(prev_betti)
    betti_numbers.append(prev_dim_kernel) # last betti number is just last_kernel_dim - 0
    return betti_numbers


def get_node_nbrs(node, graph_matrix):
    column = graph_matrix[:, node]
    row = graph_matrix[node, :]
    column_nbrs = np.argwhere(column == True).flatten()
    row_nbrs = np.argwhere(row == True).flatten()
    return np.concatenate((column_nbrs, row_nbrs))
