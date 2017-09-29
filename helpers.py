import numpy as np


def get_boundary_operator(simplicial_complex, k):
    domain_basis = [simplex for simplex in simplicial_complex if len(simplex) == k + 1]
    codomain_basis = [simplex for simplex in simplicial_complex if len(simplex) == k]
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


def get_snf(matrix):
    # we follow the Wikipedia's algorithm from https://en.wikipedia.org/wiki/Smith_normal_form#Algorithm
    rows = matrix.shape[1]
    prev = -1
    for t in range(rows):
        # Step I: Choosing a pivot
        non_zero_indicies = np.transpose(np.argwhere(matrix != 0))
        non_zero_clms = np.transpose(np.argwhere(matrix != 0))[1]
        j = min(np.argwhere(prev < non_zero_clms).flatten())
        if not matrix[t, j]:
            place = np.argwhere(non_zero_clms == j)[0][0]
            relevant_rownum = non_zero_indicies[0][place]
            swap_rows(matrix, j, relevant_rownum)

        # Step II: Improving the pivot

