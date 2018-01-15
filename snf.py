# refactor from: http://blog.dlfer.xyz/post/2016-10-27-smith-normal-form/
# this contains a bug (coming from the original code) which seemingly does not affect mod 2 SNF
# look for an example here: https://uk.mathworks.com/help/symbolic/mupad_ref/linalg-smithform.html

import numpy as np


def get_arg_absmin(matrix, s):
    # finds the last entry which is minimal in absolute value and non-zero
    matrix_abs = np.absolute(matrix[s:, s:])
    masked = np.ma.masked_equal(matrix_abs, 0, copy=False)
    index = np.argmin(np.flip(masked.flatten(), 0))
    unraveled = np.unravel_index(masked.size - index - 1, masked.shape)
    return np.array(unraveled) + np.array((s, s))


def swap_rows(matrix, row1, row2):
    matrix[row1], matrix[row2] = matrix[row2], matrix[row1].copy()


def swap_columns(matrix, col1, col2):
    matrix[:, col1], matrix[:, col2] = matrix[:, col2], matrix[:, col1].copy()


def add_row_to_another(matrix, row1, row2, scaling_factor=1):
    matrix[row1] += scaling_factor * matrix[row2]


def add_column_to_another(matrix, col1, col2, scaling_factor=1):
    matrix[:, col1] += scaling_factor * matrix[:, col2]


def change_row_sign(matrix, row):
    matrix[row] = -matrix[row]


def change_sign_column(matrix, column):
    matrix[:, column] = -matrix[:, column]


def is_lone(matrix, s):
    temp_1 = np.argwhere(matrix[s:s + 1, s + 1:] != 0)
    temp_2 = np.argwhere(matrix[s+1:, s:s+1] != 0)
    return not bool(temp_1.size + temp_2.size)


def get_nextentry(matrix, s):
    # find and element which is not divisible by matrix[s][s]
    indexes = np.argwhere(matrix[s + 1:, s + 1:] % matrix[s][s] != 0)
    if not indexes.size:
        return None
    i_row, i_col = indexes[0]
    return i_row + s + 1, i_col + s + 1


def put_in_snf(matrix):
    # puts matrix in Smith Normal Form
    n_rows, n_columns = matrix.shape
    for s in range(min(matrix.shape)):
        while not is_lone(matrix, s):
            row, col = get_arg_absmin(matrix, s)  # the non-zero entry with min |.|
            swap_rows(matrix, s, row)
            swap_columns(matrix, s, col)
            for x_row in range(s + 1, n_rows):
                if matrix[x_row][s]:
                    k = matrix[x_row][s] // matrix[s][s]
                    add_row_to_another(matrix, x_row, s, scaling_factor=-k)
            for x_col in range(s + 1, n_columns):
                if matrix[s][x_col]:
                    k = matrix[s][x_col] // matrix[s][s]
                    add_column_to_another(matrix, x_col, s, scaling_factor=-k)
            if is_lone(matrix, s):
                res = get_nextentry(matrix, s)
                if res:
                    x_row, _ = res
                    add_row_to_another(matrix, s, x_row)
                elif matrix[s][s] < 0:
                    change_row_sign(matrix, s)


def get_snf(matrix):
    # puts matrix in Smith Normal Form and returns left_matrix, right_matrix
    n_rows, n_columns = matrix.shape
    left_matrix = np.identity(n_rows)
    right_matrix = np.identity(n_columns)
    for s in range(min(matrix.shape)):
        while not is_lone(matrix, s):
            row, col = get_arg_absmin(matrix, s)  # the non-zero entry with min |.|
            swap_rows(matrix, s, row)
            swap_rows(left_matrix, s, row)
            swap_columns(matrix, s, col)
            swap_columns(right_matrix, s, col)
            for x_row in range(s + 1, n_rows):
                if matrix[x_row][s]:
                    k = matrix[x_row][s] // matrix[s][s]
                    add_row_to_another(matrix, x_row, s, scaling_factor=-k)
                    add_row_to_another(left_matrix, x_row, s, scaling_factor=-k)
            for x_col in range(s + 1, n_columns):
                if matrix[s][x_col]:
                    k = matrix[s][x_col] // matrix[s][s]
                    add_column_to_another(matrix, x_col, s, scaling_factor=-k)
                    add_column_to_another(right_matrix, x_col, s, scaling_factor=-k)
            if is_lone(matrix, s):
                res = get_nextentry(matrix, s)
                if res:
                    x_row, _ = res
                    add_row_to_another(matrix, s, x_row)
                    add_row_to_another(left_matrix, s, x_row)
                else:
                    if matrix[s][s] < 0:
                        change_row_sign(matrix, s)
                        change_row_sign(left_matrix, s)
    return left_matrix, right_matrix
