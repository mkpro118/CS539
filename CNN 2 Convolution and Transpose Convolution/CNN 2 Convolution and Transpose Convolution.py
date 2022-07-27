# Cell 1
# Setup

import numpy as np

in_bounds = lambda t, f, x: t + f <= x

# Cell 2
# Question 1, setup

out_size = lambda x, f, p, s: (x - f + 2 * p) // s + 1
extract = lambda x, t, l, f: x[t:t + f, l: l + f]


def convolve(
        X: np.ndarray,
        kernel: np.ndarray,
        stride: int = None,
        padding: int = None
    ) -> np.ndarray:

    stride = stride or 1
    padding = padding or 0

    kernel_size = len(kernel)

    output_height = out_size(X.shape[0], kernel_size, padding, stride)
    output_width = out_size(X.shape[1], kernel_size, padding, stride)

    output = np.zeros((output_height, output_width))

    input_ = X

    if padding:
        zeros = np.zeros((input_.shape[0], padding))
        input_ = np.hstack((
            _ := zeros.reshape(input_.shape[0], -1),
            input_,
            _
        ))

        zeros = np.zeros((input_.shape[1], padding))
        input_ = np.vstack((
            _ := zeros.reshape(-1, input_.shape[1]),
            input_,
            _
        ))

    input_height, input_width = input_.shape

    # Code for convolution
    top, left, out_x, out_y = 0, 0, 0, 0
    while in_bounds(top, kernel_size, input_height):

        output[out_y, out_x] = np.sum(extract(input_, top, left, kernel_size) * kernel)
        out_x += 1
        left += stride
        if not in_bounds(left, kernel_size, input_width):
            left, out_x, out_y = 0, 0, out_y + 1
            top += stride
    return output

# Cell 3
A = np.array([
    [4, 9, 4, 5, 6],
    [8, 3, 6, 8, 5],
    [6, 8, 1, 9, 0],
    [5, 8, 1, 1, 3],
])

B = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0],
])

C = convolve(A, B, stride=2, padding=1)
print(f'Convolved Matrix\n{C}')


# Cell 4
def convolve_transpose(
        X: np.ndarray,
        kernel: np.ndarray,
        stride: int = None,
        padding: int = None
    ) -> np.ndarray:
    stride = stride or 1
    padding = padding or 0

    input_ = X

    zeros, i = np.zeros((input_.shape[0])), 1
    i = 1
    while in_bounds(i, stride - 1, input_.shape[1]):
        input_ = np.insert(input_, i, zeros, axis=1)
        i += stride

    zeros, i = np.zeros((input_.shape[1])), 1
    while in_bounds(i, stride - 1, input_.shape[0]):
        input_ = np.insert(input_, i, zeros, axis=0)
        i += stride

    if padding:
        zeros = np.zeros((input_.shape[0], padding))
        input_ = np.hstack((
            _ := zeros.reshape(input_.shape[0], -1),
            input_,
            _
        ))

        zeros = np.zeros((input_.shape[1], padding))
        input_ = np.vstack((
            _ := zeros.reshape(-1, input_.shape[1]),
            input_,
            _
        ))

    output = convolve(input_, kernel, stride=1, padding=padding)
    return output[
        padding: output.shape[0] - padding,
        padding: output.shape[1] - padding
    ]


# Cell 5
A = np.array([
    [9, 4, 5, ],
    [3, 6, 8, ],
    [8, 1, 9, ],
])

B = np.array([
    [0, 1, 0],
    [1, 4, 1],
    [0, 1, 0],
])
B = B / 8.

C = convolve_transpose(A, B, stride=2, padding=1)
print(f'Transpose Convolved Matrix\n{C}')
