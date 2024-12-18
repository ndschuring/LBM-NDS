import jax
import jax.numpy as jnp

# def nabla_test(grid):
#     d = grid.ndim
#     if d == 1:
#         return jnp.gradient(grid, axis=0)
#     if d == 2:
#         return jnp.stack((jnp.gradient(grid, axis=0), jnp.gradient(grid, axis=1)), axis=-1)
#     if d == 3:
#         return jnp.stack((jnp.gradient(grid, axis=0), jnp.gradient(grid, axis=1), jnp.gradient(grid, axis=2)), axis=-1)
def nabla_(grid):
    # using grad_2D from sacha
    return grad_2d(grid).transpose(1, 2, 0)

def nabla(grid):
    # using built-in methods
    return jnp.stack(jnp.gradient(grid), axis=-1)

def laplacian_(grid):
    # low accuracy, causes problems
    d = grid.ndim
    if d == 1:
        gx = jnp.gradient(grid)
        gxx = jnp.gradient(gx, axis=0)
        return gxx
    if d == 2:
        gx, gy = jnp.gradient(grid)
        gxx = jnp.gradient(gx, axis=0)
        gyy = jnp.gradient(gy, axis=1)
        return gxx+gyy
    if d == 3:
        gx, gy, gz = jnp.gradient(grid)
        gxx = jnp.gradient(gx, axis=0)
        gyy = jnp.gradient(gy, axis=1)
        gzz = jnp.gradient(gz, axis=2)
        return gxx+gyy+gzz

def laplacian(grid):
    # works well!
    laplacian_ = jnp.zeros_like(grid)  # Initialize a new array with the same shape as the input grid, filled with zeros
    NX = grid.shape[0]
    NY = grid.shape[1]
    grad_padded = jnp.pad(grid, 2, mode='edge') #-> wrap for periodic, edge for non-periodic
    grad_ineg2_jneg2 = grad_padded[:NX, :NY]
    grad_ineg2_jneg1 = grad_padded[:NX, 1:NY + 1]
    grad_ineg2_j0 = grad_padded[:NX, 2:NY + 2]
    grad_ineg2_jpos1 = grad_padded[:NX, 3:NY + 3]
    grad_ineg2_jpos2 = grad_padded[:NX, 4:NY + 4]

    grad_ineg1_jneg2 = grad_padded[1:NX + 1, :NY]
    grad_ineg1_jneg1 = grad_padded[1:NX + 1, 1:NY + 1]
    grad_ineg1_j0 = grad_padded[1:NX + 1, 2:NY + 2]
    grad_ineg1_jpos1 = grad_padded[1:NX + 1, 3:NY + 3]
    grad_ineg1_jpos2 = grad_padded[1:NX + 1, 4:NY + 4]

    grad_i0_jneg2 = grad_padded[2:NX + 2, :NY]
    grad_i0_jneg1 = grad_padded[2:NX + 2, 1:NY + 1]
    grad_i0_j0 = grad_padded[2:NX + 2, 2:NY + 2]
    grad_i0_jpos1 = grad_padded[2:NX + 2, 3:NY + 3]
    grad_i0_jpos2 = grad_padded[2:NX + 2, 4:NY + 4]

    grad_ipos1_jneg2 = grad_padded[3:NX + 3, :NY]
    grad_ipos1_jneg1 = grad_padded[3:NX + 3, 1:NY + 1]
    grad_ipos1_j0 = grad_padded[3:NX + 3, 2:NY + 2]
    grad_ipos1_jpos1 = grad_padded[3:NX + 3, 3:NY + 3]
    grad_ipos1_jpos2 = grad_padded[3:NX + 3, 4:NY + 4]

    grad_ipos2_jneg2 = grad_padded[4:NX + 4, :NY]
    grad_ipos2_jneg1 = grad_padded[4:NX + 4, 1:NY + 1]
    grad_ipos2_j0 = grad_padded[4:NX + 4, 2:NY + 2]
    grad_ipos2_jpos1 = grad_padded[4:NX + 4, 3:NY + 3]
    grad_ipos2_jpos2 = grad_padded[4:NX + 4, 4:NY + 4]

    laplacian_ = laplacian_.at[:, :].set(
        0 * grad_ineg2_jneg2 + (-1 / 30) * grad_ineg2_jneg1 + (-1 / 60) * grad_ineg2_j0 + (
                -1 / 30) * grad_ineg2_jpos1 + 0 * grad_ineg2_jpos2 +
        (-1 / 30) * grad_ineg1_jneg2 + (4 / 15) * grad_ineg1_jneg1 + (13 / 15) * grad_ineg1_j0 + (
                4 / 15) * grad_ineg1_jpos1 + (-1 / 30) * grad_ineg1_jpos2 +
        (-1 / 60) * grad_i0_jneg2 + (13 / 15) * grad_i0_jneg1 + (-21 / 5) * grad_i0_j0 + (13 / 15) * grad_i0_jpos1 + (
                -1 / 60) * grad_i0_jpos2 +
        (-1 / 30) * grad_ipos1_jneg2 + (4 / 15) * grad_ipos1_jneg1 + (13 / 15) * grad_ipos1_j0 + (
                4 / 15) * grad_ipos1_jpos1 + (-1 / 30) * grad_ipos1_jpos2 +
        0 * grad_ipos2_jneg2 + (-1 / 30) * grad_ipos2_jneg1 + (-1 / 60) * grad_ipos2_j0 + (
                -1 / 30) * grad_ipos2_jpos1 + 0 * grad_ipos2_jpos2)

    return laplacian_

def grad_2d(grid):
    """
    Sascha's version
    :param grid:
    :return:
    """
    # Initialize a 3D array to store the x and y components of the gradient
    grad = jnp.zeros((2, grid.shape[0], grid.shape[1]))

    # Compute the x gradient for the bulk (interior) points
    bulk_x = grad.at[0, 1:-2, :].set((
                                             grid[0:-3, :] -  # Contribution from node to the right
                                             grid[2:-1, :]  # Contribution from node to the left
                                     ) / 2
                                     )

    # Compute the y gradient for the bulk (interior) points
    bulk_y = grad.at[1, :, 1:-2].set((
                                             grid[:, 2:-1] -  # Contribution from node above
                                             grid[:, 0:-3]  # Contribution from node below
                                     ) / 2
                                     )

    # Compute the x gradient for the left edge
    left_x = grad.at[0, 0, :].set((
            grid[0, :] -  # Contribution from the edge
            grid[1, :]  # Contribution from the right
    )
    )

    # Compute the x gradient for the right edge
    right_x = grad.at[0, -1, :].set((
            grid[-2, :] -  # Contribution from the left
            grid[-1, :]  # Contribution from the edge
    )
    )

    # Compute the y gradient for the bottom edge
    bottom_y = grad.at[1, :, 0].set((
            grid[:, 0] -  # Contribution from the edge
            grid[:, 1]  # Contribution from the right
    )
    )

    # Compute the y gradient for the top edge
    top_y = grad.at[1, :, -1].set((
            grid[:, -2] -  # Contribution from the bottom
            grid[:, -1]  # Contribution from the edge
    )
    )

    # Combine the contributions from bulk and edges to get the final gradient
    return bulk_x + bulk_y + left_x + right_x + bottom_y + top_y

def laplacian_2d(grid):
    """
    Sascha's version
    :param grid:
    :return:
    """
    laplacian = jnp.zeros_like(grid)  # Initialize a new array with the same shape as the input grid, filled with zeros

    # bulk
    bulk = laplacian.at[1:-2, 1:-2].set(  # Compute the Laplacian for the bulk (interior) nodes
        grid[2:-1, 1:-2] +  # Contribution from nodes to the right
        grid[0:-3, 1:-2] +  # Contribution from nodes to the left
        grid[1:-2, 2:-1] +  # Contribution from nodes below
        grid[1:-2, 0:-3] -  # Contribution from nodes above
        4 * grid[1:-2, 1:-2]  # Subtract 4 times the central node value
    )

    # left
    left = laplacian.at[0, 1:-2].set(  # Compute the Laplacian for the left edge nodes
        grid[1, 1:-2] +  # Contribution from nodes to the right
        grid[0, 0:-3] +  # Contribution from nodes below
        grid[0, 2:-1] -  # Contribution from nodes above
        3 * grid[0, 1:-2]  # Subtract 3 times the central node value
    )

    # right
    right = laplacian.at[-1, 1:-2].set(  # Compute the Laplacian for the bottom edge nodes
        grid[-2, 1:-2] +  # Contribution from nodes to the left
        grid[-1, 0:-3] +  # Contribution from nodes below
        grid[-1, 2:-1] -  # Contribution from nodes above
        3 * grid[-1, 1:-2]  # Subtract 3 times the central node value
    )

    # bottom
    bottom = laplacian.at[1:-2, 0].set(  # Compute the Laplacian for the left edge nodes
        grid[1:-2, 1] +  # Contribution from nodes above
        grid[0:-3, 0] +  # Contribution from nodes to the left
        grid[2:-1, 0] -  # Contribution from nodes to the right
        3 * grid[1:-2, 0]  # Subtract 3 times the central node value
    )

    # top
    top = laplacian.at[1:-2, -1].set(  # Compute the Laplacian for the right edge nodes
        grid[1:-2, -2] +  # Contribution from nodes below
        grid[0:-3, -1] +  # Contribution from nodes to the left
        grid[2:-1, -1] -  # Contribution from nodes to the right
        3 * grid[1:-2, -1]  # Subtract 3 times the central node value
    )

    # bottom-left
    bottomleft = laplacian.at[0, 0].set(  # Compute the Laplacian for the top-left corner node
        grid[0, 1] +  # Contribution from node above
        grid[1, 0] -  # Contribution from node to the right
        2 * grid[0, 0]  # Subtract 2 times the central node value
    )

    # topleft
    topleft = laplacian.at[0, -1].set(  # Compute the Laplacian for the top-right corner node
        grid[0, -2] +  # Contribution from node below
        grid[1, -1] -  # Contribution from node to the right
        2 * grid[0, -1]  # Subtract 2 times the central node value
    )

    # bottomright
    bottomright = laplacian.at[-1, 0].set(  # Compute the Laplacian for the bottom-left corner node
        grid[-2, 0] +  # Contribution from node to the left
        grid[-1, 1] -  # Contribution from node above
        2 * grid[-1, 0]  # Subtract 2 times the central node value
    )

    # topright
    topright = laplacian.at[-1, -1].set(  # Compute the Laplacian for the bottom-right corner node
        grid[-1, -2] +  # Contribution from node below
        grid[-2, -1] -  # Contribution from node to the left
        2 * grid[-1, -1]  # Subtract 2 times the central node value
    )

    return bulk + top + bottom + left + right + topleft + topright + bottomleft + bottomright




if __name__ == '__main__':
    nx, ny, nz = 5, 6, 7
    # Create a random key
    key = jax.random.PRNGKey(0)
    # Generate a random matrix
    testgrid = jnp.array([[1, 2, 1],
                          [0, 0, 1],
                          [3, 2, 1]])
    # grid2D = jax.random.uniform(key, shape=(nx, ny))
    # grid3D = jax.random.uniform(key, shape=(nx, ny, nz))
    # # print(testgrid.shape)
    # print(nabla(testgrid).shape)
    # print(nabla(testgrid)[:, :, 0])
    # print(nabla(testgrid)[:, :, 1])
    # print(10 * "-")
    # print(grad_2d(testgrid)[0])
    # print(grad_2d(testgrid)[1])
    # print(10*"-")
    # print(jnp.gradient(testgrid)[0])
    # print(jnp.gradient(testgrid)[1])
    # print(10*"-")
    print(laplacian(testgrid))
    print(nabla(testgrid))
    print(nabla_(testgrid))
    print(grad_2d(testgrid))
    # print(nabla(grid3D).shape)
    # gx, gy = jnp.gradient(grid2D)
    # print(jnp.gradient(grid2D).shape)
    # gx2, gy2 = nabla(grid2D)

    # print(gx, gy)
    # print("----------")
    # print(gx2, gy2)