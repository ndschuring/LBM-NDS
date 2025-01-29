import jax
import jax.numpy as jnp
from PIL import Image
import re
import os


def nabla(grid):
    # using built-in methods
    return jnp.stack(jnp.gradient(grid), axis=-1)

def laplacian(grid):
    # works well!
    laplacian_ = jnp.zeros_like(grid)  # Initialize a new array with the same shape as the input grid, filled with zeros
    NX = grid.shape[0]
    NY = grid.shape[1]
    # grid_padded = jnp.pad(grid, 2, mode='edge') #-> wrap for periodic, edge for non-periodic
    grid_padded = jnp.pad(grid, ((2, 2), (0, 0)), mode='edge') # Horizontal axis
    grid_padded = jnp.pad(grid_padded, ((0, 0), (2, 2)), mode='edge')  # Vertical Axis
    grid_ineg2_jneg2 = grid_padded[:NX, :NY]
    grid_ineg2_jneg1 = grid_padded[:NX, 1:NY + 1]
    grid_ineg2_j0 = grid_padded[:NX, 2:NY + 2]
    grid_ineg2_jpos1 = grid_padded[:NX, 3:NY + 3]
    grid_ineg2_jpos2 = grid_padded[:NX, 4:NY + 4]

    grid_ineg1_jneg2 = grid_padded[1:NX + 1, :NY]
    grid_ineg1_jneg1 = grid_padded[1:NX + 1, 1:NY + 1]
    grid_ineg1_j0 = grid_padded[1:NX + 1, 2:NY + 2]
    grid_ineg1_jpos1 = grid_padded[1:NX + 1, 3:NY + 3]
    grid_ineg1_jpos2 = grid_padded[1:NX + 1, 4:NY + 4]

    grid_i0_jneg2 = grid_padded[2:NX + 2, :NY]
    grid_i0_jneg1 = grid_padded[2:NX + 2, 1:NY + 1]
    grid_i0_j0 = grid_padded[2:NX + 2, 2:NY + 2]
    grid_i0_jpos1 = grid_padded[2:NX + 2, 3:NY + 3]
    grid_i0_jpos2 = grid_padded[2:NX + 2, 4:NY + 4]

    grid_ipos1_jneg2 = grid_padded[3:NX + 3, :NY]
    grid_ipos1_jneg1 = grid_padded[3:NX + 3, 1:NY + 1]
    grid_ipos1_j0 = grid_padded[3:NX + 3, 2:NY + 2]
    grid_ipos1_jpos1 = grid_padded[3:NX + 3, 3:NY + 3]
    grid_ipos1_jpos2 = grid_padded[3:NX + 3, 4:NY + 4]

    grid_ipos2_jneg2 = grid_padded[4:NX + 4, :NY]
    grid_ipos2_jneg1 = grid_padded[4:NX + 4, 1:NY + 1]
    grid_ipos2_j0 = grid_padded[4:NX + 4, 2:NY + 2]
    grid_ipos2_jpos1 = grid_padded[4:NX + 4, 3:NY + 3]
    grid_ipos2_jpos2 = grid_padded[4:NX + 4, 4:NY + 4]

    laplacian_ = laplacian_.at[:, :].set(
        0 * grid_ineg2_jneg2 + (-1 / 30) * grid_ineg2_jneg1 + (-1 / 60) * grid_ineg2_j0 + (
                -1 / 30) * grid_ineg2_jpos1 + 0 * grid_ineg2_jpos2 +
        (-1 / 30) * grid_ineg1_jneg2 + (4 / 15) * grid_ineg1_jneg1 + (13 / 15) * grid_ineg1_j0 + (
                4 / 15) * grid_ineg1_jpos1 + (-1 / 30) * grid_ineg1_jpos2 +
        (-1 / 60) * grid_i0_jneg2 + (13 / 15) * grid_i0_jneg1 + (-21 / 5) * grid_i0_j0 + (13 / 15) * grid_i0_jpos1 + (
                -1 / 60) * grid_i0_jpos2 +
        (-1 / 30) * grid_ipos1_jneg2 + (4 / 15) * grid_ipos1_jneg1 + (13 / 15) * grid_ipos1_j0 + (
                4 / 15) * grid_ipos1_jpos1 + (-1 / 30) * grid_ipos1_jpos2 +
        0 * grid_ipos2_jneg2 + (-1 / 30) * grid_ipos2_jneg1 + (-1 / 60) * grid_ipos2_j0 + (
                -1 / 30) * grid_ipos2_jpos1 + 0 * grid_ipos2_jpos2)

    return laplacian_

def mask_from_image(image):
    """
    Initialize a boolean mask from a greyscale image.
    black pixel -> wall
    :param image: numpy.ndarray or PIL.Image
    :return: JAX boolean array
    """
    if type(image) is None:
        raise ValueError("Image is NoneType")
    collision_mask = jnp.array(image==0).T[:, ::-1]
    return collision_mask

def initialize_circle(nx, ny, r):
    """
    Initialize a boolean grid of dimensions (nx, ny) with False everywhere, and set points
    within a circle of radius r at the center to True.
    :param nx: Number of rows in the grid.
    :param ny: Number of columns in the grid.
    :param r: Radius of the circle at the center.
    :return: jnp.ndarray: A 2D boolean array representing the initialized grid.
    """
    # Create a grid of indices
    x = jnp.arange(nx)
    y = jnp.arange(ny)
    xx, yy = jnp.meshgrid(x, y, indexing='ij')
    # Compute the distance from the center
    center_x, center_y = nx // 2, ny // 2
    distance_from_center = jnp.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    # Create a boolean grid where True is inside the circle and False is outside
    grid = distance_from_center <= r
    return grid, nx, ny

def place_circle(grid, r, value=1, center_x=None, center_y=None):
    """
    Place a circle of radius r at centre of provided grid, with value of value.
    :param grid:
    :param r: Radius of the circle at the center.
    :return: jnp.ndarray: A 2D array representing the initialized grid.
    """
    nx, ny = grid.shape
    # Create a grid of indices
    x = jnp.arange(nx)
    y = jnp.arange(ny)
    xx, yy = jnp.meshgrid(x, y, indexing='ij')
    # Compute the distance from the center
    if center_x is None:
        center_x = nx // 2
    if center_y is None:
        center_y = ny // 2
    distance_from_center = jnp.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    # Create a boolean grid where True is inside the circle and False is outside
    boolean_circle = distance_from_center <= r
    grid = jnp.where(boolean_circle, value, grid)
    return grid


def images_to_gif(image_folder, duration=100):
    def extract_iteration_number(filename):
        match = re.search(r'it(\d+)', filename)
        return int(match.group(1)) if match else float('inf')
    print("Creating GIF...")
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))]
    if not image_files:
        print("No images found in the specified folder.")
        return
    image_files.sort(key=extract_iteration_number)
    frames = [Image.open(os.path.join(image_folder, img)) for img in image_files]
    gif_path = os.path.join(image_folder, "output.gif")
    frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)
    print(f"GIF saved as {gif_path}")

def nabla_(grid):
    # using grad_2D from sacha, unused
    return grad_2d(grid).transpose(1, 2, 0)

def laplacian_(grid):
    # low accuracy, causes problems, unused
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

def grad_2d(grid):
    """
    Sascha's version
    unused
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
    unused
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
    testgrid2 = jnp.zeros((50, 50))
    # testgrid2 = jax.random.uniform(key, shape=(50, 50))
    # testgrid2 = place_circle(testgrid2, 4)
    # debug_numpy = np.asarray(testgrid2)
    # grid2D = jax.random.uniform(key, shape=(nx, ny))
    laplace = laplacian(testgrid2)
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
    # print(laplacian(testgrid))
    # print(nabla(testgrid))
    # print(nabla_(testgrid))
    # print(grad_2d(testgrid))
    # print(nabla(grid3D).shape)
    # gx, gy = jnp.gradient(grid2D)
    # print(jnp.gradient(grid2D).shape)
    # gx2, gy2 = nabla(grid2D)

    # print(gx, gy)
    # print("----------")
    # print(gx2, gy2)