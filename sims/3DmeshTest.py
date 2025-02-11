import trimesh
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # mesh = trimesh.load_mesh('C:/Users/ndsch/PycharmProjects/LBM-NDS/src/masks/channel_grooves.3mf')
    # mesh = trimesh.load_mesh('C:/Users/ndsch/PycharmProjects/LBM-NDS/src/masks/channel_grooves.3mf')
    mesh = trimesh.load('C:/Users/ndsch/PycharmProjects/LBM-NDS/src/masks/channel_grooves.3mf', force="mesh")
    mesh = trimesh.load('C:/Users/ndsch/PycharmProjects/LBM-NDS/src/masks/channel_grooves.stl')
    print(mesh.is_watertight)
    print(mesh.bounding_box.extents)
    resolution = 30
    pitch = mesh.bounding_box.extents.min() / resolution
    # mesh = mesh.voxelized(pitch=mesh.bounding_box.extents[0] / resolution).fill()
    # mesh = mesh.voxelized(pitch=mesh.bounding_box.extents.min() / resolution).fill()
    mesh = trimesh.voxel.creation.voxelize_binvox(mesh, pitch=pitch).fill()
    mesh.show()
    boolean_array = mesh.matrix
    boolean_array = np.asarray(boolean_array)
    boolean_array = ~boolean_array
    grid_padded = np.pad(boolean_array, ((1, 1), (0, 0), (1, 1)), mode='constant', constant_values=True)  # Horizontal axis
    print(grid_padded.shape)
    # print(~boolean_array)
    # print("test")