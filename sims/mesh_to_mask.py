from pathlib import Path
import numpy as np
import trimesh
import os

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent
    mask_path = project_root / "src" / "masks"
    # model_files = [f for f in os.listdir(str(mask_path)) if f.lower().endswith(('stl', '3mf'))]
    model_files = [f for f in os.listdir(str(mask_path)) if f.lower().endswith(('stl', '3mf')) and not os.path.exists(os.path.join(str(mask_path), os.path.splitext(f)[0] + '.npy'))]
    print(f"Files to convert: {model_files}")
    for model_file in model_files:
        model_path = mask_path / model_file
        mesh_name = Path(model_path).stem + ".npy"
        print(f"\nConverting {model_file} to {mesh_name}")
        mesh = trimesh.load(str(model_path))
        if not mesh.is_watertight:
            raise RuntimeError("Mesh should be watertight")
        print(f"Mesh dimensions (z, x, y): {mesh.bounding_box.extents}")
        resolution = 5
        pitch = mesh.bounding_box.extents.min() / resolution
        mesh = mesh.voxelized(pitch=pitch, max_iter=30).fill()
        # mesh.show()
        boolean_array = mesh.matrix.transpose(1, 2, 0)
        boolean_array = ~boolean_array
        collision_mask = np.pad(boolean_array, ((1, 1), (0, 0), (1, 1)), mode='constant', constant_values=True)
        print(f"Collision mask shape: {collision_mask.shape}")
        save_path = project_root / "src" / "masks" / mesh_name
        np.save(str(save_path), collision_mask)
        print(f"Collision mask saved to {save_path}")
