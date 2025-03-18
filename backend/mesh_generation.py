import trimesh
import numpy as np
import logging

logger = logging.getLogger(__name__)

def save_mesh_as_obj(mesh: trimesh.Trimesh, filename: str) -> None:
    """
    Save the mesh as an OBJ file, preserving all details including colors, textures, and normals.
    
    Args:
        mesh (trimesh.Trimesh): Mesh to save.
        filename (str): Path to save the OBJ file.
    """
    try:
        logger.info(f"Saving mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
        
        # Ensure vertex colors
        if not hasattr(mesh.visual, 'vertex_colors') or mesh.visual.vertex_colors is None:
            logger.info("No vertex colors found, assigning default gray")
            mesh.visual.vertex_colors = np.full((len(mesh.vertices), 4), [128, 128, 128, 255], dtype=np.uint8)
        
        # Ensure normals
        if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
            logger.info("Computing vertex normals")
            mesh.vertex_normals = trimesh.geometry.weighted_vertex_normals(
                mesh.vertices, mesh.faces, mesh.face_normals, mesh.face_angles
            )
        
        # Ensure UV coordinates if texture exists
        if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
            if hasattr(mesh.visual, 'material') and mesh.visual.material.image is not None:
                logger.info("Generating default UV coordinates for texture")
                mesh.visual.uv = trimesh.geometry.generate_texture_coords(mesh)
        
        # Assign a material
        material_name = "material_0"
        if not hasattr(mesh.visual, 'material') or mesh.visual.material is None:
            mesh.visual.material = trimesh.visual.material.SimpleMaterial(name=material_name)
        
        # Export with all details
        export_options = {
            'include_normals': True,
            'include_color': True,
            'include_texture': True if mesh.visual.uv is not None else False,
            'material_name': material_name
        }
        
        with open(filename, 'w') as f:
            mesh.export(file_obj=f, file_type='obj', **export_options)
            
        logger.info(f"Mesh successfully saved to {filename}")
        
    except Exception as e:
        logger.error(f"Failed to save mesh as OBJ: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to save mesh as OBJ: {str(e)}")

if __name__ == "__main__":
    # Test with a dummy mesh
    logging.basicConfig(level=logging.INFO)
    
    dummy_mesh = trimesh.creation.box(extents=[1, 1, 1])
    # Add random vertex colors
    dummy_mesh.visual.vertex_colors = np.random.randint(0, 255, (dummy_mesh.vertices.shape[0], 4))
    # Add random vertex texture coordinates
    dummy_mesh.visual.uv = np.random.random((dummy_mesh.vertices.shape[0], 2))
    
    save_mesh_as_obj(dummy_mesh, "test_mesh.obj")