import trimesh
import numpy as np
import logging
import os
import time
import gc  # Add garbage collection
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@contextmanager
def time_operation(operation_name):
    """Context manager to time operations and log their duration."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.info(f"{operation_name} completed in {duration:.2f} seconds")

def save_mesh_as_obj(mesh: trimesh.Trimesh, filename: str, max_retries: int = 3, chunk_size: int = 5000) -> None:
    """
    Save the mesh as an OBJ file with improved XPU support and memory optimization.
    Uses chunking for large meshes to prevent memory issues.
    
    Args:
        mesh (trimesh.Trimesh): Mesh to save.
        filename (str): Path to save the OBJ file.
        max_retries (int): Maximum number of retries if saving fails.
        chunk_size (int): Number of vertices to process at once for large meshes.
    """
    import gc
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)) if os.path.dirname(filename) else '.', exist_ok=True)
    
    # Force garbage collection at the start
    gc.collect()
    
    # Convert tensor mesh to numpy if needed
    if hasattr(mesh.vertices, 'device'):
        logger.info("Converting tensor mesh to numpy for saving")
        try:
            # Do conversion in chunks to save memory
            vertices_count = len(mesh.vertices)
            face_count = len(mesh.faces)
            
            logger.info(f"Converting {vertices_count} vertices and {face_count} faces")
            
            # Process vertices in chunks if it's a large mesh
            if vertices_count > 10000:
                vertices_chunks = []
                for i in range(0, vertices_count, chunk_size):
                    end_idx = min(i + chunk_size, vertices_count)
                    if hasattr(mesh.vertices, 'cpu'):
                        chunk = mesh.vertices[i:end_idx].cpu().numpy()
                    else:
                        chunk = np.array(mesh.vertices[i:end_idx])
                    vertices_chunks.append(chunk)
                    gc.collect()  # Force garbage collection after each chunk
                
                vertices = np.concatenate(vertices_chunks)
                del vertices_chunks
                gc.collect()
            else:
                vertices = mesh.vertices.cpu().numpy() if hasattr(mesh.vertices, 'cpu') else np.array(mesh.vertices)
            
            # Process faces in chunks if it's a large mesh
            if face_count > 10000:
                faces_chunks = []
                for i in range(0, face_count, chunk_size):
                    end_idx = min(i + chunk_size, face_count)
                    if hasattr(mesh.faces, 'cpu'):
                        chunk = mesh.faces[i:end_idx].cpu().numpy() 
                    else:
                        chunk = np.array(mesh.faces[i:end_idx])
                    faces_chunks.append(chunk)
                    gc.collect()  # Force garbage collection after each chunk
                
                faces = np.concatenate(faces_chunks)
                del faces_chunks
                gc.collect()
            else:
                faces = mesh.faces.cpu().numpy() if hasattr(mesh.faces, 'cpu') else np.array(mesh.faces)
            
            # Create a new mesh with numpy arrays
            numpy_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Transfer visual properties if available - in chunks for large meshes
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                if vertices_count > 10000:
                    color_chunks = []
                    for i in range(0, vertices_count, chunk_size):
                        end_idx = min(i + chunk_size, vertices_count)
                        if hasattr(mesh.visual.vertex_colors, 'cpu'):
                            chunk = mesh.visual.vertex_colors[i:end_idx].cpu().numpy()
                        else:
                            chunk = np.array(mesh.visual.vertex_colors[i:end_idx])
                        color_chunks.append(chunk)
                        gc.collect()
                    
                    numpy_mesh.visual.vertex_colors = np.concatenate(color_chunks)
                    del color_chunks
                else:
                    numpy_mesh.visual.vertex_colors = mesh.visual.vertex_colors.cpu().numpy() if hasattr(mesh.visual.vertex_colors, 'cpu') else np.array(mesh.visual.vertex_colors)
            
            # Use the numpy mesh for saving
            mesh = numpy_mesh
            
            # Force garbage collection after conversion
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to convert tensor mesh to numpy: {e}")
    
    # Check if mesh is too large and needs simplification - use more aggressive simplification
    vertices_count = len(mesh.vertices)
    if vertices_count > 10000:
        logger.info(f"Mesh is very large ({vertices_count} vertices), applying aggressive simplification")
        try:
            target_faces = min(5000, len(mesh.faces) // 3)  # More aggressive reduction
            original_faces = len(mesh.faces)
            mesh = mesh.simplify_quadratic_decimation(target_faces)
            logger.info(f"Simplified mesh from {original_faces} to {len(mesh.faces)} faces")
            gc.collect()  # Force garbage collection after simplification
        except Exception as e:
            logger.warning(f"Failed to simplify mesh: {e}")
    
    # Reduce chunk size for very large meshes
    if vertices_count > 50000:
        chunk_size = min(1000, chunk_size)
    
    # Write the OBJ file directly with ultra-small chunks for memory-constrained environments
    retries = 0
    while retries < max_retries:
        try:
            logger.info(f"Saving mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
            
            # Check if the mesh has valid faces and vertices
            if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
                logger.warning("Mesh has no faces or vertices, creating a placeholder cube")
                mesh = trimesh.creation.box(extents=[1, 1, 1])
            
            # Write the OBJ file directly for better memory efficiency
            try:
                logger.info("Writing OBJ file directly using ultra memory-efficient approach")
                with open(filename, 'w') as f:
                    f.write("# OBJ file generated by Text-to-3D with memory optimization\n")
                    
                    # Process vertices in tiny chunks
                    optimal_chunk = min(chunk_size, max(100, len(mesh.vertices) // 20))
                    logger.info(f"Using chunk size of {optimal_chunk} for writing")
                    
                    # Write vertices in small chunks
                    for i in range(0, len(mesh.vertices), optimal_chunk):
                        end_i = min(i + optimal_chunk, len(mesh.vertices))
                        for v in mesh.vertices[i:end_i]:
                            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                        
                        # Free memory after each small chunk
                        if (i // optimal_chunk) % 5 == 0:
                            gc.collect()
                    
                    # Write face normals directly instead of vertex normals to save memory
                    if hasattr(mesh, 'face_normals') and mesh.face_normals is not None:
                        for i in range(0, len(mesh.face_normals), optimal_chunk):
                            end_i = min(i + optimal_chunk, len(mesh.face_normals))
                            for n in mesh.face_normals[i:end_i]:
                                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
                            
                            # Free memory
                            if (i // optimal_chunk) % 5 == 0:
                                gc.collect()
                    # Skip UV coordinates to save memory unless necessary
                    
                    # Write faces with extremely small chunks
                    for i in range(0, len(mesh.faces), optimal_chunk):
                        end_i = min(i + optimal_chunk, len(mesh.faces))
                        for face in mesh.faces[i:end_i]:
                            # Simple face format without UVs or normals to save memory
                            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
                        
                        # Free memory very frequently
                        if (i // optimal_chunk) % 3 == 0:
                            gc.collect()
                
                logger.info(f"Mesh successfully saved to {filename}")
                return
            except Exception as direct_write_err:
                logger.warning(f"Direct OBJ writing failed: {direct_write_err}, trying simpler approach")
                
                # Try an even simpler approach - just vertices and faces
                with open(filename, 'w') as f:
                    f.write("# OBJ file - simplified output\n")
                    
                    # Write vertices line by line
                    for i, v in enumerate(mesh.vertices):
                        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                        if i % 1000 == 0:
                            gc.collect()  # Force GC more frequently
                    
                    # Write faces line by line
                    for i, face in enumerate(mesh.faces):
                        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
                        if i % 1000 == 0:
                            gc.collect()  # Force GC more frequently
                
                logger.info(f"Mesh saved using simplified approach to {filename}")
                return
            
        except Exception as e:
            retries += 1
            logger.error(f"Attempt {retries}/{max_retries} to save mesh failed: {str(e)}")
            
            # Force garbage collection
            gc.collect()
            
            if retries >= max_retries:
                logger.error(f"Failed to save mesh after {max_retries} attempts")
                # Create and save a dummy mesh as fallback
                logger.warning("Creating and saving a dummy cube as fallback")
                from mesh_generation import create_dummy_mesh
                dummy_mesh = create_dummy_mesh()
                try:
                    # Save the dummy mesh in the simplest possible way
                    with open(filename, 'w') as f:
                        f.write("# Fallback cube OBJ\n")
                        for v in dummy_mesh.vertices:
                            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                        for face in dummy_mesh.faces:
                            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
                    return
                except Exception as dummy_err:
                    logger.error(f"Failed to save dummy mesh: {dummy_err}")
                    raise RuntimeError(f"Failed to save mesh as OBJ: {str(e)}")
            
            # If we're here, we'll retry with a greatly simplified version of the mesh
            logger.warning(f"Aggressively simplifying mesh for retry attempt {retries}")
            try:
                # Further reduce complexity for each retry
                target_faces = max(len(mesh.faces) // (retries * 5), 500)  # Much more aggressive reduction
                
                logger.info(f"Simplifying mesh to just {target_faces} faces")
                mesh = mesh.simplify_quadratic_decimation(target_faces)
                
                # Force garbage collection
                gc.collect()
            except Exception as simplify_e:
                logger.warning(f"Mesh simplification failed: {simplify_e}, creating minimal mesh")
                try:
                    # Create a minimal version of the mesh - just a simplified convex hull
                    try:
                        hull = mesh.convex_hull
                        if len(hull.faces) > 100:
                            mesh = hull.simplify_quadratic_decimation(100)
                        else:
                            mesh = hull
                    except:
                        # If convex hull fails, create a dummy mesh
                        mesh = create_dummy_mesh()
                    
                    # Force garbage collection
                    gc.collect()
                except Exception as cleanup_err:
                    logger.warning(f"Basic mesh cleanup failed: {cleanup_err}, using dummy mesh")
                    mesh = create_dummy_mesh()
                
            # Delay before retrying
            time.sleep(1)
def create_dummy_mesh(size: float = 1.0, color: list = None) -> trimesh.Trimesh:
    """
    Create a simple cube mesh for cases where mesh generation fails.
    
    Args:
        size (float): Size of the cube.
        color (list): RGB color values as a list [r, g, b] where each value is in range [0, 255].
                     Defaults to gray if None is provided.
    
    Returns:
        trimesh.Trimesh: A simple cube mesh.
    """
    if color is None:
        color = [128, 128, 128, 255]  # Default gray color with full opacity
    elif len(color) == 3:
        color = color + [255]  # Add alpha channel if not provided
    
    # Create a cube mesh
    mesh = trimesh.creation.box(extents=[size, size, size])
    
    # Assign color to all vertices
    vertex_colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
    vertex_colors[:] = color
    mesh.visual.vertex_colors = vertex_colors
    
    return mesh

def create_uv_sphere(radius: float = 1.0, count: list = [32, 32], color: list = None) -> trimesh.Trimesh:
    """
    Create a UV sphere mesh as an alternative primitive.
    
    Args:
        radius (float): Radius of the sphere.
        count (list): Resolution [vertical, horizontal].
        color (list): RGB color values as a list [r, g, b].
                     Defaults to gray if None provided.
    
    Returns:
        trimesh.Trimesh: A spherical mesh.
    """
    if color is None:
        color = [128, 128, 128, 255]
    elif len(color) == 3:
        color = color + [255]  # Add alpha channel
    
    try:
        # Create a sphere mesh
        mesh = trimesh.creation.uv_sphere(radius=radius, count=count)
        
        # Assign color to all vertices
        vertex_colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
        vertex_colors[:] = color
        mesh.visual.vertex_colors = vertex_colors
        
        return mesh
    except Exception as e:
        logger.warning(f"Failed to create UV sphere: {e}, falling back to box")
        return create_dummy_mesh(size=radius*2, color=color)

def merge_meshes(meshes: list) -> trimesh.Trimesh:
    """
    Merge multiple meshes into a single mesh.
    
    Args:
        meshes (list): List of trimesh.Trimesh objects to merge.
    
    Returns:
        trimesh.Trimesh: A single merged mesh.
    """
    if not meshes:
        logger.warning("No meshes to merge, returning dummy mesh")
        return create_dummy_mesh()
    
    if len(meshes) == 1:
        return meshes[0]
    
    try:
        # Use trimesh's built-in concatenation
        return trimesh.util.concatenate(meshes)
    except Exception as e:
        logger.error(f"Failed to merge meshes: {e}")
        
        # Manual merge as a fallback
        try:
            vertices_list = []
            faces_list = []
            colors_list = []
            vertex_offset = 0
            
            for mesh in meshes:
                if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
                    continue
                
                vertices_list.append(mesh.vertices)
                faces_list.append(mesh.faces + vertex_offset)
                vertex_offset += len(mesh.vertices)
                
                # Handle colors if available
                if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                    colors_list.append(mesh.visual.vertex_colors)
                else:
                    # Default color (gray)
                    colors_list.append(np.full((len(mesh.vertices), 4), [128, 128, 128, 255], dtype=np.uint8))
            
            if not vertices_list:
                logger.warning("No valid meshes to merge, returning dummy mesh")
                return create_dummy_mesh()
            
            # Concatenate all data
            vertices = np.vstack(vertices_list)
            faces = np.vstack(faces_list) if faces_list else np.array([], dtype=np.int64).reshape(0, 3)
            colors = np.vstack(colors_list) if colors_list else None
            
            # Create the merged mesh
            merged_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            if colors is not None:
                merged_mesh.visual.vertex_colors = colors
            
            return merged_mesh
            
        except Exception as manual_err:
            logger.error(f"Manual mesh merging failed: {manual_err}")
            return create_dummy_mesh()

def simplify_mesh(mesh: trimesh.Trimesh, target_faces: int = 5000) -> trimesh.Trimesh:
    """
    Simplify a mesh to reduce its complexity.
    
    Args:
        mesh (trimesh.Trimesh): The mesh to simplify.
        target_faces (int): Target number of faces after simplification.
    
    Returns:
        trimesh.Trimesh: The simplified mesh.
    """
    if len(mesh.faces) <= target_faces:
        return mesh  # No simplification needed
    
    try:
        logger.info(f"Simplifying mesh from {len(mesh.faces)} to {target_faces} faces")
        simplified = mesh.simplify_quadratic_decimation(target_faces)
        return simplified
    except Exception as e:
        logger.warning(f"Mesh simplification failed: {e}")
        return mesh  # Return original mesh if simplification fails

def center_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Center a mesh at the origin.
    
    Args:
        mesh (trimesh.Trimesh): The mesh to center.
    
    Returns:
        trimesh.Trimesh: The centered mesh.
    """
    try:
        # Get the center of mass
        center = mesh.centroid
        
        # Create a translation matrix
        translation = np.eye(4)
        translation[:3, 3] = -center
        
        # Apply the translation
        centered_mesh = mesh.copy()
        centered_mesh.apply_transform(translation)
        
        return centered_mesh
    except Exception as e:
        logger.warning(f"Failed to center mesh: {e}")
        return mesh  # Return original mesh if centering fails