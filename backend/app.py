import os
import uuid
import subprocess
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from .model import generate_mesh_from_text, get_shap_e_model
from .mesh_generation import save_mesh_as_obj
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Text-to-3D API", description="Generate 3D models from text descriptions")

# Add CORS middleware for frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Adjust this for your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories if they don't exist
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
output_dir = os.path.join(project_root, "generated_meshes")
os.makedirs(output_dir, exist_ok=True)

# Mount the generated_meshes directory to serve files directly
app.mount("/models", StaticFiles(directory=output_dir), name="models")

class TextInput(BaseModel):
    text: str
    format: str = "glb"
    device: str = None  # Optional device override

class GenerationResponse(BaseModel):
    message: str
    model_url: str
    format: str

@app.post("/generate", response_model=GenerationResponse)
async def generate_model(input: TextInput, background_tasks: BackgroundTasks):
    """
    Generate a 3D model from text and return it as a GLB or OBJ file.
    
    Args:
        input (TextInput): JSON body with 'text', 'format', and optional 'device'.
        background_tasks: FastAPI background tasks for cleanup.
    
    Returns:
        GenerationResponse: Response with URL to the generated model.
    """
    text = input.text
    format_type = input.format.lower()
    device = input.device
    
    if format_type not in ["glb", "obj"]:
        raise HTTPException(status_code=400, detail="Format must be 'glb' or 'obj'")
    
    try:
        unique_id = str(uuid.uuid4())
        base_name = os.path.join(output_dir, unique_id)
        obj_file = f"{base_name}.obj"
        glb_file = f"{base_name}.glb"
        
        # Initialize model with specified or default device
        model = get_shap_e_model(device=device)
        logger.info(f"Using device: {model.device}")
        
        # Generate and save mesh
        mesh = generate_mesh_from_text(text)
        save_mesh_as_obj(mesh, obj_file)
        
        output_file = obj_file
        if format_type == "glb":
            blender_script = os.path.join(current_dir, "blender_script.py")
            blender_cmd = ["blender", "-b", "-P", blender_script, "--", obj_file, glb_file]
            result = subprocess.run(blender_cmd, check=True, cwd=current_dir, capture_output=True, text=True)
            logger.info("Blender output: %s", result.stdout)
            output_file = glb_file
            background_tasks.add_task(lambda: os.remove(obj_file) if os.path.exists(obj_file) else None)
        
        if not os.path.exists(output_file):
            raise RuntimeError(f"Output file not found at {output_file}")
        
        model_url = f"/models/{os.path.basename(output_file)}"
        return GenerationResponse(
            message="Model generated successfully",
            model_url=model_url,
            format=format_type
        )
    except Exception as e:
        logger.error(f"Error during model generation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model generation failed: {str(e)}")

@app.post("/generate", response_model=GenerationResponse)
async def generate_model(input: TextInput, background_tasks: BackgroundTasks):
    """
    Generate a 3D model from text and return it as a GLB or OBJ file.
    
    Args:
        input (TextInput): JSON body with 'text' field and optional 'format' field.
        background_tasks: FastAPI background tasks for cleaning up temporary files.
    
    Returns:
        GenerationResponse: Response with URL to the generated model.
    """
    text = input.text
    format_type = input.format.lower()
    
    if format_type not in ["glb", "obj"]:
        raise HTTPException(status_code=400, detail="Format must be either 'glb' or 'obj'")
    
    try:
        # Generate unique filenames
        unique_id = str(uuid.uuid4())
        base_name = os.path.join(output_dir, unique_id)
        obj_file = f"{base_name}.obj"
        glb_file = f"{base_name}.glb"
        
        # Generate mesh
        logger.info("Generating mesh for prompt: '%s'", text)
        mesh = generate_mesh_from_text(text)
        logger.info("Mesh generation completed")
        
        # Save mesh as OBJ
        logger.info("Saving mesh as OBJ to %s", obj_file)
        save_mesh_as_obj(mesh, obj_file)
        logger.info("OBJ saved successfully")
        
        # If GLB format is requested, convert OBJ to GLB
        output_file = obj_file
        if format_type == "glb":
            logger.info("Converting to GLB...")
            blender_script = os.path.join(current_dir, "blender_script.py")
            blender_cmd = ["blender", "-b", "-P", blender_script, "--", obj_file, glb_file]
            result = subprocess.run(blender_cmd, check=True, cwd=current_dir, capture_output=True, text=True)
            logger.info("Blender output: %s", result.stdout)
            output_file = glb_file
            
            # Schedule cleanup of OBJ file after response is sent
            background_tasks.add_task(lambda: os.remove(obj_file) if os.path.exists(obj_file) else None)
        
        # Check if the file exists
        if not os.path.exists(output_file):
            raise RuntimeError(f"Output file not found at {output_file}")
        
        # Return the URL to the generated model
        model_url = f"/models/{os.path.basename(output_file)}"
        return GenerationResponse(
            message="Model generated successfully",
            model_url=model_url,
            format=format_type
        )
    
    except Exception as e:
        logger.error(f"Error during model generation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model generation failed: {str(e)}")

@app.get("/download/{filename}")
async def download_model(filename: str):
    """
    Download a generated model file.
    
    Args:
        filename: Filename of the model to download.
    
    Returns:
        FileResponse: The model file.
    """
    file_path = os.path.join(output_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine the media type based on the file extension
    media_type = "model/gltf-binary" if filename.endswith(".glb") else "model/obj"
    
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=filename
    )

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    
    Returns:
        dict: Status message.
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)