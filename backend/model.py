import torch
import numpy as np
import os
import logging
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh
import trimesh

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ShapEModel:
    """Class to handle Shap-E model for text-to-3D mesh generation."""
    
    def __init__(self, device=None):
        """
        Initialize the Shap-E model.

        Args:
            device (str, optional): Device to run the model on ('cuda', 'cpu', or None for auto-detection).
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.xpu.is_available():
                self.device = torch.device("xpu")  # Use with caution
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")
    
    # Load models
        logger.info("Loading Shap-E models...")
        self._load_models()

    def _load_models(self):
        """Load the text-to-3D mesh models for Shap-E."""
        try:
            # Load the transmitter model (for decoding latents to meshes)
            self.xm = load_model('transmitter', device=self.device)
            logger.info("Transmitter model loaded successfully")
            
            # Load the text-to-latent model
            self.model = load_model('text300M', device=self.device)
            logger.info("Text300M model loaded successfully")
            
            # Load the diffusion process
            self.diffusion = diffusion_from_config(load_config('diffusion'))
            logger.info("Diffusion process loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Shap-E models: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to load Shap-E models: {str(e)}")

    def generate_mesh(self, text_prompt):
        """
        Generate a 3D mesh from a text prompt using Shap-E.

        Args:
            text_prompt (str): Text description of the 3D object.

        Returns:
            trimesh.Trimesh: Generated 3D mesh.
        """
        try:
            logger.info(f"Generating mesh for prompt: '{text_prompt}'")
        
            # Generate latent representation from text
            batch_size = 1
            guidance_scale = 15.0  # High guidance for better adherence to prompt
            
            latents = sample_latents(
                batch_size=batch_size,
                model=self.model,
                diffusion=self.diffusion,
                guidance_scale=guidance_scale,
                model_kwargs=dict(texts=[text_prompt] * batch_size),
                progress=True,
                clip_denoised=True,
                use_fp16=True,
                use_karras=True,
                karras_steps=128,  # Increased for finer diffusion steps
                sigma_min=1e-3,
                sigma_max=160,
                s_churn=0,
            )
            
            # Move latent to CPU for decoding if GPU memory is an issue
            latents = latents.to('xpu')
            xm_cpu = self.xm.to('xpu')
            
            # Decode with higher grid size for detail (modify Shap-E if needed)
            mesh = decode_latent_mesh(xm_cpu, latents[0], grid_size=256).tri_mesh()
            logger.info("Mesh generation completed")
            return mesh
            
        except Exception as e:
            logger.error(f"Mesh generation failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to generate mesh: {str(e)}")
        
# Global instance of ShapEModel
_shap_e_model = None

def get_shap_e_model(device='xpu'):
    """
    Get or create a global instance of ShapEModel.

    Args:
        device (str): Device to run the model on ('xpu' or 'cpu').

    Returns:
        ShapEModel: A global instance of the model.
    """
    global _shap_e_model
    if _shap_e_model is None:
        _shap_e_model = ShapEModel(device=device)
    return _shap_e_model

def generate_mesh_from_text(text_prompt):
    """
    Generate a 3D mesh from text using the global ShapEModel.

    Args:
        text_prompt (str): Text description of the 3D object.

    Returns:
        trimesh.Trimesh: Generated mesh object.
    """
    model = get_shap_e_model()
    return model.generate_mesh(text_prompt)

if __name__ == "__main__":
    test_prompt = "a tiger walking in the jungle"
    try:
        model = ShapEModel()
        logger.info("Generating mesh...")
        mesh = model.generate_mesh(test_prompt)
        
        # Add this to check the type
        logger.info(f"Mesh type: {type(mesh)}")
        
        # Save mesh
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(os.path.dirname(current_dir), "generated_meshes")
        os.makedirs(output_dir, exist_ok=True)
        mesh_filename = f"{test_prompt.replace(' ', '_')}.obj"
        mesh_path = os.path.join(output_dir, mesh_filename)
        mesh.export(mesh_path)
        
        logger.info(f"Mesh saved to {mesh_path}")
        logger.info("Process completed successfully")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())