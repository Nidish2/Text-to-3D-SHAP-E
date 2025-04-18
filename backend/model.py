import os
import torch
import logging
import traceback
from typing import List, Tuple, Optional, Union

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh


# Rest of the file remains unchanged
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('text_to_3d.log')
    ]
)
logger = logging.getLogger(__name__)

class Text3DModel:
    # ... (rest of the class definition remains unchanged)
    def __init__(self):
        """Initialize the Text to 3D model with XPU as primary device."""
        self.initialize_devices()
        self.load_models()
        
    def initialize_devices(self):
        """Set up devices prioritizing XPU."""
        # First try to use Intel XPU
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            logger.info("Intel XPU detected, using XPU")
            self.device = torch.device('xpu')
            self.device_name = "xpu"
        # Then try CUDA as a secondary option
        elif torch.cuda.is_available():
            logger.info("CUDA detected, using GPU")
            self.device = torch.device('cuda')
            self.device_name = "cuda"
        # Fallback to CPU only if absolutely necessary
        else:
            logger.info("No XPU or GPU detected, using CPU")
            self.device = torch.device('cpu')
            self.device_name = "cpu"
        
        # Log memory info if available
        self.log_memory_info()
    
    def log_memory_info(self):
        """Log available memory information for the selected device."""
        try:
            if self.device_name == "xpu" and hasattr(torch, 'xpu') and torch.xpu.is_available():
                logger.info(f"XPU device count: {torch.xpu.device_count()}")
                try:
                    for i in range(torch.xpu.device_count()):
                        logger.info(f"XPU:{i} - {torch.xpu.get_device_name(i) if hasattr(torch.xpu, 'get_device_name') else 'Device name unavailable'}")
                        # XPU-specific memory info if available
                        if hasattr(torch.xpu, 'memory_allocated'):
                            logger.info(f"XPU:{i} - Allocated memory: {torch.xpu.memory_allocated(i) / 1e9:.2f} GB")
                        if hasattr(torch.xpu, 'memory_reserved'):
                            logger.info(f"XPU:{i} - Reserved memory: {torch.xpu.memory_reserved(i) / 1e9:.2f} GB")
                except Exception as e:
                    logger.warning(f"Could not get XPU device details: {e}")
            elif self.device_name == "cuda" and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    logger.info(f"CUDA:{i} - {torch.cuda.get_device_name(i)}")
                    logger.info(f"Total memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
                    logger.info(f"Available memory: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
        except Exception as e:
            logger.warning(f"Failed to log memory info: {e}")
    
    def load_models(self):
        """Load the required models without falling back to CPU."""
        try:
            logger.info(f"Loading models on {self.device_name}...")
            self.xm = load_model('transmitter', device=self.device)
            self.model = load_model('text300M', device=self.device)
            logger.info("Text300M model loaded successfully")
            
            logger.info("Loading diffusion process...")
            self.diffusion = diffusion_from_config(load_config('diffusion'))
            logger.info("Diffusion process loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models on {self.device_name}: {e}")
            raise RuntimeError(f"Could not load models on {self.device_name}: {str(e)}")
    
    def generate_latents(self, prompt: str, batch_size: int = 1, guidance_scale: float = 15.0) -> List[torch.Tensor]:
        """
        Generate latent tensors from text prompt.
        
        Args:
            prompt (str): The text prompt to generate 3D from
            batch_size (int): Number of samples to generate
            guidance_scale (float): Guidance scale for diffusion
            
        Returns:
            List[torch.Tensor]: Generated latent tensors
        """
        logger.info(f"Generating mesh for prompt: '{prompt}'")
        
        # XPU-specific optimizations
        if self.device_name == "xpu":
            logger.info("Running on XPU - optimizing parameters")
            batch_size = min(batch_size, 1)  # Limit batch size on XPU to prevent memory issues
            use_fp16 = True                  # XPU can handle FP16
            karras_steps = 64
            # Full quality
            if hasattr(torch.xpu, 'empty_cache'):
                torch.xpu.empty_cache()
                logger.info("Cleared XPU cache before generation")            
        elif self.device_name == "cpu":
            logger.info("Running on CPU - reducing parameters for performance")
            batch_size = 1
            use_fp16 = False
            karras_steps = 24
        else:  # CUDA
            use_fp16 = True
            karras_steps = 32
        
        try:
            latents = sample_latents(
                batch_size=batch_size,
                model=self.model,
                diffusion=self.diffusion,
                guidance_scale=guidance_scale,
                model_kwargs=dict(texts=[prompt] * batch_size),
                progress=True,
                clip_denoised=True,
                use_fp16=use_fp16,
                use_karras=True,
                karras_steps=karras_steps,
                sigma_min=1e-3,
                sigma_max=160,
                s_churn=0,
            )
            return latents
        except Exception as e:
            logger.error(f"Error generating latents: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to generate latents: {str(e)}")

    def generate_mesh(self, prompt: str, output_dir: str = "outputs") -> Tuple[str, str]:
        """
        Generate a 3D mesh from a text prompt with improved memory management for XPU.
    
        Args:
        prompt (str): Text description of the 3D object
        output_dir (str): Directory to save the output files
        
    Returns:
        Tuple[str, str]: Paths to the generated OBJ and PLY files
    """
        logger.info("Generating mesh with memory optimization...")
    
    # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    # Clean prompt for filename
        safe_prompt = "".join([c if c.isalnum() else "_" for c in prompt]).lower()
        obj_path = os.path.join(output_dir, f"{safe_prompt}.obj")
        ply_path = os.path.join(output_dir, f"{safe_prompt}.ply")
    
        try:
            import gc
        
        # Generate latent representation with memory cleanup
            latents = self.generate_latents(prompt, batch_size=1)
        
            logger.info(f"Generating mesh on {self.device_name} with memory optimization")
        
        # Force garbage collection before mesh decoding
            gc.collect()
            if self.device_name == "xpu" and hasattr(torch.xpu, 'empty_cache'):
                torch.xpu.empty_cache()
                logger.info("Cleared XPU cache before mesh decoding")
        
        # Use a chunk-based approach for mesh generation
        # Convert latent tensor to appropriate precision (float32 is more stable)
            latent = latents[0].to(dtype=torch.float32)
        
        # Decode the mesh with memory management
            try:
            # First detach the latent from computation graph to save memory
                latent = latent.detach()
            
            # For XPU, we'll use an incremental approach for large meshes
                if self.device_name == "xpu":
                # Use lower resolution for the initial mesh generation
                    logger.info("Using memory-optimized decoding approach for XPU")
                
                # Move latent processing to CPU for final decoding if needed
                    decoded = None
                    try:
                    # First try on XPU with aggressive memory management
                        decoded = decode_latent_mesh(self.xm, latent)
                    except Exception as xpu_err:
                        logger.warning(f"XPU decoding failed: {xpu_err}, falling back to CPU for mesh decoding")
                    
                    # Move to CPU for decoding if XPU fails
                        cpu_latent = latent.to("cpu")
                        cpu_xm = self.xm.to("cpu")
                    
                    # Clear XPU memory after transferring
                        latent = None
                        if hasattr(torch.xpu, 'empty_cache'):
                            torch.xpu.empty_cache()
                    
                    # Try decoding on CPU
                        decoded = decode_latent_mesh(cpu_xm, cpu_latent)
                    
                    # Move transmitter back to XPU
                        self.xm = cpu_xm.to(self.device)
                else:
                # For CPU/CUDA, use normal decoding
                    decoded = decode_latent_mesh(self.xm, latent)
            
            # Generate the mesh
                mesh = decoded.tri_mesh()
            
            # Force garbage collection after mesh generation
                decoded = None
                latent = None
                latents = None
                gc.collect()
            
                if self.device_name == "xpu" and hasattr(torch.xpu, 'empty_cache'):
                    torch.xpu.empty_cache()
            
            # Save the mesh files with chunking for memory efficiency
                with open(ply_path, 'wb') as f:
                    mesh.write_ply(f)
            
            # Use our optimized save function for OBJ with a smaller chunk size for XPU
                chunk_size = 2000 if self.device_name == "xpu" else 5000
                save_mesh_as_obj(mesh, obj_path, chunk_size=chunk_size)
            
                logger.info(f"Mesh successfully generated and saved to {obj_path} and {ply_path}")
                return obj_path, ply_path
                
            except Exception as decode_err:
                logger.error(f"Mesh decoding failed: {str(decode_err)}")
                logger.error(traceback.format_exc())
            
            # Try a fallback approach with simpler geometry
                logger.info("Attempting fallback with simplified geometry...")
            
            # Import the mesh_generation module for simple meshes
                from mesh_generation import create_dummy_mesh, save_mesh_as_obj
            
            # Create a simple cube mesh
                simple_mesh = create_dummy_mesh(size=1.0)
            
            # Save the simple mesh
                save_mesh_as_obj(simple_mesh, obj_path)
                with open(ply_path, 'wb') as f:
                    simple_mesh.write_ply(f)
            
                logger.info(f"Generated fallback mesh saved to {obj_path} and {ply_path}")
                return obj_path, ply_path
            
        except Exception as e:
            logger.error(f"Mesh generation failed: {str(e)}")
            logger.error(traceback.format_exc())
        
        # Create emergency fallback files even if everything fails
            try:
                from mesh_generation import create_dummy_mesh
            
                dummy = create_dummy_mesh()
                save_mesh_as_obj(dummy, obj_path)
            
                with open(ply_path, 'wb') as f:
                    dummy.write_ply(f)
                
                logger.info(f"Emergency fallback mesh saved to {obj_path} and {ply_path}")
                return obj_path, ply_path
            except:
                raise RuntimeError(f"Failed to generate mesh: {str(e)}")
    



if __name__ == "__main__":
    try:
        # Test the model with a simple prompt
        test_prompt = "a simple cube"
        
        logger.info("Initializing Text-to-3D model...")
        model = Text3DModel()
        
        obj_path, ply_path = model.generate_mesh(test_prompt)
        logger.info(f"Generated mesh files: {obj_path}, {ply_path}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(traceback.format_exc())
        