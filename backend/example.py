import os
import argparse
import logging
from model import Text3DModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('text_to_3d_example.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Text-to-3D Model Generator")
    parser.add_argument("--prompt", type=str, default="a simple cube", 
                        help="Text description of the 3D object to generate")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save the generated meshes")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of samples to generate (only works on CUDA)")
    args = parser.parse_args()
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Initialize the model
        logger.info(f"Initializing Text-to-3D model...")
        model = Text3DModel()
        
        # Generate mesh from prompt
        logger.info(f"Generating 3D mesh for prompt: '{args.prompt}'")
        obj_path, ply_path = model.generate_mesh(args.prompt, args.output_dir)
        
        logger.info(f"Successfully generated 3D mesh!")
        logger.info(f"OBJ file saved to: {obj_path}")
        logger.info(f"PLY file saved to: {ply_path}")
        
        # Print usage instructions
        print("\n" + "="*50)
        print("GENERATION SUCCESSFUL!")
        print("="*50)
        print(f"Generated from prompt: '{args.prompt}'")
        print(f"Files saved to:")
        print(f"  - OBJ: {obj_path}")
        print(f"  - PLY: {ply_path}")
        print("\nYou can view these files using any 3D model viewer.")
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Error generating 3D model: {e}")
        print(f"\nERROR: Failed to generate 3D model: {e}")
        print("Check the log file for more details.")

if __name__ == "__main__":
    main()