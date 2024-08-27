import os
import torch
import lightning as L
from sam2.build_sam import build_sam2

def find_checkpoint_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    checkpoint_path = os.path.join(project_root, 'segment-anything-2', 'checkpoints', 'sam2_hiera_large.pt')
    
    if not os.path.exists(checkpoint_path):
        # Try alternative path
        checkpoint_path = os.path.join(project_root, 'checkpoints', 'sam2_hiera_large.pt')
    
    return checkpoint_path

def confirm_setup():
    print("Checking dependencies...")
    assert torch.__version__ >= "2.3.1", f"PyTorch version should be >=2.3.1, but is {torch.__version__}"
    assert L.__version__ >= "2.0.0", f"Lightning version should be >=2.0.0, but is {L.__version__}"
    
    print("Checking SAM2 checkpoint...")
    checkpoint_path = find_checkpoint_path()
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    
    print("Building SAM2 model...")
    model = build_sam2("sam2_hiera_l.yaml", checkpoint_path)
    print("SAM2 model built successfully!")
    
    print("All checks passed! You're ready for the tutorials!")

if __name__ == "__main__":
    confirm_setup()