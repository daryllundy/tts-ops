import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def is_mps_available() -> bool:
    """
    Check if MPS (Metal Performance Shaders) backend is available.
    
    Returns:
        bool: True if MPS is available, False otherwise.
    """
    if not hasattr(torch.backends, "mps"):
        return False
    return torch.backends.mps.is_available()

def detect_best_device() -> str:
    """
    Detect the best available device in priority order: CUDA -> MPS -> CPU.
    
    Returns:
        str: The device string (e.g., "cuda:0", "mps", "cpu").
    """
    if torch.cuda.is_available():
        return "cuda:0"
    
    if is_mps_available():
        return "mps"
        
    return "cpu"

def resolve_device(device_str: str) -> str:
    """
    Resolve a device string to a concrete device.
    Handles "auto" by detecting the best available device.
    Validates explicit devices.
    
    Args:
        device_str: The device string to resolve (e.g., "auto", "cuda:0", "mps", "cpu").
        
    Returns:
        str: The resolved device string.
        
    Raises:
        ValueError: If the requested device is not available or invalid.
    """
    if device_str == "auto":
        resolved = detect_best_device()
        logger.info(f"Auto-detected device: {resolved}")
        return resolved
        
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(f"Device '{device_str}' requested but CUDA is not available.")
        return device_str
        
    if device_str == "mps":
        if not is_mps_available():
            raise ValueError("Device 'mps' requested but MPS is not available.")
        return "mps"
        
    if device_str == "cpu":
        return "cpu"
        
    # Fallback for other valid torch devices or if validation is loose, 
    # but for this spec we want strict validation.
    # However, let's allow other valid torch devices if they don't fall into above categories,
    # but maybe we should be strict as per requirements "Add comprehensive error messages for unavailable devices".
    # Let's try to create a device object to validate format, but check availability for known types.
    try:
        torch.device(device_str)
    except RuntimeError:
         raise ValueError(f"Invalid device string: '{device_str}'")

    return device_str
