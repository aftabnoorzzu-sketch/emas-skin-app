"""
Image preprocessing utilities for E-MAS model.
Handles loading, normalization, and denormalization of dermoscopic images.
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image(image_path, target_size=None):
    """
    Load an image from file path.
    
    Args:
        image_path: Path to image file
        target_size: Tuple (width, height) to resize to (optional)
        
    Returns:
        PIL Image
    """
    try:
        image = Image.open(image_path).convert('RGB')
        
        if target_size:
            image = image.resize(target_size, Image.BILINEAR)
        
        return image
    except Exception as e:
        raise ValueError(f"Error loading image from {image_path}: {str(e)}")


def preprocess_image(image, input_size=224, normalize=True):
    """
    Preprocess image for model input.
    
    Args:
        image: PIL Image or numpy array
        input_size: Target input size (default 224 for E-MAS)
        normalize: Whether to apply ImageNet normalization
        
    Returns:
        Preprocessed tensor [1, 3, H, W]
    """
    # Convert numpy to PIL if needed
    if isinstance(image, np.ndarray):
        if image.dtype == np.float32 or image.dtype == np.float64:
            # Assume [0, 1] range
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
    
    # Build transform pipeline
    transform_list = [transforms.Resize((input_size, input_size))]
    
    if normalize:
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        transform_list.append(transforms.ToTensor())
    
    transform = transforms.Compose(transform_list)
    
    # Apply transforms
    tensor = transform(image)
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor


def preprocess_image_batch(images, input_size=224, normalize=True):
    """
    Preprocess a batch of images.
    
    Args:
        images: List of PIL Images or numpy arrays
        input_size: Target input size
        normalize: Whether to apply ImageNet normalization
        
    Returns:
        Batch tensor [B, 3, H, W]
    """
    processed = []
    for img in images:
        tensor = preprocess_image(img, input_size, normalize)
        processed.append(tensor.squeeze(0))
    
    return torch.stack(processed, dim=0)


def denormalize_image(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Denormalize a normalized image tensor for visualization.
    
    Args:
        tensor: Normalized tensor [C, H, W] or [B, C, H, W]
        mean: Mean used for normalization
        std: Std used for normalization
        
    Returns:
        Denormalized tensor in [0, 1] range
    """
    # Clone to avoid modifying original
    tensor = tensor.clone()
    
    # Handle batch dimension
    if tensor.dim() == 4:
        # [B, C, H, W]
        for i in range(tensor.size(0)):
            for c in range(3):
                tensor[i, c] = tensor[i, c] * std[c] + mean[c]
    else:
        # [C, H, W]
        for c in range(3):
            tensor[c] = tensor[c] * std[c] + mean[c]
    
    # Clamp to valid range
    tensor = torch.clamp(tensor, 0, 1)
    
    return tensor


def tensor_to_numpy(tensor, denormalize=False):
    """
    Convert tensor to numpy array for visualization.
    
    Args:
        tensor: PyTorch tensor [C, H, W] or [B, C, H, W]
        denormalize: Whether to denormalize first
        
    Returns:
        Numpy array [H, W, C] or [B, H, W, C] in [0, 255] range
    """
    if denormalize:
        tensor = denormalize_image(tensor)
    
    # Move to CPU and convert to numpy
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Handle batch dimension
    if tensor.dim() == 4:
        # [B, C, H, W] -> [B, H, W, C]
        array = tensor.permute(0, 2, 3, 1).numpy()
    else:
        # [C, H, W] -> [H, W, C]
        array = tensor.permute(1, 2, 0).numpy()
    
    # Convert to uint8
    array = (array * 255).astype(np.uint8)
    
    return array


def numpy_to_tensor(array, normalize=True, input_size=224):
    """
    Convert numpy array to tensor.
    
    Args:
        array: Numpy array [H, W, C] in [0, 255] range
        normalize: Whether to apply ImageNet normalization
        input_size: Target size
        
    Returns:
        Tensor [1, C, H, W]
    """
    # Convert to PIL
    if array.dtype != np.uint8:
        array = (array * 255).astype(np.uint8)
    
    image = Image.fromarray(array)
    
    return preprocess_image(image, input_size, normalize)


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Useful for enhancing dermoscopic images.
    
    Args:
        image: PIL Image or numpy array
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Enhanced image as PIL Image
    """
    import cv2
    
    # Convert to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(enhanced)


def remove_hair(image, kernel_size=15):
    """
    Remove hair artifacts from dermoscopic images using morphological operations.
    
    Args:
        image: PIL Image or numpy array
        kernel_size: Size of morphological kernel
        
    Returns:
        Hair-removed image as PIL Image
    """
    import cv2
    
    # Convert to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Black hat transform to detect dark hair
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Threshold to create hair mask
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Inpaint to remove hair
    inpainted = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return Image.fromarray(inpainted)


def center_crop(image, crop_size):
    """
    Center crop an image.
    
    Args:
        image: PIL Image
        crop_size: Tuple (width, height)
        
    Returns:
        Cropped PIL Image
    """
    width, height = image.size
    crop_width, crop_height = crop_size
    
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    
    return image.crop((left, top, right, bottom))


def get_preprocessing_pipeline(pipeline_type='standard', input_size=224):
    """
    Get a preprocessing pipeline by name.
    
    Args:
        pipeline_type: 'standard', 'augmented', or 'minimal'
        input_size: Target input size
        
    Returns:
        transforms.Compose pipeline
    """
    if pipeline_type == 'standard':
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    elif pipeline_type == 'augmented':
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    elif pipeline_type == 'minimal':
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor()
        ])
    
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")


if __name__ == "__main__":
    # Test preprocessing
    print("Testing preprocessing utilities...")
    
    # Create dummy image
    dummy_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    dummy_image = Image.fromarray(dummy_array)
    
    # Test preprocess
    tensor = preprocess_image(dummy_image, input_size=224)
    print(f"Preprocessed tensor shape: {tensor.shape}")
    print(f"Tensor range: [{tensor.min():.4f}, {tensor.max():.4f}]")
    
    # Test denormalize
    denorm = denormalize_image(tensor.squeeze(0))
    print(f"Denormalized range: [{denorm.min():.4f}, {denorm.max():.4f}]")
    
    # Test tensor to numpy
    numpy_img = tensor_to_numpy(tensor.squeeze(0), denormalize=True)
    print(f"Numpy image shape: {numpy_img.shape}")
    print(f"Numpy image dtype: {numpy_img.dtype}")
    
    print("Preprocessing tests completed successfully!")
