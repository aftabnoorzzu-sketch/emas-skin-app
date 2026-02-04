"""
Grad-CAM implementation for E-MAS model explainability.
Generates class activation heatmaps for visualization.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Generates visual explanations by highlighting important regions
    in the input image that contribute to the model's prediction.
    
    Args:
        model: PyTorch model
        target_layer: Layer to compute gradients from
        device: Device to run computations on
    """
    
    def __init__(self, model, target_layer, device='cpu'):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward pass activations."""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_image: Input tensor [1, C, H, W]
            target_class: Target class index (None for predicted class)
            
        Returns:
            cam: Grad-CAM heatmap [H, W] normalized to [0, 1]
        """
        self.model.eval()
        
        # Forward pass
        input_image = input_image.to(self.device)
        output = self.model(input_image)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0, target_class] = 1
        output.backward(gradient=one_hot_output, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients (weights)
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros_like(activations[0])
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input image size
        cam = cam.cpu().numpy()
        
        return cam
    
    def generate_cam_batch(self, input_images, target_classes=None):
        """
        Generate Grad-CAM heatmaps for a batch.
        
        Args:
            input_images: Input tensor [B, C, H, W]
            target_classes: Target class indices [B] (None for predicted classes)
            
        Returns:
            cams: List of Grad-CAM heatmaps
        """
        self.model.eval()
        
        batch_size = input_images.size(0)
        cams = []
        
        for i in range(batch_size):
            img = input_images[i:i+1]
            target = target_classes[i] if target_classes is not None else None
            cam = self.generate_cam(img, target)
            cams.append(cam)
        
        return cams


def visualize_gradcam(image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        image: Original image (PIL Image or numpy array [H, W, C])
        cam: Grad-CAM heatmap [H, W]
        alpha: Transparency of heatmap overlay
        colormap: OpenCV colormap to use
        
    Returns:
        overlay: Image with heatmap overlay [H, W, C]
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure image is in correct format
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Resize CAM to match image size
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Apply colormap
    cam_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    if image.shape[2] == 3:
        overlay = cv2.addWeighted(image, 1 - alpha, cam_colored, alpha, 0)
    else:
        # Grayscale image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(image_rgb, 1 - alpha, cam_colored, alpha, 0)
    
    return overlay


def generate_multi_branch_gradcam(model, input_image, device='cpu'):
    """
    Generate Grad-CAM for multiple branches of E-MAS model.
    
    Args:
        model: EMAS model instance
        input_image: Input tensor [1, C, H, W]
        device: Device to run on
        
    Returns:
        Dictionary containing heatmaps for each branch
    """
    results = {}
    
    # MobileNetV2 branch
    mobilenet_layer = model.mobilenet_reduce[0]
    gradcam_mobilenet = GradCAM(model, mobilenet_layer, device)
    cam_mobilenet = gradcam_mobilenet.generate_cam(input_image)
    results['mobilenet'] = cam_mobilenet
    
    # EfficientNet-B0 branch
    efficientnet_layer = model.efficientnet_reduce[0]
    gradcam_efficientnet = GradCAM(model, efficientnet_layer, device)
    cam_efficientnet = gradcam_efficientnet.generate_cam(input_image)
    results['efficientnet'] = cam_efficientnet
    
    # Fused features (ASPP output)
    fused_layer = model.aspp.conv_reduction[0]
    gradcam_fused = GradCAM(model, fused_layer, device)
    cam_fused = gradcam_fused.generate_cam(input_image)
    results['fused'] = cam_fused
    
    return results


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ implementation for better localization.
    Extends Grad-CAM with pixel-wise weighting.
    """
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generate Grad-CAM++ heatmap.
        
        Args:
            input_image: Input tensor [1, C, H, W]
            target_class: Target class index
            
        Returns:
            cam: Grad-CAM++ heatmap [H, W]
        """
        self.model.eval()
        
        # Forward pass
        input_image = input_image.to(self.device)
        output = self.model(input_image)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0, target_class] = 1
        output.backward(gradient=one_hot_output, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Grad-CAM++ weights
        grads_power_2 = gradients ** 2
        grads_power_3 = grads_power_2 * gradients
        
        # Sum in spatial dimension
        sum_activations = torch.sum(activations, dim=(1, 2), keepdim=True)
        
        # Compute alpha values
        eps = 1e-8
        aij = grads_power_2 / (2 * grads_power_2 + sum_activations * grads_power_3 + eps)
        
        # Weighted gradients
        weights = torch.sum(aij * F.relu(gradients), dim=(1, 2))
        
        # Weighted combination
        cam = torch.zeros_like(activations[0])
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()


if __name__ == "__main__":
    # Test Grad-CAM
    import sys
    sys.path.append('..')
    from models.emas import create_emas_model
    from utils.preprocess import preprocess_image
    
    print("Testing Grad-CAM...")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_emas_model(num_classes=7, pretrained=False, device=device)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Initialize Grad-CAM
    target_layer = model.mobilenet_reduce[0]
    gradcam = GradCAM(model, target_layer, device)
    
    # Generate CAM
    cam = gradcam.generate_cam(dummy_input)
    print(f"CAM shape: {cam.shape}")
    print(f"CAM min: {cam.min():.4f}, max: {cam.max():.4f}")
    
    print("Grad-CAM test completed successfully!")
