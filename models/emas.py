"""
E-MAS: Efficient Multi-Scale Attention System
Implementation of the research paper model for dermoscopic skin lesion classification.

Architecture:
- MobileNetV2 + EfficientNet-B0 backbones (ImageNet pretrained)
- Point-wise feature fusion (element-wise multiplication)
- ASPP (Atrous Spatial Pyramid Pooling) with dilation rates [6, 12, 18]
- SE (Squeeze-and-Excitation) attention mechanism
- Global Average Pooling + Classifier head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module.
    Captures multi-scale contextual information using atrous convolutions
    with different dilation rates.
    """
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()
        
        # 1x1 convolution branch
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 convolution with dilation rate 6
        self.conv_3x3_r6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 convolution with dilation rate 12
        self.conv_3x3_r12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 convolution with dilation rate 18
        self.conv_3x3_r18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling branch
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_global = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 convolution to reduce concatenated features
        self.conv_reduction = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Forward pass through ASPP module.
        
        Args:
            x: Input feature tensor [B, C, H, W]
            
        Returns:
            Output feature tensor [B, out_channels, H, W]
        """
        # Get input spatial dimensions
        h, w = x.size(2), x.size(3)
        
        # Apply all branches
        conv_1x1 = self.conv_1x1(x)
        conv_r6 = self.conv_3x3_r6(x)
        conv_r12 = self.conv_3x3_r12(x)
        conv_r18 = self.conv_3x3_r18(x)
        
        # Global pooling branch
        global_feat = self.global_avg_pool(x)
        global_feat = self.conv_global(global_feat)
        global_feat = F.interpolate(global_feat, size=(h, w), 
                                    mode='bilinear', align_corners=False)
        
        # Concatenate all branches
        concat = torch.cat([conv_1x1, conv_r6, conv_r12, conv_r18, global_feat], dim=1)
        
        # Reduce channels
        output = self.conv_reduction(concat)
        
        return output


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation attention block.
    Adaptively recalibrates channel-wise feature responses.
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        
        # Squeeze: Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation: Two FC layers with reduction
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Forward pass through SE block.
        
        Args:
            x: Input feature tensor [B, C, H, W]
            
        Returns:
            Output feature tensor [B, C, H, W] with channel attention applied
        """
        b, c, _, _ = x.size()
        
        # Squeeze
        y = self.global_avg_pool(x).view(b, c)
        
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        
        # Scale
        return x * y.expand_as(x)


class EMAS(nn.Module):
    """
    E-MAS: Efficient Multi-Scale Attention System
    
    Ensemble model combining MobileNetV2 and EfficientNet-B0 with
    ASPP and SE attention for dermoscopic image classification.
    
    Args:
        num_classes: Number of output classes (7 for HAM10000, 3 for PH2)
        pretrained: Whether to use ImageNet pretrained weights
    """
    def __init__(self, num_classes=7, pretrained=True):
        super(EMAS, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained backbones
        # MobileNetV2 - efficient feature extraction
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        # Remove classifier, keep feature extractor
        self.mobilenet_features = mobilenet.features
        
        # EfficientNet-B0 - strong representational capability
        efficientnet = models.efficientnet_b0(pretrained=pretrained)
        # Extract features only
        self.efficientnet_features = efficientnet.features
        
        # Get output channels from backbones
        # MobileNetV2 last conv has 1280 channels
        # EfficientNet-B0 last conv has 1280 channels
        mobilenet_out_channels = 1280
        efficientnet_out_channels = 1280
        
        # 1x1 convolutions to match channel dimensions before fusion
        self.mobilenet_reduce = nn.Sequential(
            nn.Conv2d(mobilenet_out_channels, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.efficientnet_reduce = nn.Sequential(
            nn.Conv2d(efficientnet_out_channels, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # ASPP module for multi-scale feature extraction
        # Input: 512 channels (after reduction), Output: 256 channels
        self.aspp = ASPP(in_channels=512, out_channels=256)
        
        # SE attention block
        self.se_block = SEBlock(channels=256, reduction=16)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier head with two dense layers
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights for newly added layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def extract_features(self, x):
        """
        Extract fused features from both backbones.
        
        Args:
            x: Input tensor [B, 3, 224, 224]
            
        Returns:
            Fused feature tensor before ASPP [B, 512, H, W]
        """
        # Extract features from MobileNetV2
        mobilenet_feat = self.mobilenet_features(x)  # [B, 1280, 7, 7]
        mobilenet_feat = self.mobilenet_reduce(mobilenet_feat)  # [B, 512, 7, 7]
        
        # Extract features from EfficientNet-B0
        efficientnet_feat = self.efficientnet_features(x)  # [B, 1280, 7, 7]
        efficientnet_feat = self.efficientnet_reduce(efficientnet_feat)  # [B, 512, 7, 7]
        
        # Point-wise (element-wise) multiplication for feature fusion
        # F_fused = F_mobilenet ⊙ F_efficientnet
        fused_feat = mobilenet_feat * efficientnet_feat  # [B, 512, 7, 7]
        
        return fused_feat, mobilenet_feat, efficientnet_feat
    
    def forward(self, x, return_features=False):
        """
        Forward pass through E-MAS model.
        
        Args:
            x: Input tensor [B, 3, 224, 224]
            return_features: If True, return intermediate features for visualization
            
        Returns:
            logits: Output logits [B, num_classes]
            (optional) features: Dictionary of intermediate features
        """
        # Extract and fuse features from both backbones
        fused_feat, mobilenet_feat, efficientnet_feat = self.extract_features(x)
        
        # Apply ASPP for multi-scale context
        aspp_out = self.aspp(fused_feat)  # [B, 256, 7, 7]
        
        # Apply SE attention
        se_out = self.se_block(aspp_out)  # [B, 256, 7, 7]
        
        # Global Average Pooling
        gap_out = self.global_avg_pool(se_out)  # [B, 256, 1, 1]
        gap_out = gap_out.view(gap_out.size(0), -1)  # [B, 256]
        
        # Classifier
        logits = self.classifier(gap_out)  # [B, num_classes]
        
        if return_features:
            features = {
                'mobilenet': mobilenet_feat,
                'efficientnet': efficientnet_feat,
                'fused': fused_feat,
                'aspp': aspp_out,
                'se': se_out,
                'gap': gap_out
            }
            return logits, features
        
        return logits
    
    def get_gradcam_target_layer(self, backbone='mobilenet'):
        """
        Get the target layer for Grad-CAM visualization.
        
        Args:
            backbone: Which backbone to visualize ('mobilenet', 'efficientnet', or 'fused')
            
        Returns:
            Target layer for Grad-CAM
        """
        if backbone == 'mobilenet':
            # Return the last conv layer of MobileNetV2 feature reducer
            return self.mobilenet_reduce[0]
        elif backbone == 'efficientnet':
            # Return the last conv layer of EfficientNet-B0 feature reducer
            return self.efficientnet_reduce[0]
        elif backbone == 'fused':
            # Return ASPP output for fused features
            return self.aspp.conv_reduction[0]
        else:
            raise ValueError(f"Unknown backbone: {backbone}")


def create_emas_model(num_classes=7, pretrained=True, device='cpu'):
    """
    Factory function to create E-MAS model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        device: Device to load model on
        
    Returns:
        EMAS model instance
    """
    model = EMAS(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)
    return model


if __name__ == "__main__":
    # Test model creation and forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_emas_model(num_classes=7, pretrained=True, device=device)
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Forward pass without features
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    
    # Forward pass with features
    logits, features = model(x, return_features=True)
    print("\nIntermediate feature shapes:")
    for name, feat in features.items():
        if isinstance(feat, torch.Tensor):
            print(f"  {name}: {feat.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
