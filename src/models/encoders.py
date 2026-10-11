"""
Model encoders for each modality.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class RGBEncoder(nn.Module):
    """EfficientNet-B0 encoder for RGB images."""
    
    def __init__(
        self,
        pretrained: bool = True,
        frozen: bool = True,
        output_dim: int = 512,
        num_classes: int = 7
    ):
        super().__init__()
        
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Get feature dimension
        in_features = self.backbone.classifier[1].in_features
        
        # Remove classifier
        self.backbone.classifier = nn.Identity()
        
        # Freeze backbone if requested
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(in_features, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Classification head (for standalone training)
        self.classifier = nn.Linear(output_dim, num_classes)
        
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        features = self.backbone(x)
        projected = self.projection(features)
        
        if return_features:
            return projected, features
        
        logits = self.classifier(projected)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get projected features without classification head."""
        features = self.backbone(x)
        return self.projection(features)


class ThermalEncoder(nn.Module):
    """EfficientNet-B0 encoder adapted for thermal (single-channel) images."""
    
    def __init__(
        self,
        pretrained: bool = True,
        frozen: bool = True,
        output_dim: int = 512,
        in_channels: int = 1
    ):
        super().__init__()
        
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Adapt first layer for single channel
        if in_channels == 1:
            old_conv = self.backbone.features[0][0]
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            # Initialize by averaging RGB weights
            with torch.no_grad():
                new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
            self.backbone.features[0][0] = new_conv
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.projection = nn.Sequential(
            nn.Linear(in_features, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.projection(features)


class TabularEncoder(nn.Module):
    """MLP encoder for tabular symptom data."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [512, 256],
        output_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class FusionEncoder(nn.Module):
    """Multi-modal fusion encoder with attention and missing-modality masking."""
    
    def __init__(
        self,
        rgb_dim: int = 512,
        thermal_dim: int = 512,
        tabular_dim: int = 256,
        fusion_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Project all modalities to same dimension
        self.rgb_proj = nn.Linear(rgb_dim, fusion_dim)
        self.thermal_proj = nn.Linear(thermal_dim, fusion_dim)
        self.tabular_proj = nn.Linear(tabular_dim, fusion_dim)
        
        # Multi-head attention for fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.output_dim = fusion_dim
    
    def _get_modality_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Detect if modality is all zeros (missing). Returns boolean mask [B]."""
        # Sum over feature dim, if sum is near zero, modality is missing
        return x.abs().sum(dim=1) < 1e-6
    
    def forward(
        self,
        rgb_features: torch.Tensor,
        thermal_features: torch.Tensor,
        tabular_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            rgb_features: [B, rgb_dim]
            thermal_features: [B, thermal_dim]
            tabular_features: [B, tabular_dim]
        Returns:
            fused_features: [B, fusion_dim]
        """
        # Detect missing modalities
        rgb_missing = self._get_modality_mask(rgb_features)
        thermal_missing = self._get_modality_mask(thermal_features)
        tabular_missing = self._get_modality_mask(tabular_features)
        
        # Project to common dimension
        rgb_proj = self.rgb_proj(rgb_features).unsqueeze(1)      # [B, 1, fusion_dim]
        thermal_proj = self.thermal_proj(thermal_features).unsqueeze(1)  # [B, 1, fusion_dim]
        tabular_proj = self.tabular_proj(tabular_features).unsqueeze(1)  # [B, 1, fusion_dim]
        
        # Create attention mask for missing modalities
        # True = masked (don't attend), False = attend
        batch_size = rgb_features.size(0)
        attn_mask = torch.zeros(batch_size, 3, dtype=torch.bool, device=rgb_features.device)
        attn_mask[:, 0] = rgb_missing
        attn_mask[:, 1] = thermal_missing
        attn_mask[:, 2] = tabular_missing
        
        # Concatenate for attention
        modal_features = torch.cat([rgb_proj, thermal_proj, tabular_proj], dim=1)  # [B, 3, fusion_dim]
        
        # Self-attention across modalities with masking
        attended, _ = self.attention(
            modal_features, modal_features, modal_features,
            key_padding_mask=attn_mask
        )
        
        # Flatten and pass through fusion MLP
        fused = attended.flatten(1)  # [B, 3 * fusion_dim]
        fused = self.fusion_mlp(fused)
        
        return fused


class FusionModel(nn.Module):
    """Complete fusion model with three output heads."""
    
    def __init__(
        self,
        rgb_encoder: RGBEncoder,
        thermal_encoder: ThermalEncoder,
        tabular_encoder: TabularEncoder,
        fusion_encoder: FusionEncoder,
        num_visual_classes: int = 7,
        num_triage_classes: int = 3
    ):
        super().__init__()
        
        self.rgb_encoder = rgb_encoder
        self.thermal_encoder = thermal_encoder
        self.tabular_encoder = tabular_encoder
        self.fusion_encoder = fusion_encoder
        
        fusion_dim = fusion_encoder.output_dim
        
        # Three output heads
        self.fever_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, 1)  # Binary: fever flag
        )
        
        self.visual_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, num_visual_classes)
        )
        
        self.triage_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, num_triage_classes)
        )
    
    def forward(
        self,
        rgb_images: torch.Tensor,
        thermal_images: torch.Tensor,
        tabular_features: torch.Tensor
    ) -> dict:
        """
        Returns:
            dict with keys: 'fever', 'visual', 'triage' (logits)
        """
        # Encode each modality
        rgb_feat = self.rgb_encoder.get_features(rgb_images)
        thermal_feat = self.thermal_encoder(thermal_images)
        tabular_feat = self.tabular_encoder(tabular_features)
        
        # Fuse
        fused = self.fusion_encoder(rgb_feat, thermal_feat, tabular_feat)
        
        # Heads
        fever_logits = self.fever_head(fused).squeeze(-1)
        visual_logits = self.visual_head(fused)
        triage_logits = self.triage_head(fused)
        
        return {
            'fever': fever_logits,
            'visual': visual_logits,
            'triage': triage_logits,
            'fused_features': fused
        }
    
    def get_features(
        self,
        rgb_images: torch.Tensor,
        thermal_images: torch.Tensor,
        tabular_features: torch.Tensor
    ) -> torch.Tensor:
        """Get fused features without heads."""
        rgb_feat = self.rgb_encoder.get_features(rgb_images)
        thermal_feat = self.thermal_encoder(thermal_images)
        tabular_feat = self.tabular_encoder(tabular_features)
        return self.fusion_encoder(rgb_feat, thermal_feat, tabular_feat)