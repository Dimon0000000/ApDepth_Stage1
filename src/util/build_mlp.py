import torch
import torch.nn as nn

class ConvFeatureAdapter(nn.Module):
    """
    卷积适配器：保留空间结构的特征对齐模块。
    结构：Proj(1x1) -> GroupNorm -> SiLU -> Context(3x3) -> Proj(1x1)
    """
    def __init__(self, in_channels, out_channels, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = out_channels
        
        self.net = nn.Sequential(
            # 1. 初始投影：调整通道数
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            # 使用 GroupNorm 而不是 BatchNorm，因为 BatchSize 通常较小
            nn.GroupNorm(num_groups=32, num_channels=hidden_channels), 
            nn.SiLU(),
            
            # 2. 空间上下文：感知局部邻域信息
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=hidden_channels),
            nn.SiLU(),
            
            # 3. 输出投影
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # x shape: [B, C, H, W]
        return self.net(x)

def build_conv_adapter(in_channels=1280, out_channels=1536, hidden_channels=1280):
    return ConvFeatureAdapter(
        in_channels=in_channels, 
        out_channels=out_channels, 
        hidden_channels=hidden_channels
    )

def build_mlp_(hidden_size=640, projector_dim=1024, z_dim=768):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )