import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False) 
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)
        self.relu = nn.ReLU(inplace=False) 
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return self.relu(out)

class CustomCNN(nn.Module):
    def __init__(self, num_classes=7):  
        super().__init__()
        
        # Feet: Initial feature extraction
        self.feet = nn.Sequential(
            ConvBlock(3, 32, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Body: Feature processing with reduced complexity
        self.body = nn.Sequential(
            # Stage 1
            ConvBlock(32, 64),
            ResidualBlock(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            # Stage 2
            ConvBlock(64, 128),
            ResidualBlock(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            # Stage 3
            ConvBlock(128, 256),
            ResidualBlock(256),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1)
        )
        
        # Head: Classification with simpler structure
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)  # Langsung ke output classes
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.feet(x)
        x = self.body(x)
        x = self.head(x)
        return x

if __name__ == "_main_":
    # Test model
    model = CustomCNN(num_classes=7)
    x = torch.randn(1, 3, 177, 177)
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")