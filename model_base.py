import torch
import torch.nn    as nn
import torchvision as tv

class BasicConvolution(nn.Module):
    def __init__(self, input_ch : int, output_ch : int, kernel_size : int = 3) -> None:
        super().__init__()

        self.convolution   = nn.Conv2d(input_ch, output_ch, kernel_size = kernel_size, padding = 'same')
        self.normalization = nn.BatchNorm2d(output_ch)
        self.activation    = nn.LeakyReLU()
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.convolution(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x

class BasicConvBlock(nn.Module):
    def __init__(self, input_ch : int, output_ch : int, kernel_size : int = 3) -> None:
        super().__init__()
        self.conv = BasicConvolution(input_ch, output_ch, kernel_size)
        # self.activation = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(kernel_size = 3, stride = 1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x
    
class UpBasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(UpBasicConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class UpSimpleCNN(nn.Module):
    def __init__(self, output_class: int) -> None:
        super(UpSimpleCNN, self).__init__()

        self.block_1 = nn.Conv2d(3, 64, kernel_size=3, padding='same')
        self.block_2 = UpBasicConvBlock(64, 128, 3)
        self.block_3 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.block_4 = UpBasicConvBlock(128, 64, 3)
        self.block_5 = nn.AdaptiveMaxPool2d(6)
        self.flatten = nn.Flatten()

        self.head = nn.Sequential(
            nn.Linear(2304, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, output_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.flatten(x)

        x = self.head(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, output_class : int) -> None:
        super().__init__()

        self.block_1 = nn.Conv2d(3, 64, kernel_size = 3, padding = 'same')
        self.block_2 = BasicConvBlock(64, 128, 3)
        self.block_3 = nn.Conv2d(128, 128, kernel_size = 3, padding = 'same')
        self.block_4 = BasicConvBlock(128, 64, 3)
        self.block_5 = nn.AdaptiveMaxPool2d(6)
        self.flatten = nn.Flatten()

        self.head = nn.Sequential(
            nn.Linear(2304, 2304),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(2304, output_class),
            nn.Softmax(1),
        )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:

        x = self.block_1(x) # (N,  3, 256, 256) => (N,  96,  85,  85)
        x = self.block_2(x) # (N,  96,  85,  85) => (N,  256,  28,  28)
        x = self.block_3(x)
        x = self.block_4(x) # (N,  384,  28,  28) => (N,  256,   9,   9)
        x = self.block_5(x)
        x = self.flatten(x) # (N, 256,   4,   4) => (N, 4096)    

        # print(x.size())

        # RESHAPE OPERATION
        bz = x.size(0)      # batch   size
        cz = x.size(1)      # channel size 
        x  = x.view(bz, cz) # (N, 128,   1,   1) => (N, 128)  ## 4D Tensor => 2D Tensor
        
        x = self.head(x)    # (N, 128)           => (N, output_class)
        return x

class BasicMobileNet(nn.Module):
    def __init__(self, output_classes : int) -> None:
        super().__init__()

        self.base = tv.models.mobilenet_v3_small(weights = tv.models.MobileNet_V3_Small_Weights.DEFAULT)
        self.base.classifier = nn.Sequential(
            nn.Linear(576, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_classes)
        )

        # self.base.classifier[3] = nn.Linear(1024, output_classes)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.base(x)
        return x
    
class SimpleNet(nn.Module):
    def __init__(self, output_classes : int) -> None:
        super().__init__()

        self.base = torch.hub.load("coderx7/simplenet_pytorch:v1.0.0", "simplenetv1_5m_m1", pretrained=True)
        self.base.fc = nn.Linear(512, output_classes)