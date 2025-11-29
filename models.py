import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
     
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ImprovedImageEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super(ImprovedImageEncoder, self).__init__()
     
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(32, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, output_dim)
        )
     
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
     
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
     
        attention = self.spatial_attention(x)
        x = x * attention
     
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
     
        x = self.fc(x)
     
        return x

class ImprovedMLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.3):
        super(ImprovedMLPEncoder, self).__init__()
     
        layers = []
        prev_dim = input_dim
     
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
     
        layers.append(nn.Linear(prev_dim, output_dim))
     
        self.network = nn.Sequential(*layers)
     
        self.skip_connection = nn.Linear(input_dim, output_dim)
     
    def forward(self, x):
        main_path = self.network(x)
        skip_path = self.skip_connection(x)
        return main_path + skip_path

class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super(CrossModalAttention, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1))
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
     
    def forward(self, x1, x2, x3):
        features = torch.stack([x1, x2, x3], dim=1) # [B, 3, D]
     
        Q = self.query(features)
        K = self.key(features)
        V = self.value(features)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.temperature * np.sqrt(Q.size(-1)))
        attention_weights = F.softmax(attention_scores, dim=-1)

        attended_features = torch.matmul(attention_weights, V)

        x1_att = attended_features[:, 0, :]
        x2_att = attended_features[:, 1, :]
        x3_att = attended_features[:, 2, :]
     
        return x1_att, x2_att, x3_att, attention_weights

class ImprovedMultiModalFusionNetwork(nn.Module):
    def __init__(self, structural_dim, material_dim, num_outputs):
        super(ImprovedMultiModalFusionNetwork, self).__init__()

        self.image_encoder = ImprovedImageEncoder(output_dim=256)

        self.structural_encoder = ImprovedMLPEncoder(
            input_dim=structural_dim,
            hidden_dims=[64, 128],
            output_dim=128,
            dropout_rate=0.3
        )

        self.material_encoder = ImprovedMLPEncoder(
            input_dim=material_dim,
            hidden_dims=[128, 128],
            output_dim=128,
            dropout_rate=0.3
        )

        self.align_dim = 256
        self.structural_align = nn.Linear(128, self.align_dim)
        self.material_align = nn.Linear(128, self.align_dim)
     
        self.cross_attention = CrossModalAttention(self.align_dim)

        self.prediction_head = nn.Sequential(
            nn.Linear(self.align_dim * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_outputs)
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.align_dim * 3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_outputs),
            nn.Softplus() # 
        )
     
    def forward(self, image, structural, material_props):

        image_features = self.image_encoder(image)
        structural_features = self.structural_encoder(structural)
        material_features = self.material_encoder(material_props)
  
        structural_features = self.structural_align(structural_features)
        material_features = self.material_align(material_features)

        img_att, struct_att, mat_att, attention_weights = self.cross_attention(
            image_features, structural_features, material_features
        )

        combined_features = torch.cat([img_att, struct_att, mat_att], dim=1)

        predictions = self.prediction_head(combined_features)
        uncertainty = self.uncertainty_head(combined_features)
     
        return predictions, uncertainty, attention_weights