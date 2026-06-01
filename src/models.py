from __future__ import annotations

import torch
from torch import nn

from config import NUM_CLASSES


class UrbanDensityCNNV1(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.35),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class UrbanDensityCNNV2(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.35),
            nn.Linear(96, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.pool(features)
        return self.classifier(pooled)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)
        outputs = outputs + residual
        return self.relu(outputs)


class UrbanDensityCNNV3(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            ResidualBlock(32, 32, dropout=0.05),
            ResidualBlock(32, 64, stride=2, dropout=0.08),
            ResidualBlock(64, 64, dropout=0.08),
            ResidualBlock(64, 128, stride=2, dropout=0.10),
            ResidualBlock(128, 128, dropout=0.10),
            ResidualBlock(128, 192, stride=2, dropout=0.12),
            ResidualBlock(192, 192, dropout=0.12),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.40),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.stem(inputs)
        outputs = self.features(outputs)
        outputs = self.pool(outputs)
        return self.classifier(outputs)


class UrbanDensityCNNV4(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            ResidualBlock(32, 32, dropout=0.03),
            ResidualBlock(32, 64, stride=2, dropout=0.05),
            ResidualBlock(64, 64, dropout=0.05),
            ResidualBlock(64, 128, stride=2, dropout=0.08),
            ResidualBlock(128, 128, dropout=0.08),
            ResidualBlock(128, 192, stride=2, dropout=0.10),
            ResidualBlock(192, 192, dropout=0.10),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 160),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.30),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(160, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(161, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.35),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.stem(inputs)
        outputs = self.features(outputs)
        outputs = self.pool(outputs)
        embedding = self.embedding(outputs)
        built_ratio = self.regression_head(embedding)
        logits = self.classifier(torch.cat([embedding, built_ratio], dim=1))
        return {"logits": logits, "built_ratio": built_ratio.squeeze(1)}


def build_model(model_name: str) -> nn.Module:
    if model_name == "urban_cnn_v1":
        return UrbanDensityCNNV1()
    if model_name == "urban_cnn_v2":
        return UrbanDensityCNNV2()
    if model_name == "urban_cnn_v3":
        return UrbanDensityCNNV3()
    if model_name == "urban_cnn_v4":
        return UrbanDensityCNNV4()
    raise ValueError(f"Modelo desconhecido: {model_name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
