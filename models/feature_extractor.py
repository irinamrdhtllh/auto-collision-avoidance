import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class BaseResNetFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512) -> None:
        super().__init__(observation_space, features_dim)

        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # 1x1 conv layer to reduce the channel dimension to the resnet conv1 input channels
        self.conv = nn.Conv2d(
            in_channels=observation_space.shape[0],
            out_channels=resnet.conv1.in_channels,
            kernel_size=1,
            bias=False,
        )
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

        # Adjust linear output features dim
        self.linear = nn.Linear(resnet.fc.in_features, features_dim)

        # Freeze the ResNet layers
        for param in resnet.parameters():
            param.requires_grad = False
        # Remove the last layer
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.conv(x)
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)

        return x
    

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, observation_space, features_dim) -> None:
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # 1x1 conv layer to reduce the channel dimension to the resnet conv1 input channels
        self.conv = nn.Conv2d(
            in_channels=observation_space.shape[0],
            out_channels=resnet.conv1.in_channels,
            kernel_size=1,
            bias=False,
        )
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

        # Freeze the ResNet layers
        for param in resnet.parameters():
            param.requires_grad = False
        # Remove the last layer
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        # Adjust linear output features dim
        self.linear = nn.Linear(resnet.fc.in_features, features_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)

        return x
    

class MLPFeatureExtractor(nn.Module):
    def __init__(self, observation_space, features_dim) -> None:
        super().__init__()
        self.net = nn.Sequential(
            *create_mlp(observation_space.shape[0], features_dim, net_arch=[32, 16])
        )

    def forward(self, x):
        return self.net(x)


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim) -> None:
        super().__init__(observation_space, features_dim)

        self.extractors = nn.ModuleDict(
            {
                "image": ResNetFeatureExtractor(observation_space["image"], features_dim),
                "obj_distance": MLPFeatureExtractor(
                    observation_space["obj_distance"], features_dim
                ),
            }
        )

        # Update the features dim manually
        self._features_dim = 2 * features_dim

    def forward(self, x):
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(x[key]))
        return torch.cat(encoded_tensor_list, dim=1)