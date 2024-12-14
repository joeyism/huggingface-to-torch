from torch import nn


class ResNetConvLayer(nn.Module):

    def __init__(
        self,
        input_size: int = 3,
        output_size: int = 64,
        kernel_size: tuple[int, int] = (7, 7),
        stride: tuple[int, int] = (2, 2),
        padding: tuple[int, int] = (3, 3),
        activation = None
    ):
        super().__init__()
        self.convolution = nn.Conv2d(
            input_size,
            output_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )  # [N + 2P - K]/S + 1
        self.normalization = nn.BatchNorm2d(
            output_size,
        )
        self.activation = activation or nn.ReLU()

    def forward(self, x):  # (3, 224, 224)
        x = self.convolution(x)  # (64, 112, 112) floor(224 + 2*3 - 7)/2 + 1
        x = self.normalization(x)  # (64, 112, 112)
        x = self.activation(x)  # (64, 112, 112)
        return x


class ResNetEmbeddings(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedder = ResNetConvLayer()
        self.pooler = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

    def forward(self, x):  # (3, 224, 224)
        x = self.embedder(x)  # (64, 112, 112)
        x = self.pooler(x)  # (64, 56, 56)
        return x


class ResNetShortCut(nn.Module):

    def __init__(
        self,
        input_size: int = 64,
        output_size: int = 256,
        kernel_size: tuple[int, int] = (1, 1),
        stride: tuple[int, int] = (1, 1),
        eps: float = 1e-05,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.convolution = nn.Conv2d(
            input_size, output_size, kernel_size=kernel_size, stride=stride, bias=False
        )
        self.normalization = nn.BatchNorm2d(
            output_size,
            eps=eps,
            momentum=momentum,
            affine=True,
            track_running_stats=True,
        )

    def forward(self, x):  # (64, 56, 56)
        x = self.convolution(x)
        x = self.normalization(x)
        return x


class ResNetBottleNeckLayer(nn.Module):

    def __init__(
        self,
        input_size: int = 64,
        output_size: int = 256,
        hidden_size: int = 64,
        has_shortcut: bool = True,
    ):
        super().__init__()
        self.shortcut = (
            ResNetShortCut(
                input_size=input_size,
                output_size=output_size,
                kernel_size=(1, 1),
                stride=(1, 1),
            )
            if has_shortcut
            else nn.Identity()
        )
        self.layer = nn.Sequential(
            ResNetConvLayer(
                input_size=input_size,
                output_size=hidden_size,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            ),
            ResNetConvLayer(
                input_size=hidden_size,
                output_size=hidden_size,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            ResNetConvLayer(
                input_size=hidden_size,
                output_size=output_size,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                activation=nn.Identity() # prevent vanishing gradient
            ),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.shortcut(x) + self.layer(x)
        x = self.activation(x)
        return x


class ResNetStage(nn.Module):

    def __init__(
        self,
        input_size: int = 64,
        output_size: int = 256,
        hidden_size: int = 64,
        length: int = 3,
    ):
        super().__init__()
        seq = [
            ResNetBottleNeckLayer(
                input_size=input_size,
                output_size=output_size,
                hidden_size=hidden_size,
                has_shortcut=True,
            )
        ]
        for _ in range(length - 1):
            seq.append(
                ResNetBottleNeckLayer(
                    input_size=output_size,
                    output_size=output_size,
                    hidden_size=hidden_size,
                    has_shortcut=False,
                )
            )

        self.layers = nn.Sequential(*seq)

    def forward(self, x):
        return self.layers(x)


class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                ResNetStage(input_size=64, output_size=256, hidden_size=64, length=3),
                ResNetStage(input_size=256, output_size=512, hidden_size=128, length=4),
                ResNetStage(
                    input_size=512, output_size=1024, hidden_size=256, length=6
                ),
                ResNetStage(
                    input_size=1024, output_size=2048, hidden_size=512, length=3
                ),
            ]
        )

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x


class ResNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = ResNetEmbeddings()
        self.encoder = ResNetEncoder()
        self.pooler = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.embedder(x)
        x = self.encoder(x)
        x = self.pooler(x)
        return x


class ResNetForImageClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNetModel()
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1), nn.Linear(2048, 1000, bias=True)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    import torch
    from datasets import load_dataset
    from torchvision.transforms import functional as f
    from transformers import AutoImageProcessor
    from transformers import \
        ResNetForImageClassification as ResNetForImageClassificationHF

    from _utils import copy_weights

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    image_tensor = f.pil_to_tensor(image)
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    inputs = processor(image, return_tensors="pt")
    model = ResNetForImageClassification()
    model.eval()
    model_hf = ResNetForImageClassificationHF.from_pretrained("microsoft/resnet-50")
    before_copy = torch.clone(
        model.resnet.encoder.stages[3].layers[2].layer[0].convolution.weight
    )
    copy_weights(model, model_hf)
    after_copy = torch.clone(
        model.resnet.encoder.stages[3].layers[2].layer[0].convolution.weight
    )
    assert not (before_copy == after_copy).any()
    logits = model(inputs["pixel_values"])
    predicted_label = logits.argmax(-1).item()
    print(model_hf.config.id2label[predicted_label])
