import torch
from torch import nn


class ResnetBlock2D(torch.nn.Module):
    def __init__(self, conv1_sizes=[32, 128, 128], conv2_sizes=[32, 128, 128]):
        super().__init__()
        self.norm1 = nn.GroupNorm(
            conv1_sizes[0], conv1_sizes[1], eps=1e-06, affine=True
        )
        self.conv1 = nn.Conv2d(
            conv1_sizes[1],
            conv1_sizes[2],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.norm2 = nn.GroupNorm(
            conv2_sizes[0], conv2_sizes[1], eps=1e-06, affine=True
        )
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.conv2 = nn.Conv2d(
            conv2_sizes[1],
            conv2_sizes[2],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.nonlinearity = nn.SiLU()

    def forward(self, x):
        x1 = self.norm1(x)
        x1 = self.nonlinearity(x1)
        x1 = self.conv1(x1)

        x2 = self.norm2(x)
        x2 = self.nonlinearity(x2)
        x2 = self.conv2(x)
        return x1 + x2


class Downsample2D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2))

    def forward(self, x):
        return self.conv(x)


class DownEncoderBlock2D(torch.nn.Module):
    def __init__(
        self,
        convs=[
            {"conv1_sizes": [32, 128, 128], "conv2_sizes": [32, 128, 128]},
            {"conv1_sizes": [32, 128, 128], "conv2_sizes": [32, 128, 128]},
        ],
    ):
        super().__init__()
        self.resnets = nn.ModuleList(
            [ResnetBlock2D(**conv_kwargs) for conv_kwargs in convs]
        )
        self.downsamplers = nn.ModuleList([Downsample2D()])

    def forward(self, x):
        x = self.resnets(x)
        return self.downsamplers(x)


class VAE(torch.nn.Module):
    pass


if __name__ == "__main__":
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
        StableDiffusionPipeline,
    )
    from _utils import copy_weights

    model = VAE()
    model.eval()

    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    model_hf = pipe.vae

    copy_weights(ResnetBlock2D(), model_hf.encoder.down_blocks[0].resnets[0])
    copy_weights(Downsample2D(), model_hf.encoder.down_blocks[0].downsamplers[0])
    copy_weights(DownEncoderBlock2D(), model_hf.encoder.down_blocks[0])
    # copy_weights(model, model_hf)
