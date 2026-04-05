from typing import Tuple, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        groups: int = 8,
        norm_type: Literal["group", "instance", "identity"] = "group",
    ):
        super().__init__()

        def make_norm(ch):
            if norm_type == "group":
                g = min(groups, ch)
                return nn.GroupNorm(g, ch)
            elif norm_type == "instance":
                return nn.InstanceNorm3d(ch, affine=True)
            elif norm_type == "identity":
                return nn.Identity()
            else:
                raise ValueError(f"Unknown norm_type: {norm_type}")

        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = make_norm(out_ch)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = make_norm(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        return x


class PatchToVoxelDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_steps: int = 3,
        target_shape: Tuple[int, int, int] = (112, 112, 112),
        final_norm: Literal["group", "instance", "identity"] = "identity",
    ):
        super().__init__()

        self.target_shape = target_shape
        self.num_steps = num_steps
        self.num_classes = num_classes

        self.proj = nn.Conv3d(embed_dim, hidden_dim, kernel_size=1)

        self.layers = nn.ModuleList()
        in_ch = hidden_dim

        for i in range(num_steps):
            out_ch = max(in_ch // 2, 64)

            norm_type = final_norm if i == num_steps - 1 else "group"

            self.layers.append(
                ConvBlock3D(
                    in_ch,
                    out_ch,
                    norm_type=norm_type,
                )
            )
            in_ch = out_ch

        self.classifier = nn.Conv3d(in_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Nz, Ny, Nx, D)
        """
        B, Nz, Ny, Nx, D = x.shape
        target_Z, target_Y, target_X = self.target_shape

        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.proj(x)

        start_Z, start_Y, start_X = Nz, Ny, Nx

        for i, layer in enumerate(self.layers):
            x = layer(x)

            alpha = (i + 1) / self.num_steps
            curr_Z = round(start_Z + alpha * (target_Z - start_Z))
            curr_Y = round(start_Y + alpha * (target_Y - start_Y))
            curr_X = round(start_X + alpha * (target_X - start_X))

            x = F.interpolate(
                x,
                size=(curr_Z, curr_Y, curr_X),
                mode="trilinear",
                align_corners=False,
            )

        return self.classifier(x)
