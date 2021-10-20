from einops.layers.torch import Rearrange
from torch import nn

from acon import MetaAconC
from apsa import APSA


class LayerNorm2d(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.img2emb = Rearrange("b c h w -> b h w c")
        self.norm = nn.LayerNorm(channel)
        self.emb2img = Rearrange("b h w c -> b c h w")

    def forward(self, x):
        x = self.img2emb(x)
        x = self.norm(x)
        x = self.emb2img(x)
        return x


class PreModule(nn.Module):
    def __init__(self, in_chans, out_chans, norm_layer):
        super().__init__()

        self.norm = norm_layer
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=7, stride=4, padding=3)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        return x


class ConvHead(nn.Module):
    def __init__(self, chans, norm_layer, act_layer=nn.GELU()):
        super().__init__()

        self.norm1 = norm_layer
        self.conv1 = nn.Conv2d(chans, chans, kernel_size=3, stride=1, padding=1)
        self.norm2 = norm_layer
        self.actv1 = act_layer
        self.conv2 = nn.Conv2d(chans, chans, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        feature = self.norm1(x)
        feature = self.conv1(feature)
        feature = self.norm2(feature)
        feature = self.actv1(feature)
        feature = self.conv2(feature)
        return feature + x


class ApsaBlock(nn.Module):
    def __init__(
        self,
        chans,
        img_size,
        window_size,
        patch_size,
        heads,
        pos_emb=True,
        first_half=False,
    ):
        super().__init__()

        self.attn = APSA(
            chans,
            img_size,
            window_size,
            patch_size,
            pooled_size=None,
            pooling_method="avg",
            heads=heads,
            attn_drop=None,
            pos_emb=pos_emb,
            orig_value=False,
            dropout=0.0,
            first_half=first_half,
        )
        self.head = ConvHead(
            chans, norm_layer=LayerNorm2d(chans), act_layer=MetaAconC(chans)
        )

    def forward(self, x):
        x = self.attn(x)
        x = self.head(x)
        return x


class DownSampler(nn.Module):
    def __init__(self, in_chans, out_chans, norm_layer):
        super().__init__()

        self.norm = norm_layer
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        return x


class StageModule(nn.Module):
    def __init__(
        self,
        in_chans,
        img_size,
        down_sample,
        layers,
        window_size,
        patch_size,
        heads,
        pos_emb,
        last_stage=False,
    ):
        super().__init__()

        inner_chans = in_chans * 2 if down_sample else in_chans
        self.down_sample = (
            DownSampler(
                in_chans=in_chans,
                out_chans=inner_chans,
                norm_layer=LayerNorm2d(in_chans),
            )
            if down_sample
            else nn.Identity()
        )

        self.layers = nn.ModuleList([])
        for _ in range(layers if not last_stage else layers * 2):
            self.layers.append(
                ApsaBlock(
                    inner_chans,
                    img_size=img_size // 2 if down_sample else img_size,
                    window_size=window_size,
                    patch_size=patch_size,
                    heads=heads,
                    pos_emb=pos_emb,
                    first_half=False if not last_stage else True,
                )
            )

    def forward(self, x):
        x = self.down_sample(x)
        for block in self.layers:
            x = block(x)
        return x


class ApsaSwin(nn.Module):
    def __init__(
        self,
        layers,
        window_size,
        heads,
        dim=96,
        img_size=224,
        channels=3,
        num_classes=None,
    ):
        super().__init__()

        self.pre_stage = PreModule(
            in_chans=channels, out_chans=dim, norm_layer=LayerNorm2d(channels),
        )

        self.stage1 = StageModule(
            dim,
            img_size // 4,
            down_sample=False,
            layers=layers[0],
            window_size=window_size[0],
            heads=heads[0],
            patch_size=1,
            pos_emb=True,
            last_stage=False,
        )
        self.stage2 = StageModule(
            dim,
            img_size // 4,
            down_sample=True,
            layers=layers[1],
            window_size=window_size[1],
            heads=heads[1],
            patch_size=1,
            pos_emb=True,
            last_stage=False,
        )
        self.stage3 = StageModule(
            dim * 2,
            img_size // 8,
            down_sample=True,
            layers=layers[2],
            window_size=window_size[2],
            heads=heads[2],
            patch_size=1,
            pos_emb=True,
            last_stage=False,
        )
        self.stage4 = StageModule(
            dim * 4,
            img_size // 16,
            down_sample=True,
            layers=layers[3],
            window_size=window_size[3],
            heads=heads[3],
            patch_size=1,
            pos_emb=True,
            last_stage=True,
        )

        self.mlp_head = (
            nn.Sequential(nn.LayerNorm(dim * 8), nn.Linear(dim * 8, num_classes))
            if num_classes is not None and num_classes is not 0  # bug in Pylance
            else nn.Identity()
        )

    def forward(self, img):
        x = self.pre_stage(img)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.mean(dim=[2, 3])
        return self.mlp_head(x)


def APSA_SWIN_B(
    hidden_dim=96,
    layers=(1, 2, 16, 1),
    heads=(4, 8, 16, 32),
    window_size=[7, 7, 7, 7],
    img_size=224,
    channels=3,
    num_classes=1000,
):
    return ApsaSwin(
        dim=hidden_dim,
        layers=layers,
        heads=heads,
        window_size=window_size,
        img_size=img_size,
        channels=channels,
        num_classes=num_classes,
    )
