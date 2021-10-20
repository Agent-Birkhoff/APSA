from torch import nn

from acon import MetaAconC
from apsa import APSA


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_chans,
        img_size,
        window_size,
        patch_size,
        downsample,
        pooling="avg",
        heads=4,
        attn_drop=None,
        dropout=0.5,
        orig_value=False,
        pos_emb=True,
        half=False,
    ):
        super().__init__()

        img_size = img_size // 2 if downsample else img_size

        assert img_size % window_size == 0, "img_size must be divisible by window_size!"
        assert (
            window_size % patch_size == 0
        ), "window_size must be divisible by patch_size!"
        assert pooling in {"avg", "max"}, "pooling must be either avg or max!"

        out_chans = in_chans * 2 if downsample else in_chans
        self.conv_layer1 = nn.Conv2d(
            in_chans,
            out_chans,
            kernel_size=3,
            stride=2 if downsample else 1,
            padding=1,
            bias=False,
        )
        self.norm_layer1 = nn.BatchNorm2d(out_chans)
        self.actv_layer1 = MetaAconC(out_chans)
        self.conv_layer2 = nn.Conv2d(
            out_chans, out_chans, kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.attn_layer = nn.ModuleList([])
        for i in range(2 if half else 1):
            self.attn_layer.append(
                APSA(
                    channel=out_chans,
                    size=img_size,
                    window_size=window_size,
                    patch_size=patch_size,
                    pooling_method=pooling,
                    heads=heads,
                    attn_drop=attn_drop,
                    dropout=dropout,
                    orig_value=orig_value,
                    pos_emb=pos_emb,
                    first_half=half,
                )
            )
        self.norm_layer2 = nn.BatchNorm2d(out_chans)
        self.shortcut = (
            nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=2, bias=False)
            if downsample
            else nn.Identity()
        )
        self.actv_layer2 = MetaAconC(out_chans)

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.conv_layer1(x)
        x = self.norm_layer1(x)
        x = self.actv_layer1(x)

        x = self.conv_layer2(x)
        for block in self.attn_layer:
            x = block(x)
        x = self.norm_layer2(x)

        x += identity
        x = self.actv_layer2(x)

        return x


class ApsaNet(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        in_chans=3,
        start_chans=4,
        img_size=224,
        window_size=[28, 28, 14, 14],
        patch_size=[7, 7, 2, 2],
        attn_drop=None,
        orig_value=False,
        pos_emb=True,
    ):
        super().__init__()

        assert start_chans >= in_chans, "start_chans must be >= in_chans!"

        channel = start_chans
        self.conv0 = nn.Conv2d(in_chans, channel, kernel_size=1, stride=1, bias=False)
        self.norm0 = nn.BatchNorm2d(channel)
        self.actv0 = MetaAconC(channel)
        self.layer1 = BasicBlock(
            in_chans=channel,
            img_size=img_size,
            window_size=window_size[0],
            patch_size=patch_size[0],
            downsample=False,
            attn_drop=attn_drop,
            orig_value=orig_value,
            pos_emb=pos_emb,
        )
        self.layer2 = BasicBlock(
            in_chans=channel,
            img_size=img_size,
            window_size=window_size[1],
            patch_size=patch_size[1],
            downsample=True,
            attn_drop=attn_drop,
            orig_value=orig_value,
            pos_emb=pos_emb,
        )
        self.layer3 = BasicBlock(
            in_chans=channel * 2,
            img_size=img_size // 2,
            window_size=window_size[2],
            patch_size=patch_size[2],
            downsample=True,
            attn_drop=attn_drop,
            orig_value=orig_value,
            pos_emb=pos_emb,
        )
        self.layer4 = BasicBlock(
            in_chans=channel * 4,
            img_size=img_size // 4,
            window_size=window_size[3],
            patch_size=patch_size[3],
            downsample=True,
            attn_drop=attn_drop,
            orig_value=orig_value,
            pos_emb=pos_emb,
        )
        self.conv1 = nn.Conv2d(
            channel * 8, channel * 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.norm1 = nn.BatchNorm2d(channel * 32)
        self.actv1 = MetaAconC(channel * 32)
        self.conv2 = nn.Conv2d(
            channel * 32, channel * 128, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.norm2 = nn.BatchNorm2d(channel * 128)
        self.actv2 = MetaAconC(channel * 128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(channel * 128, num_classes)

    def forward(self, x):
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.actv0(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.actv1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.actv2(x)

        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
