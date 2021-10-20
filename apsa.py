import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn


class AttenWeight(nn.Module):
    def __init__(self, dim, heads=8, dim_per_head=64):
        super().__init__()

        self.heads = heads
        self.scale = dim_per_head ** -0.5

        inner_dim = dim_per_head * heads
        self.to_qk = nn.Linear(dim, inner_dim * 2, bias=False)
        self.attn = nn.Softmax(dim=-1)

    def forward(self, x):
        qk = self.to_qk(x).chunk(2, dim=-1)
        query, key = map(
            lambda t: rearrange(t, "... n (h d) -> ... h n d", h=self.heads), qk
        )
        weight = einsum("... h q d, ... h k d -> ... h q k", query, key) * self.scale
        return self.attn(weight)


class LePE(nn.Module):
    def __init__(self, dim, resolution, heads, window_num=None, window=True):
        super().__init__()

        if not window:
            assert window_num is not None, "window_num can not be None!"

        self.emb2img = (
            Rearrange("b hd (h w) c -> (b hd) c h w", h=resolution)
            if window
            else Rearrange("b n1 hd (h w) c -> (b n1 hd) c h w", h=resolution)
        )
        self.get_lepe = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim
        )
        self.img2emb = (
            Rearrange("(b hd) c h w -> b hd (h w) c", hd=heads)
            if window
            else Rearrange(
                "(b n1 hd) c h w -> b n1 hd (h w) c", hd=heads, n1=window_num
            )
        )

    def forward(self, x):
        lepe = self.emb2img(x)
        lepe = self.get_lepe(lepe)
        lepe = self.img2emb(lepe)
        return lepe


class APSA(nn.Module):
    def __init__(
        self,
        channel,
        size,
        window_size,
        patch_size,
        pooled_size=None,
        pooling_method="avg",
        heads=8,
        window_dim_per_head=None,
        patch_dim_per_head=None,
        attn_drop=None,
        dropout=0.5,
        orig_value=True,
        pos_emb=False,
        first_half=False,
    ):
        super().__init__()

        assert size % window_size == 0, "size must be divisible by window_size!"
        assert (
            window_size % patch_size == 0
        ), "window_size must be divisible by patch_size!"
        assert pooling_method in {
            "avg",
            "max",
        }, "pooling_method must be either avg or max!"

        self.heads = heads
        self.pos_emb = pos_emb
        self.first_half = first_half

        patch_per_window_1d = window_size // patch_size
        self.patch_flatten = Rearrange(
            "b c (h w1 p1) (w w2 p2) -> b (h w) (w1 w2) (p1 p2 c)",
            w1=patch_per_window_1d,
            w2=patch_per_window_1d,
            p1=patch_size,
            p2=patch_size,
        )

        patch_dim = (patch_size ** 2) * channel
        self.patch_norm = nn.LayerNorm(patch_dim)
        patch_dim_per_head = (
            patch_dim // heads if not patch_dim_per_head else patch_dim_per_head
        )
        self.patch_weight = AttenWeight(patch_dim, heads, patch_dim_per_head)
        self.attn_drop = (
            nn.Dropout(attn_drop) if attn_drop is not None else nn.Identity()
        )
        patch_inner_dim = patch_dim_per_head * heads
        self.patch_value = (
            nn.Linear(patch_dim, patch_inner_dim, bias=False)
            if not orig_value
            else nn.Identity()
        )
        window_num_1d = size // window_size
        self.patch_pos = LePE(
            patch_dim_per_head,
            patch_per_window_1d,
            heads=heads,
            window_num=window_num_1d ** 2,
            window=False,
        )
        self.patch_out = (
            nn.Sequential(
                nn.Linear(patch_inner_dim, patch_dim, bias=False), nn.Dropout(dropout)
            )
            if not orig_value
            else nn.Identity()
        )

        if not first_half:
            pooled_size = patch_size * window_num_1d if not pooled_size else pooled_size
            self.pooling = (
                nn.AdaptiveAvgPool2d((pooled_size, pooled_size))
                if pooling_method == "avg"
                else nn.AdaptiveMaxPool2d(
                    (pooled_size, pooled_size)
                )  # if pooling_method=="max"
            )

            self.window_flatten = Rearrange(
                "b c (h w1) (w w2) -> b (h w) (w1 w2 c)",
                h=window_num_1d,
                w=window_num_1d,
            )

            pooled_dim = ((pooled_size // window_num_1d) ** 2) * channel
            self.pooled_norm = nn.LayerNorm(pooled_dim)
            window_dim_per_head = (
                pooled_dim // heads if not window_dim_per_head else window_dim_per_head
            )
            window_dim = (window_size ** 2) * channel
            self.window_norm = nn.LayerNorm(window_dim)
            assert (
                pooled_size % window_num_1d == 0
            ), "pooled_size must be divisible by sqrt(window_num)!"
            self.window_weight = AttenWeight(pooled_dim, heads, window_dim_per_head)
            window_inner_dim = window_dim_per_head * heads
            self.window_value = (
                nn.Linear(window_dim, window_inner_dim, bias=False)
                if not orig_value
                else nn.Identity()
            )
            self.window_pos = LePE(
                window_dim_per_head, window_num_1d, heads=heads, window=True,
            )
            self.window_out = (
                nn.Sequential(
                    nn.Linear(window_inner_dim, window_dim, bias=False),
                    nn.Dropout(dropout),
                )
                if not orig_value
                else nn.Identity()
            )

        self.to_out = Rearrange(
            "b (h w) (w1 w2 c) -> b c (h w1) (w w2)",
            h=window_num_1d,
            w1=window_size,
            c=channel,
        )

    def forward(self, x):
        # actually n2==q==k==v n1==q2==k2==v2
        flattened = self.patch_flatten(x)  # b c h w -> b n1 n2 d
        flattened = self.patch_norm(flattened)
        patch_w = self.patch_weight(flattened)  # b n1 n2 d -> b n1 h q k
        patch_w = self.attn_drop(patch_w)
        patch_v = self.patch_value(flattened)  # b n1 n2 d
        patch_v = rearrange(
            patch_v, "... n (h d) -> ... h n d", h=self.heads
        )  # b n1 n2 d -> b n1 h v d2
        patch_feature = einsum(
            "... h n v, ... h v d -> ... h n d", patch_w, patch_v
        )  # b n1 h n2 d2
        if self.pos_emb:
            patch_feature += self.patch_pos(patch_v)
        patch_feature = rearrange(
            patch_feature, "... h n d -> ... n (h d)"
        )  # b n1 h n2 d2 -> b n1 n2 d
        patch_feature = self.patch_out(patch_feature)
        patch_feature = rearrange(
            patch_feature, "... n d -> ... (n d)"
        )  # b n1 n2 d -> b n1 d3

        if not self.first_half:
            flattened = self.window_flatten(self.pooling(x))  # b c h2 w2 -> b n1 d5
            flattened = self.pooled_norm(flattened)
            window_w = self.window_weight(flattened)  # b n1 d5 -> b h q2 k2
            # window_w=self.attn_drop(window_w)  # optional
            diag = window_w.diagonal(dim1=-2, dim2=-1)  # b h n1
            non_diag = window_w - diag.diag_embed()  # b h q2 k2
            line_index = diag < 0.5  # b h n1
            diag_scaled = torch.where(
                line_index,
                torch.tensor(0.5, dtype=diag.dtype, device=diag.device),
                diag,  # Known pytorch bug
            )  # b h n1
            diag_scaled = diag_scaled.diag_embed()  # b h q2 k2
            diag = repeat(diag, "... n -> ... n new", new=1)  # b h n1 1
            non_diag_scaled = (0.5 / (1 - diag)) * non_diag  # b h q2 k2
            line_index = repeat(line_index, "... n -> ... n new", new=1)  # b h n1 1
            non_diag_scaled = torch.where(
                line_index, non_diag_scaled, non_diag
            )  # b h q2 k2
            patch_feature = rearrange(
                patch_feature, "... n (h d) -> ... h n d", h=self.heads
            )  # b n1 d3 -> b h v2 d4
            window_feature = einsum(
                "... h n v, ... h v d -> ... h n d", diag_scaled, patch_feature
            )  # b h n1 d4
            window_feature = rearrange(
                window_feature, "... h n d -> ... n (h d)"
            )  # b h n1 d4 -> b n1 d3
            flattened = self.window_flatten(x)  # b c h2 w2 -> b n1 d3
            flattened = self.window_norm(flattened)
            window_v = self.window_value(flattened)  # b n1 d3
            window_v = rearrange(
                window_v, "... n (h d) -> ... h n d", h=self.heads
            )  # b n1 d3 -> b h v2 d4
            window_f = einsum(
                "... h n v, ... h v d -> ... h n d", non_diag_scaled, window_v
            )
            if self.pos_emb:
                window_f += self.window_pos(window_v)
            window_f = rearrange(
                window_f, "... h n d -> ... n (h d)"
            )  # b h n1 d4 -> b n1 d3
            window_feature += self.window_out(window_f)  # b n1 d3
        else:
            window_feature = patch_feature

        return self.to_out(window_feature) + x
