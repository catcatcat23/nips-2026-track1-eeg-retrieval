# REVE architecture adapted from Braindecode (BSD-3-Clause), Jonathan Lys.
# See LICENSE_BRAINDECODE.txt. All inference code is contained in this file.
import json
import math
import logging
from pathlib import Path
from typing import Union
import torch
import torch.nn.functional as F
from torch import nn
from benchmark_utils.base_solver import CompetSolver
HAS_SDPA = True
logger = logging.getLogger(__name__)

class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return F.gelu(gates) * x


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, geglu: bool):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, hidden_dim * 2 if geglu else hidden_dim, bias=False),
            GEGLU() if geglu else nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


#################################################################################
#                                  Attention                                    #
#################################################################################


class ClassicalAttention(nn.Module):
    def __init__(self, heads: int, use_sdpa: bool | None = None):
        super().__init__()
        self.heads = heads

        if use_sdpa is None:
            self.use_sdpa = HAS_SDPA
        elif use_sdpa is True and not HAS_SDPA:
            logger.warning(
                "SDPA (Scaled Dot Product Attention) was requested, but it is not "
                "available in your current PyTorch version. Falling back to naive "
                "implementation. Please upgrade to PyTorch >= 2.2 for SDPA support."
            )
            self.use_sdpa = False
        else:
            self.use_sdpa = use_sdpa

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        # Split concatenated QKV into separate tensors
        # qkv shape: (batch, seq_len, 3 * heads * head_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention: split last dim into (heads, head_dim)
        # (batch, seq_len, heads * head_dim) -> (batch, heads, seq_len, head_dim)
        q, k, v = (
            t.reshape(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2)
            for t in (q, k, v)
        )

        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            scale = q.shape[-1] ** -0.5
            out = ((q @ k.transpose(-1, -2)) * scale).softmax(-1) @ v
        out = out.transpose(1, 2).reshape(qkv.shape[0], qkv.shape[1], -1)
        return out


class Attention(nn.Module):
    """
    Multi-head self-attention layer with RMSNorm.
    """

    def __init__(self, dim: int, heads: int = 8, head_dim: int = 64):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.attend = ClassicalAttention(self.heads, use_sdpa=True)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x)
        out = self.attend(qkv)
        return self.to_out(out)


#################################################################################
#                                  Transformer                                  #
#################################################################################


class TransformerBackbone(nn.Module):
    """
    Transformer backbone consisting of multiple layers of attention and feed-forward networks.
    """

    def __init__(self, dim, depth, heads, head_dim, mlp_dim, geglu):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            self.dim,
                            heads=heads,
                            head_dim=head_dim,
                        ),
                        FeedForward(self.dim, mlp_dim, geglu),
                    ]
                )
            )

    def forward(
        self, x, return_out_layers=False
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        out_layers = [x] if return_out_layers else []
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            if return_out_layers:
                out_layers.append(x)
        return out_layers if return_out_layers else x


##################################################################################
#                                       4D PE                                    #
##################################################################################


class FourierEmb4D(nn.Module):
    """
    Fourier positional embedding for 4D positions (x, y, z, t).
    This version allows for a reduced number of frequencies (n_freqs),
    and ensures the output embedding has the specified dimension.

    Parameters
    ----------
    dimension : int
        The dimension of the output embedding. Must be an even number.
    freqs : int
        The number of frequencies to use for the Fourier embedding.
    increment_time : float, optional
        The time increment to scale the time dimension. Default is 0.1.
    margin : float, optional
        The margin to add to the position coordinates to avoid boundary issues. Default is 0.4.

    """

    def __init__(
        self, dimension: int, freqs: int, increment_time=0.1, margin: float = 0.4
    ):
        super().__init__()
        self.dimension = dimension
        self.freqs = freqs
        self.increment_time = increment_time
        self.margin = margin

    def forward(self, positions_: torch.Tensor) -> torch.Tensor:
        positions = positions_.clone()
        positions[:, :, -1] *= self.increment_time
        input_shape = positions.shape
        batch_dims = list(input_shape[:-1])

        freqs_w = torch.arange(self.freqs).to(positions)
        freqs_z = freqs_w[:, None]
        freqs_y = freqs_z[:, None]
        freqs_x = freqs_y[:, None]
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        p_z = 2 * math.pi * freqs_z / width
        p_w = 2 * math.pi * freqs_w / width
        positions = positions[..., None, None, None, None, :]
        loc = (
            positions[..., 0] * p_x
            + positions[..., 1] * p_y
            + positions[..., 2] * p_z
            + positions[..., 3] * p_w
        )
        batch_dims.append(-1)
        loc = loc.view(batch_dims)

        half_dim = self.dimension // 2
        current_dim = loc.shape[-1]
        if current_dim != half_dim:
            if current_dim > half_dim:
                loc = loc[..., :half_dim]
            else:
                raise ValueError(
                    f"Input dimension ({current_dim}) is too small for target "
                    f"embedding dimension ({self.dimension}). Expected at least {half_dim}."
                )

        emb = torch.cat([torch.cos(loc), torch.sin(loc)], dim=-1)
        return emb

    @classmethod
    def add_time_patch(cls, pos: torch.Tensor, num_patches: int) -> torch.Tensor:
        """
        Expand the position tensor by adding a time dimension, handling batched data.

        Parameters
        ----------
        pos : torch.Tensor
            Input tensor of shape (B, C, 3), where B is the batch size,
            C is the number of channels, and 3 represents x, y, z.
        num_patches : int
            The number of time patches.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, C * num_patches, 4), where each position is repeated with each time value.
        """
        batch, nchans, _ = pos.shape
        # Repeat each position for each time step
        pos_repeated = pos.unsqueeze(2).repeat(
            1, 1, num_patches, 1
        )  # Shape: (batch, nchans, num_patches, 3)
        # Generate time values with the specified increment
        time_values = torch.arange(
            0, num_patches, 1, device=pos.device
        ).float()  # Shape: (num_patches,)
        time_values = time_values.view(1, 1, num_patches, 1).expand(
            batch, nchans, num_patches, 1
        )  # (batch, nchans, num_patches, 1)
        # Concatenate the repeated positions with the time values along the last dimension
        pos_with_time = torch.cat(
            (pos_repeated, time_values), dim=-1
        )  # Shape: (batch, nchans, num_patches, 4)
        # Reshape to (batch, nchans * num_patches, 4)
        pos_with_time = pos_with_time.view(batch, nchans * num_patches, 4)

        return pos_with_time



class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.net = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, 3072),
                                 nn.GELU(), nn.Dropout(0.1), nn.Linear(3072, 1536))

    def forward(self, x):
        return x + self.alpha * self.net(x)


class RidgeResidual(nn.Module):
    def __init__(self):
        super().__init__()
        for name, shape in [('mean',(1536,)),('scale',(1536,)),('weight',(1536,1536)),('bias',(1536,))]:
            self.register_buffer(name, torch.zeros(shape))
        self.delta=nn.Sequential(nn.LayerNorm(1536),ResidualBlock(),ResidualBlock(),nn.LayerNorm(1536),nn.Linear(1536,1536))
    def forward(self,h):
        return ((h-self.mean)/self.scale)@self.weight+self.bias+0.1*self.delta(h)


class ReveRetrieval(nn.Module):
    """Frozen inference-only copy of the fine-tuned REVE + mean/linear head."""
    def __init__(self, positions, n_times, sfreq):
        super().__init__()
        self.register_buffer('positions', torch.tensor(positions, dtype=torch.float32), persistent=False)
        self.input_times = int(n_times)
        self.input_sfreq = float(sfreq)
        self.resampled_times = round(self.input_times * 200.0 / self.input_sfreq)
        if self.resampled_times < 200:
            raise ValueError('REVE requires at least one second of EEG')
        self.to_patch_embedding = nn.Sequential(nn.Linear(200,512))
        self.fourier4d = FourierEmb4D(512, freqs=4)
        self.mlp4d = nn.Sequential(nn.Linear(4,512,bias=False),nn.GELU(),nn.LayerNorm(512))
        self.ln = nn.LayerNorm(512)
        self.transformer = TransformerBackbone(512,22,8,64,int(512*2.66),True)
        self.head = nn.Linear(512,1536)
        self.recovery = RidgeResidual()

    def forward(self, eeg):
        patches = eeg.unfold(2,200,180)
        pos = self.positions.to(eeg).expand(len(eeg),-1,-1)
        pos = FourierEmb4D.add_time_patch(pos,patches.shape[2])
        tokens = self.to_patch_embedding(patches).flatten(1,2)
        tokens = tokens + self.ln(self.fourier4d(pos)+self.mlp4d(pos))
        return self.recovery(self.head(self.transformer(tokens).mean(dim=1)))

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        if X.ndim != 3 or X.shape[1] != len(self.positions) or X.shape[-1] != self.input_times:
            raise ValueError('Input shape disagrees with meta')
        if len(X) == 0:
            return torch.empty(0, 1536, device=self.positions.device)
        results = []
        # Bound activations independently of platform batch size and device.
        chunk_size = 32 if self.positions.device.type == 'cuda' else 16
        for part in X.split(chunk_size):
            part = part.to(device=self.positions.device, dtype=torch.float32)
            if not torch.isfinite(part).all():
                raise ValueError('EEG input contains nonfinite values')
            if part.shape[-1] != self.resampled_times:
                part = F.interpolate(part, size=self.resampled_times, mode='linear', align_corners=False)
            mean = part.mean(dim=-1, keepdim=True)
            std = part.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
            part = ((part-mean)/std).clamp(-15.,15.)
            results.append(self(part))
        return torch.cat(results, dim=0)


class Solver(CompetSolver):
    name = 'REVE-v39B-RidgeResidual-Epoch8'
    requirements = []

    def load_model(self, meta):
        root = Path(meta['submission_dir'])
        if int(meta['n_outputs']) != 1536:
            raise ValueError('This trained model targets the fixed 1536-D DINOv2 space')
        names = meta.get('ch_names')
        if names is None or len(names) != int(meta['n_chans']):
            raise ValueError('meta must supply channel names in input order')
        bank = json.loads((root/'positions.json').read_text())
        lookup = {name.upper():value for name,value in bank.items()}
        coordinates = []
        for name in names:
            name = str(name).removeprefix('EEG ').split('-',1)[0].upper()
            if name not in lookup:
                raise ValueError('Channel not present in the shipped REVE position bank: '+name)
            coordinates.append(lookup[name])
        model = ReveRetrieval(coordinates,meta['n_times'],meta['sfreq'])
        state = torch.load(root/'weights.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state,strict=True)
        return model.to(meta['device']).eval()
