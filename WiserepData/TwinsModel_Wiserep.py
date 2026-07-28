import pathlib
import sys
import importlib
import torch
from torch import nn


def repo_import_setup(repo_dir: str):
    repo = pathlib.Path(repo_dir).resolve()
    pkg = repo / "package"
    cannon = repo / "cannon"
    for p in (str(pkg), str(cannon)):
        if p not in sys.path:
            sys.path.insert(0, p)
    importlib.invalidate_caches()
    return repo


def device_from_str(s: str) -> torch.device:
    """Return ``cuda``, ``mps`` (Apple GPU), or ``cpu``. ``auto`` prefers cuda, then mps, then cpu."""
    s = s.strip().lower()
    mps_ok = bool(
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    )
    if s == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_ok:
            return torch.device("mps")
        return torch.device("cpu")
    if s == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if s == "mps":
        return torch.device("mps" if mps_ok else "cpu")
    if s == "cpu":
        return torch.device("cpu")
    return torch.device("cpu")


def default_cfg():
    return dict(
        bottleneck_length=64,
        bottleneck_dim=256,
        model_dim=256,
        num_heads=8,
        num_layers=4,
        ff_dim=512,
        dropout=0.1,
        selfattn=False,
        concat=True,
        cross_attn_only=False,
        hidden_len=256,
    )


class Daepaggregator(nn.Module):
    def __init__(
        self,
        spectraTransceiverEncoder,
        bottleneck_length,
        bottleneck_dim,
        model_dim,
        num_heads,
        num_layers,
        ff_dim,
        dropout,
        selfattn,
        concat,
    ):
        super().__init__()

        self.encoder = spectraTransceiverEncoder(
            bottleneck_length=bottleneck_length,
            bottleneck_dim=bottleneck_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            selfattn=selfattn,
            concat=concat,
        )

        self.MLPEncoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 4 * bottleneck_dim),
            nn.GELU(),
            nn.Linear(4 * bottleneck_dim, bottleneck_dim),
        )

        self.bottleneck_length = bottleneck_length
        self.bottleneck_dim = bottleneck_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.selfattn = selfattn
        self.concat = concat

    def encode_raw(self, x):
        z = self.encoder(x)       # [B, L, D]
        z = self.MLPEncoder(z)    # [B, L, D]
        return z

    def forward(self, x):
        return self.encode_raw(x)


def build_daep(cfg: dict):
    from daep.SpectraLayers import spectraTransceiverEncoder, spectraTransceiverScore2stages
    from daep.daep import unimodaldaep

    encoder = Daepaggregator(
        spectraTransceiverEncoder=spectraTransceiverEncoder,
        bottleneck_length=cfg["bottleneck_length"],
        bottleneck_dim=cfg["bottleneck_dim"],
        model_dim=cfg["model_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        ff_dim=cfg["ff_dim"],
        dropout=cfg["dropout"],
        selfattn=cfg["selfattn"],
        concat=cfg["concat"],
    )

    score = spectraTransceiverScore2stages(
        bottleneck_dim=cfg["bottleneck_dim"],
        model_dim=cfg["model_dim"],
        num_heads=cfg["num_heads"],
        ff_dim=cfg["ff_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        selfattn=cfg["selfattn"],
        concat=cfg["concat"],
        cross_attn_only=cfg["cross_attn_only"],
        hidden_len=cfg["hidden_len"],
    )

    return unimodaldaep(encoder, score, name="flux")