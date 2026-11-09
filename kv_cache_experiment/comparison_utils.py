"""Shared compression defaults and reproducible inference settings."""

DEFAULT_METHODS = "curvature_min,curvature_mean,SnapKVPress,StreamingLLMPress,KnormPress,PyramidKVPress"
DEFAULT_RATIOS = "0.3,0.5,0.7,0.9"


def configure_determinism():
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
