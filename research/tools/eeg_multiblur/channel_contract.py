"""Native-channel contracts: EEG axis and REVE coordinates must stay paired."""
import torch


def register_channels(registry, domain, names):
    names = list(names)
    if not names or len(set(names)) != len(names):
        raise ValueError(f"{domain}: empty or duplicate channel names")
    if domain in registry and registry[domain] != names:
        raise ValueError(f"{domain}: channel order changed between recordings")
    registry[domain] = names


def channel_positions(encoder, names):
    registry = {}
    register_channels(registry, 'input', names)
    pos = encoder.get_positions(names)
    if pos.shape != (len(names), 3) or not torch.isfinite(pos).all():
        raise ValueError("Invalid REVE channel coordinates")
    if len(torch.unique(pos, dim=0)) != len(names):
        raise ValueError("Different channel names mapped to identical coordinates")
    return pos


def batch_positions(model, eeg, domain):
    pos = getattr(model, 'pos_' + domain)
    if eeg.ndim != 3 or eeg.shape[1] != len(pos):
        raise ValueError(f"{domain}: EEG shape {tuple(eeg.shape)} does not match {len(pos)} channels")
    return pos.to(eeg).unsqueeze(0).expand(len(eeg), -1, -1)
