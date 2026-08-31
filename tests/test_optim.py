import torch.nn as nn

from alexnet.config import Config
from alexnet.model import AlexNetBN
from alexnet.optim import build_optimizer, build_scheduler, param_groups, scaled_lr


def test_lr_scales_linearly_with_global_batch():
    cfg = Config()
    cfg.data.batch_size = 256
    assert scaled_lr(cfg, 1) == 0.01
    assert scaled_lr(cfg, 2) == 0.02  # per-GPU batch fixed, global batch doubles


def test_explicit_lr_overrides_scaling():
    cfg = Config()
    cfg.optim.lr = 0.123
    assert scaled_lr(cfg, 8) == 0.123


def test_no_weight_decay_on_bn_and_bias():
    m = AlexNetBN()
    decay, no_decay = param_groups(m, 5e-4)
    assert decay["weight_decay"] == 5e-4
    assert no_decay["weight_decay"] == 0.0
    no_decay_ids = {id(p) for p in no_decay["params"]}
    for mod in m.modules():
        for name, p in mod.named_parameters(recurse=False):
            is_bn = isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d))
            if is_bn or name == "bias":
                assert id(p) in no_decay_ids, f"{type(mod).__name__}.{name} decays"
    # Both groups together must cover every parameter exactly once.
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(m.parameters()))


def test_schedule_warms_up_then_decays_to_floor():
    cfg = Config()
    cfg.train.epochs = 10
    cfg.optim.warmup_epochs = 2
    steps = 20
    m = nn.Linear(4, 4)
    opt = build_optimizer(m, cfg, world_size=1)
    sched = build_scheduler(opt, cfg, steps_per_epoch=steps)

    lrs = []
    for _ in range(cfg.train.epochs * steps):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()

    peak = scaled_lr(cfg, 1)
    assert lrs[0] < peak                       # starts low
    assert lrs[cfg.optim.warmup_epochs * steps - 1] == max(lrs)  # peaks at warmup end
    assert max(lrs) == peak
    assert lrs[-1] < peak * 1e-3               # cosine decays to ~0
    warm = lrs[: cfg.optim.warmup_epochs * steps]
    assert warm == sorted(warm)                # warmup is monotonically increasing
