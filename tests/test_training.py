import torch
import torch.nn.functional as F

from afnn import AFNN, AFNNConfig, DynamicLRScheduler, EMA, Trainer


def make_trainer(**cfg_kw):
    cfg_args = dict(input_size=16, num_classes=4, base_channels=8,
                    channel_multipliers=(1.0, 1.0), stages=2, initial_depth=1,
                    initial_branches=2, max_branches_per_stage=4,
                    max_total_params=200_000, use_mixup=False, use_cutmix=False,
                    use_amp=False, use_ema=True, ema_decay=0.99, growth_patience=1,
                    growth_min_delta=10.0, rejuvenation_enabled=False)
    cfg_args.update(cfg_kw)
    cfg = AFNNConfig(**cfg_args)
    model = AFNN(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.defaults["_new_branch_warmup_epochs"] = cfg.new_branch_warmup_epochs
    for g in opt.param_groups:
        g["_base_lr"] = g["lr"]
    sched = DynamicLRScheduler(opt, cfg.warmup_epochs, cfg.cosine_t_max, 1e-3)
    return Trainer(model, opt, sched, cfg, device="cpu"), cfg


def batches(n=3, size=16):
    torch.manual_seed(0)
    return [(torch.randn(8, 3, size, size), torch.randint(0, 4, (8,))) for _ in range(n)]


def test_epoch_with_growth_runs():
    trainer, _ = make_trainer()
    data = batches()
    trainer.step_epoch(data, data, epoch=0)         # first epoch only sets the best loss
    before = trainer.model.count_total_branches()
    info = trainer.step_epoch(data, data, epoch=1)  # no improvement -> growth
    assert info["grew"] and trainer.model.count_total_branches() > before
    # After growth the optimizer must contain every parameter of the model.
    ids = {id(p) for g in trainer.opt.param_groups for p in g["params"]}
    assert ids == {id(p) for p in trainer.model.parameters()}
    trainer.step_epoch(data, data, epoch=2)     # keeps training with the new branch


def test_fused_ema_matches_reference():
    trainer, _ = make_trainer()
    model = trainer.model
    ref = EMA(model, 0.9)
    fused = EMA(model, 0.9)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1)
    ref.update(model)
    fused.update(model, steps=1)
    for k in ref.shadow:
        assert torch.allclose(ref.shadow[k].float(), fused.shadow[k].float())


def test_checkpoint_roundtrip(tmp_path):
    trainer, cfg = make_trainer()
    data = batches()
    trainer.step_epoch(data, data, epoch=0)
    trainer.step_epoch(data, data, epoch=1)          # includes growth
    x = data[0][0]
    trainer.model.eval()
    with torch.no_grad():
        expected = trainer.model(x).clone()

    path = str(tmp_path / "ckpt.pt")
    trainer.save_checkpoint(path, class_names=["a", "b", "c", "d"])
    model, loaded_cfg, ckpt = Trainer.load_checkpoint(path, device="cpu")
    assert loaded_cfg.to_dict() == cfg.to_dict()
    assert ckpt["class_names"] == ["a", "b", "c", "d"]
    model.eval()
    with torch.no_grad():
        assert torch.allclose(expected, model(x), atol=1e-5)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = DynamicLRScheduler(opt, cfg.warmup_epochs, cfg.cosine_t_max, 1e-3)
    resumed = Trainer(model, opt, sched, cfg, device="cpu")
    resumed.load_training_state(ckpt)
    assert len(resumed.opt.state) > 0
    model.train()
    F.cross_entropy(model(x), data[0][1]).backward()
    resumed.opt.step()


def test_ensure_device_is_a_noop_on_correct_device():
    trainer, _ = make_trainer()
    trainer.step_epoch(batches(), batches(), epoch=0)
    trainer.step_epoch(batches(), batches(), epoch=1)
    trainer._ensure_device()
    assert all(p.device.type == "cpu" for p in trainer.model.parameters())
