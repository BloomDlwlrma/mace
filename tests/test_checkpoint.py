import torch

from mace.tools.checkpoint import CheckpointBuilder, CheckpointState


def test_loading_legacy_checkpoint_preserves_new_optimizer_group_settings():
    saved_model = torch.nn.Linear(1, 1)
    saved_optimizer = torch.optim.Adam(saved_model.parameters(), lr=1e-3)
    saved_scheduler = torch.optim.lr_scheduler.ExponentialLR(saved_optimizer, gamma=0.9)
    checkpoint = {
        "model": saved_model.state_dict(),
        "optimizer": saved_optimizer.state_dict(),
        "lr_scheduler": saved_scheduler.state_dict(),
    }

    resumed_model = torch.nn.Linear(1, 1)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-3)
    resumed_scheduler = torch.optim.lr_scheduler.ExponentialLR(resumed_optimizer, gamma=0.9)
    swa_scheduler = torch.optim.swa_utils.SWALR(
        resumed_optimizer, swa_lr=2.5e-6, anneal_epochs=1
    )
    state = CheckpointState(resumed_model, resumed_optimizer, resumed_scheduler)

    CheckpointBuilder.load_checkpoint(state, checkpoint, strict=True)

    assert resumed_optimizer.param_groups[0]["swa_lr"] == 2.5e-6
    resumed_optimizer.step()
    swa_scheduler.step()
