import logging
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb


@dataclass
class TrainConfig:
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    seed: int = 1
    dt: float = 0.05
    train_frac: float = 0.8
    val_frac: float = 0.1
    num_workers: int = 0


class DeviceInterpolationDataset(Dataset):
    def __init__(
        self,
        experiment,
        times: np.ndarray,
        input_device_name: str = "spikes",
        target_device_name: str = "cursor",
    ):
        self.input_device = experiment.devices[input_device_name]
        self.target_device = experiment.devices[target_device_name]
        self.times = np.asarray(times, dtype=np.float64)

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        t = np.array([self.times[idx]], dtype=np.float64)

        x = self.input_device.interpolate(t)
        y = self.target_device.interpolate(t)

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if x.ndim == 2:
            x = x[0]
        if y.ndim == 2:
            y = y[0]

        return {
            "spikes": torch.from_numpy(x),
            "vel": torch.from_numpy(y),
        }


class LinearDecoder(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


def get_time_bounds(experiment):
    spikes_dev = experiment.devices["spikes"]
    cursor_dev = experiment.devices["cursor"]

    start_time = max(spikes_dev.start_time, cursor_dev.start_time)
    end_time = min(spikes_dev.end_time, cursor_dev.end_time)

    if not start_time < end_time:
        raise ValueError("No overlapping time range between spikes and cursor.")

    return float(start_time), float(end_time)


def split_times(times: np.ndarray, train_frac: float, val_frac: float):
    n = len(times)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    train_times = times[:n_train]
    val_times = times[n_train:n_train + n_val]
    test_times = times[n_train + n_val:n_train + n_test]

    return train_times, val_times, test_times


def compute_target_stats(dataset):
    ys = []
    for i in range(len(dataset)):
        ys.append(dataset[i]["vel"].unsqueeze(0))
    y = torch.cat(ys, dim=0)
    mean = y.mean(dim=0, keepdim=True)
    std = y.std(dim=0, keepdim=True).clamp_min(1e-6)
    return mean, std


class NormalizedWrapper(Dataset):
    def __init__(self, base_dataset, target_mean, target_std):
        self.base_dataset = base_dataset
        self.target_mean = target_mean
        self.target_std = target_std

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        sample["vel"] = (sample["vel"] - self.target_mean.squeeze(0)) / self.target_std.squeeze(0)
        return sample


def make_dataloaders(experiment, cfg: TrainConfig):
    start_time, end_time = get_time_bounds(experiment)
    times = np.arange(start_time, end_time, cfg.dt, dtype=np.float64)

    if len(times) == 0:
        raise ValueError("No sample times generated. Check dt and time bounds.")

    train_times, val_times, test_times = split_times(times, cfg.train_frac, cfg.val_frac)

    raw_train_ds = DeviceInterpolationDataset(experiment, train_times)
    target_mean, target_std = compute_target_stats(raw_train_ds)

    train_ds = NormalizedWrapper(DeviceInterpolationDataset(experiment, train_times), target_mean, target_std)
    val_ds = NormalizedWrapper(DeviceInterpolationDataset(experiment, val_times), target_mean, target_std)
    test_ds = NormalizedWrapper(DeviceInterpolationDataset(experiment, test_times), target_mean, target_std)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    return train_loader, val_loader, test_loader, target_mean, target_std


def run_epoch(model, loader, optimizer, loss_fn, device, train: bool):
    model.train(train)

    total_loss = 0.0
    total_n = 0

    for batch in loader:
        x = batch["spikes"].to(device=device, dtype=torch.float32)
        y = batch["vel"].to(device=device, dtype=torch.float32)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = loss_fn(pred, y)

            if train:
                loss.backward()
                optimizer.step()

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_n += batch_size

    return total_loss / max(total_n, 1)


def evaluate_rmse(model, loader, device, target_mean, target_std):
    model.eval()

    preds_all = []
    targets_all = []

    with torch.no_grad():
        for batch in loader:
            x = batch["spikes"].to(device=device, dtype=torch.float32)
            y = batch["vel"].to(device=device, dtype=torch.float32)

            pred = model(x).cpu()
            y = y.cpu()

            pred = pred * target_std + target_mean
            y = y * target_std + target_mean

            preds_all.append(pred)
            targets_all.append(y)

    if not preds_all:
        return float("nan")

    preds = torch.cat(preds_all, dim=0)
    targets = torch.cat(targets_all, dim=0)
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()


def main(
    experiment,
    wandb_entity: str,
    wandb_project: str = "exp_test_pipeline_linear",
    checkpoint_path: str | None = None,
    logging_level=logging.INFO,
):
    logging.basicConfig(
        level=logging_level,
        format="%(levelname)s [%(name)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    cfg = TrainConfig()

    run_name = "exp_test_pipeline_linear"
    if "SLURM_JOB_ID" in os.environ:
        run_name = f"{run_name}_{os.environ['SLURM_JOB_ID']}"

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        config=cfg.__dict__,
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_loader, val_loader, test_loader, target_mean, target_std = make_dataloaders(
        experiment=experiment,
        cfg=cfg,
    )

    first_batch = next(iter(train_loader))
    print("Batch keys:", first_batch.keys())
    print("spikes shape:", first_batch["spikes"].shape)
    print("vel shape:", first_batch["vel"].shape)

    in_features = first_batch["spikes"].shape[-1]
    out_features = first_batch["vel"].shape[-1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LinearDecoder(in_features=in_features, out_features=out_features).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(cfg.max_epochs):
        train_loss = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, loss_fn, device, train=False)
        val_rmse = evaluate_rmse(model, val_loader, device, target_mean, target_std)

        log_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_rmse": val_rmse,
        }

        if len(test_loader.dataset) > 0:
            test_loss = run_epoch(model, test_loader, optimizer, loss_fn, device, train=False)
            test_rmse = evaluate_rmse(model, test_loader, device, target_mean, target_std)
            log_data["test_loss"] = test_loss
            log_data["test_rmse"] = test_rmse

        wandb.log(log_data)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_rmse={val_rmse:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if checkpoint_path is not None and best_state is not None:
        os.makedirs(checkpoint_path, exist_ok=True)
        torch.save(best_state, os.path.join(checkpoint_path, "best_linear_decoder.pth"))

    wandb.finish()