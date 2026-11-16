"""Intel GPU (XPU) tests for ALCF Aurora; skipped unless PyTorch can see an XPU.

Run inside an Aurora compute-node allocation with:
    python -m pytest -m xpu tests/test_xpu.py
"""
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "tests"))

pytestmark = [
    pytest.mark.xpu,
    pytest.mark.skipif(
        not (hasattr(torch, "xpu") and torch.xpu.is_available()),
        reason="no Intel GPU (XPU) visible to PyTorch",
    ),
]

XPU = torch.device("xpu:0")


def _samples(seed=11):
    from extract.grain_field import Field
    from extract.samples import build_samples
    from extract.stack_tracking import track_stack
    from test_stack_tracking import moving_seeds_stack

    traj = track_stack(Field(ids=moving_seeds_stack(speed=0.5, seed=seed), cell=1.0, dz=1.0))
    return build_samples(traj, 2.0, 1.0, span=3, seed=seed)


def _hyper(params, samples, metadata, device):
    params.features = samples[0].features
    params.targets = samples[0].targets
    params.metadata = metadata
    params.device = device
    return params


def _batch(samples):
    from data_loader import DynamicHeteroGraphTemporalSignal

    return DynamicHeteroGraphTemporalSignal(samples)[0]


def _materialize(model, batch):
    # SAGEConv(-1, -1) layers are lazy: one forward pass creates their parameters
    with torch.no_grad():
        model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)


def _assert_on_xpu(tensors):
    for name, tensor in tensors.items():
        assert tensor.device.type == "xpu", f"{name} is on {tensor.device}, expected xpu"
        assert torch.isfinite(tensor).all(), f"{name} has non-finite values on xpu"


def test_xpu_backend_is_visible():
    assert torch.xpu.device_count() >= 1
    name = torch.xpu.get_device_name(XPU)
    print(f"PyTorch {torch.__version__}; {torch.xpu.device_count()} XPUs; xpu:0 = {name}")
    x = torch.arange(1024, dtype=torch.float32, device=XPU)
    total = x.square().sum()
    torch.xpu.synchronize(XPU)
    assert total.device == XPU and total.item() == pytest.approx(float(sum(i * i for i in range(1024))))


def test_regressor_runs_on_xpu_and_matches_cpu():
    from models import GrainNN_regressor
    from parameters import regressor

    samples = _samples()
    batch = _batch(samples)
    metadata = batch.metadata()

    torch.manual_seed(0)
    cpu_model = GrainNN_regressor(_hyper(regressor(0), samples, metadata, "cpu"), False, True).eval()
    xpu_model = GrainNN_regressor(_hyper(regressor(0), samples, metadata, XPU), False, True).to(XPU).eval()
    xpu_batch = _batch(samples).to(XPU)
    _materialize(cpu_model, batch)
    _materialize(xpu_model, xpu_batch)
    xpu_model.load_state_dict(cpu_model.state_dict())
    assert all(p.device.type == "xpu" for p in xpu_model.parameters())

    with torch.no_grad():
        cpu_out = cpu_model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        xpu_out = xpu_model(xpu_batch.x_dict, xpu_batch.edge_index_dict, xpu_batch.edge_attr_dict)
    torch.xpu.synchronize(XPU)

    compared = {k: v for k, v in xpu_out.items() if torch.is_tensor(v) and v.is_floating_point()}
    assert {"joint", "grain", "edge"} <= set(compared)
    _assert_on_xpu(compared)
    for key, value in compared.items():
        torch.testing.assert_close(value.cpu(), cpu_out[key], rtol=1e-3, atol=1e-4, msg=f"output '{key}'")


def test_regressor_training_step_on_xpu():
    from models import GrainNN_regressor
    from parameters import regressor

    samples = _samples()
    batch = _batch(samples)
    model = GrainNN_regressor(_hyper(regressor(0), samples, batch.metadata(), XPU), False, True).to(XPU)
    batch = batch.to(XPU)
    _materialize(model, batch)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    model.train()
    out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
    mask, y = batch["mask"], batch.y_dict
    loss = torch.mean(mask["joint"] * (y["joint"] - out["joint"]) ** 2) \
        + torch.mean(mask["grain"] * (y["grain"] - out["grain"]) ** 2) \
        + torch.mean(mask["edge"] * (y["edge"] - out["edge"]) ** 2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.xpu.synchronize(XPU)

    assert loss.device.type == "xpu" and torch.isfinite(loss)
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert grads, "no gradients were produced"
    _assert_on_xpu(grads)
    assert any(not torch.equal(before[n], p.detach()) for n, p in model.named_parameters())


def test_classifier_runs_on_xpu():
    from models import GrainNN_classifier, GrainNN_regressor
    from parameters import classifier_transfered, regressor

    samples = _samples()
    batch = _batch(samples)
    metadata = batch.metadata()
    pretrained = GrainNN_regressor(_hyper(regressor(0), samples, metadata, XPU), False, False)
    model = GrainNN_classifier(_hyper(classifier_transfered(0), samples, metadata, XPU), pretrained).to(XPU)
    batch = batch.to(XPU)

    out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
    y = batch.y_dict["edge_event"]
    qualified = torch.where(y > -1)
    loss = torch.nn.BCEWithLogitsLoss()(out["edge_event"][qualified], y[qualified].float())
    loss.backward()
    torch.xpu.synchronize(XPU)

    _assert_on_xpu({"edge_event": out["edge_event"]})
    assert torch.isfinite(loss)
    _assert_on_xpu({n: p.grad for n, p in model.named_parameters() if p.grad is not None})


def test_train_script_uses_xpu(tmp_path):
    samples = _samples(seed=11) + _samples(seed=12)
    with open(tmp_path / "dataset_train.pkl", "wb") as f:
        pickle.dump(samples, f)
    model_dir = tmp_path / "models"

    result = subprocess.run(
        [
            sys.executable, str(REPOSITORY / "train.py"),
            "--dataset", "train",
            "--device", "xpu",
            "--model_dir", str(model_dir) + os.sep,
            "--model_type", "regressor",
            "--model_id", "0",
            "--train_ratio", "0.5",
            "--epochs", "1",
            "--edge_len",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert "model parameters on device: xpu" in result.stdout

    state = torch.load(model_dir / "regressor0.pt", map_location="cpu")
    assert state and all(torch.isfinite(t).all() for t in state.values() if t.is_floating_point())
