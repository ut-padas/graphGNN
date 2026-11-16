#!/usr/bin/env python3
"""Verify that PyTorch and PyTorch Geometric can execute on a visible GPU."""

import argparse

import torch
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-rocm",
        action="store_true",
        help="fail unless PyTorch is a ROCm build",
    )
    parser.add_argument(
        "--require-xpu",
        action="store_true",
        help="fail unless PyTorch runs on an Intel XPU",
    )
    args = parser.parse_args()

    if torch.cuda.is_available() and not args.require_xpu:
        backend = torch.cuda
        device = torch.device("cuda:0")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        backend = torch.xpu
        device = torch.device("xpu:0")
    else:
        raise SystemExit(
            "GPU smoke test failed: PyTorch cannot see a GPU. Run inside a GPU "
            "allocation with the matching accelerator modules and PyTorch build."
        )

    device_count = backend.device_count()
    if args.require_rocm and torch.version.hip is None:
        raise SystemExit(
            f"GPU smoke test failed: --require-rocm was set, but PyTorch "
            f"{torch.__version__} is not a ROCm build."
        )

    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch Geometric: {torch_geometric.__version__}")
    print(f"Visible GPUs: {device_count}")
    print(f"Selected GPU: {backend.get_device_name(device)} ({device})")
    print(f"ROCm HIP version: {torch.version.hip or 'not a ROCm build'}")

    values = torch.arange(4096, dtype=torch.float32, device=device)
    result = values.square().sum()
    backend.synchronize(device)
    assert result.device == device and torch.isfinite(result).item() and result.item() > 0
    print(f"PyTorch GPU operation passed: result on {result.device}")

    graph = HeteroData()
    graph["grain"].x = torch.randn(4, 3)
    graph["joint"].x = torch.randn(3, 3)
    graph["grain", "contacts", "joint"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 1, 2]], dtype=torch.long
    )
    graph["joint", "contacted_by", "grain"].edge_index = torch.tensor(
        [[0, 1, 1, 2], [0, 1, 2, 3]], dtype=torch.long
    )
    graph = graph.to(device)
    assert all(tensor.device == device for tensor in graph.x_dict.values())
    assert all(tensor.device == device for tensor in graph.edge_index_dict.values())

    convolution = HeteroConv(
        {
            ("grain", "contacts", "joint"): SAGEConv((3, 3), 5),
            ("joint", "contacted_by", "grain"): SAGEConv((3, 3), 5),
        },
        aggr="sum",
    ).to(device)
    outputs = convolution(graph.x_dict, graph.edge_index_dict)
    loss = sum(output.square().mean() for output in outputs.values())
    loss.backward()
    backend.synchronize(device)

    assert set(outputs) == {"grain", "joint"}
    assert outputs["grain"].shape == (4, 5)
    assert outputs["joint"].shape == (3, 5)
    assert all(output.device == device and torch.isfinite(output).all().item() for output in outputs.values())
    assert any(parameter.grad is not None for parameter in convolution.parameters())
    print(f"PyG HeteroConv/SAGEConv passed: outputs { {key: tuple(value.shape) for key, value in outputs.items()} } on {device}")
    print("GPU smoke test passed.")


if __name__ == "__main__":
    main()