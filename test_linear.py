import torch

from model import rank, world_size
from model import ColumnParallelLinear, Linear


def test_column_parallel_linear():
    batch_size, seqlen = 4, 512
    in_features, out_features = 128, 256
    bias = True
    device = "cuda"
    dtype = torch.bfloat16
    x = torch.rand(
        batch_size, seqlen, in_features, dtype=torch.bfloat16, device=device, requires_grad=True
    )
    reference_linear = Linear(in_features, out_features, bias=bias, dtype=dtype).to(device)
    column_linear = ColumnParallelLinear(in_features, out_features, bias=bias, dtype=dtype, all_gather=True).to(device)

    # Forward, all_gather = True
    ref_out = reference_linear(x)
    # [4, 512, 128] @ [256 / TP, 128].T -> [4, 512, 256]  # all_gather = True
    out = column_linear(x)

    assert out.shape == torch.Size([batch_size, seqlen, out_features])
    assert torch.allclose(ref_out, out), (ref_out - out).abs().max().item()

    shard = out_features // world_size
    start, end = rank * shard, (rank + 1) * shard

    # Backward, all_gather = True
    ref_out.sum().backward()
    ref_wg, ref_bg, ref_xg = (
        reference_linear.weight.grad[start:end, :].detach().clone(),
        reference_linear.bias.grad[start:end].detach().clone(),
        x.grad.detach().clone()
    )
    out.sum().backward()
    assert torch.allclose(column_linear.weight.grad, ref_wg)
    assert torch.allclose(column_linear.bias.grad, ref_bg)
    
    # Forward, all_gather = False
    # Need to reset gradients and create new forward passes
    x.grad = None
    reference_linear.weight.grad = None
    reference_linear.bias.grad = None

    x2 = torch.rand(
        batch_size, seqlen, in_features, dtype=torch.bfloat16, device=device, requires_grad=True
    )
    ref_out2 = reference_linear(x2)
    
    column_linear = ColumnParallelLinear(in_features, out_features, bias=bias, dtype=dtype, all_gather=False).to(device)
    # [4, 512, 128] @ [256 / TP, 128].T -> [4, 512, 256 / TP]  # all_gather = False
    out2 = column_linear(x2)
    assert out2.shape == torch.Size([batch_size, seqlen, out_features // world_size])
    
    sharded_ref_output = ref_out2[:, :, start: end]
    assert torch.allclose(sharded_ref_output, out2), (sharded_ref_output - out2).abs().max().item()

    # Backward, all_gather = False
    ref_out2.sum().backward()
    ref_wg2, ref_bg2, ref_xg2 = (
        reference_linear.weight.grad[start:end, :].detach().clone(),
        reference_linear.bias.grad[start:end].detach().clone(),
        x2.grad.detach().clone()
    )
    out2.sum().backward()
    assert torch.allclose(column_linear.weight.grad, ref_wg2)
    assert torch.allclose(column_linear.bias.grad, ref_bg2)

    print("Test Linear passed")
