"""Shape validation test for SE-TransNet V4."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from se_transnet.models.se_transnet import SETransNet


def test_shapes():
    print('=' * 60)
    print('SE-TransNet V4 - Shape Validation Test')
    print('=' * 60)

    config = dict(
        num_classes=4, num_samples=800, num_channels=62,
        embed_dim=80, temporal_kernels=[25, 51, 101, 201, 401],
        pool_size=40, pool_stride=10, num_heads=8,
        fc_ratio=4, depth=6, attn_drop=0.1,
        fc_drop=0.5, spatial_drop=0.25,
    )

    net = SETransNet(**config)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'\nModel parameters: {n_params:,}')

    T_seq = (800 - 40) // 10 + 1
    print(f'Expected T_seq: {T_seq}')
    assert T_seq == 77

    print('\n--- Test 1: Forward pass ---')
    x = torch.randn(4, 62, 800)
    out = net(x)
    assert out.shape == (4, 4), f'Output shape: {out.shape}'
    print(f'  Input:  {x.shape}  Output: {out.shape}  PASS')

    print('\n--- Test 2: Gradient flow ---')
    out.sum().backward()
    n_with_grad = sum(1 for p in net.parameters()
                      if p.requires_grad and p.grad is not None
                      and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in net.parameters() if p.requires_grad)
    print(f'  Grads: {n_with_grad}/{n_total}')
    assert n_with_grad == n_total
    print('  All parameters receive gradients  PASS')

    print('\n--- Test 3: Feature extraction ---')
    net.zero_grad()
    feats = net.extract_features(torch.randn(4, 62, 800))
    assert feats.shape == (4, 256), f'Feature shape: {feats.shape}'
    print(f'  Features: {feats.shape}  PASS')

    print('\n--- Test 4: Batch size 1 ---')
    net.eval()
    with torch.no_grad():
        out3 = net(torch.randn(1, 62, 800))
    assert out3.shape == (1, 4)
    print(f'  Single sample: {out3.shape}  PASS')

    # Test 5: LOSO trainer import
    print('\n--- Test 5: LOSOTrainer import ---')
    from se_transnet.training.trainer_cs import LOSOTrainer
    print(f'  LOSOTrainer imported  PASS')
    print(f'  Methods: {[m for m in dir(LOSOTrainer) if not m.startswith("_")]}')

    if torch.cuda.is_available():
        print('\n--- Test 6: CUDA ---')
        net_gpu = SETransNet(**config).cuda()
        out_gpu = net_gpu(torch.randn(4, 62, 800).cuda())
        assert out_gpu.shape == (4, 4)
        mem = torch.cuda.max_memory_allocated() / 1e6
        print(f'  CUDA: {out_gpu.shape}  Peak VRAM: {mem:.1f} MB  PASS')

    print('\n' + '=' * 60)
    print('ALL TESTS PASSED')
    print(f'Model: {n_params:,} params | Shape: (B,62,800)->(B,4) | T_seq: {T_seq}')
    print('=' * 60)


if __name__ == '__main__':
    test_shapes()
