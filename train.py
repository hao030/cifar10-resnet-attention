"""Train a small ResNet on CIFAR-10, with or without self-attention layers on top.

Usage:
    python train.py --model resnet
    python train.py --model resnet_attn
"""
import argparse
import datetime
import json
import os
import platform
import subprocess
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


# ---------------- (1) CNN with residual connections ----------------
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:  # 1x1 conv to match shape
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))  # residual connection


class ResNet(nn.Module):
    """32x32 -> stem(64) -> stage1(64,32x32) -> stage2(128,16x16) -> stage3(256,8x8)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.stage1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.stage2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.stage3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.fc = nn.Linear(256, num_classes)

    def features(self, x):
        return self.stage3(self.stage2(self.stage1(self.stem(x))))  # (B, 256, 8, 8)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.mean(dim=(2, 3)))  # global average pooling


# ---------------- (2) add self-attention layers on top ----------------
class SelfAttentionBlock(nn.Module):
    """Pre-norm Transformer encoder block: multi-head self-attention + MLP, both with residuals."""

    def __init__(self, dim, heads=4, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class ResNetAttn(ResNet):
    def __init__(self, num_classes=10, num_attn_layers=2):
        super().__init__(num_classes)
        self.pos_embed = nn.Parameter(torch.zeros(1, 8 * 8, 256))  # learnable position embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.attn_layers = nn.Sequential(*[SelfAttentionBlock(256) for _ in range(num_attn_layers)])
        self.norm = nn.LayerNorm(256)

    def forward(self, x):
        x = self.features(x)                      # (B, 256, 8, 8)
        x = x.flatten(2).transpose(1, 2)          # (B, 64 tokens, 256)
        x = self.attn_layers(x + self.pos_embed)  # self-attention over the 64 spatial positions
        return self.fc(self.norm(x).mean(dim=1))


# ---------------- training ----------------
def get_loaders(batch_size, workers):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train = torchvision.datasets.CIFAR10("data", train=True, download=True, transform=train_tf)
    test = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=test_tf)
    kw = dict(num_workers=workers, persistent_workers=workers > 0)
    return (torch.utils.data.DataLoader(train, batch_size, shuffle=True, drop_last=True, **kw),
            torch.utils.data.DataLoader(test, 500, shuffle=False, **kw))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        n += y.numel()
    return loss_sum / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["resnet", "resnet_attn"], default="resnet")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, test_loader = get_loaders(args.batch_size, args.workers)

    model = (ResNet() if args.model == "resnet" else ResNetAttn()).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() \
        if platform.system() == "Darwin" else platform.processor()
    start_at = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"start {start_at} | chip {chip} | torch {torch.__version__}")
    print(f"model={args.model} params={n_params / 1e6:.2f}M device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr * 3, epochs=args.epochs,
                                                steps_per_epoch=len(train_loader))
    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = F.cross_entropy(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * y.numel()
            correct += (out.argmax(1) == y).sum().item()
            n += y.numel()
        test_loss, test_acc = evaluate(model, test_loader, device)
        rec = dict(epoch=epoch, train_loss=loss_sum / n, train_acc=correct / n,
                   test_loss=test_loss, test_acc=test_acc, time=time.time() - t0)
        history.append(rec)
        print(f"epoch {epoch:2d} | train loss {rec['train_loss']:.4f} acc {rec['train_acc']:.4f} | "
              f"test loss {test_loss:.4f} acc {test_acc:.4f} | {rec['time']:.0f}s", flush=True)

    end_at = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"end {end_at}")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/ckpt_{args.model}.pt")  # trained weights
    with open(f"results/history_{args.model}.json", "w") as f:
        json.dump(dict(model=args.model, params=n_params, device=str(device), chip=chip, args=vars(args),
                       torch=torch.__version__, start=start_at, end=end_at, history=history), f, indent=2)


if __name__ == "__main__":
    main()
