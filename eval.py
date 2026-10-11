"""Load saved weights and re-evaluate them on the CIFAR-10 test set.

Usage:
    python eval.py --model resnet
    python eval.py --model resnet_attn
"""
import argparse

import torch
import torchvision
import torchvision.transforms as T

from train import ResNet, ResNetAttn, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["resnet", "resnet_attn"], default="resnet")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    model = (ResNet() if args.model == "resnet" else ResNetAttn()).to(device)
    model.load_state_dict(torch.load(f"checkpoints/ckpt_{args.model}.pt", map_location=device))

    tf = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    test = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=tf)
    loader = torch.utils.data.DataLoader(test, 500, shuffle=False)

    loss, acc = evaluate(model, loader, device)
    print(f"{args.model}: test loss {loss:.4f}, test acc {acc * 100:.2f}% on {len(test)} images")


if __name__ == "__main__":
    main()
