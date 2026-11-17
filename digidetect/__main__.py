"""Usage:
    python -m digidetect                  start the website at http://localhost:8000
    python -m digidetect --port 9000 --no-browser
    python -m digidetect train            train in the terminal and print progress
        --hidden 64 --epochs 4 --lr 0.08 --full --save model.npz
"""
import argparse


def main():
    ap = argparse.ArgumentParser(prog="digidetect", description="A neural network from scratch that reads handwritten digits.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab automatically")
    sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("train", help="train in the terminal instead of the website")
    t.add_argument("--hidden", type=int, default=64, help="hidden units (try 32, 64, 128)")
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--lr", type=float, default=0.08)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--full", action="store_true", help="use the full 60k/10k MNIST (needs internet the first time)")
    t.add_argument("--save", metavar="FILE.npz", help="save the trained weights")
    args = ap.parse_args()

    if args.cmd == "train":
        from . import data
        from .trainer import Trainer
        train, test = data.load_full_mnist() if args.full else data.load_bundled()
        print(f"{train.n:,} training / {test.n:,} test images · {args.hidden} hidden units · {args.epochs} epochs")
        tr = Trainer(train, test, args.hidden, args.epochs, args.lr, args.seed)
        acc = tr.run()
        print(f"final test accuracy: {acc * 100:.2f}%")
        if args.save:
            tr.net.save(args.save)
            print(f"saved weights to {args.save}")
    else:
        from .server import serve
        serve(args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
