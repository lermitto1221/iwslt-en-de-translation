"""Batch or interactive translation using one verified inference session."""
import argparse
from pathlib import Path
import sys

from .inference import Translator


def main():
    parser = argparse.ArgumentParser()
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--checkpoint")
    model.add_argument("--bundle", help="portable directory created by mt.export")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--input", help="UTF-8 file, one source sentence per line")
    source.add_argument("--interactive", action="store_true", help="load once; type :quit to exit")
    parser.add_argument("--output", help="UTF-8 translation file (batch mode only)")
    parser.add_argument("--method", choices=("greedy", "beam"), default="beam")
    parser.add_argument("--protocol", help="saved validation-selected decoding protocol")
    parser.add_argument("--data-dir", help="original tokenizer directory, if a checkpoint was moved")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.interactive and args.output:
        parser.error("--output is only supported with --text or --input")
    if args.bundle and args.data_dir:
        parser.error("a bundle already includes its tokenizer; omit --data-dir")
    sys.stdout.reconfigure(encoding="utf-8")
    session = Translator(checkpoint=args.checkpoint, bundle=args.bundle,
                         protocol=args.protocol, device=args.device, data_dir=args.data_dir)

    def run(texts):
        values, flags = session.translate(texts, args.method, args.batch_size)
        if any(flags):
            print(f"{sum(flags)}/{len(texts)} outputs reached the generation token cap.", file=sys.stderr)
        return "\n".join(values) + "\n"

    if args.interactive:
        print("English -> German. Enter :quit to exit. Model loaded once.")
        while True:
            try:
                text = input("EN> ")
            except (EOFError, KeyboardInterrupt):
                break
            if text.strip() == ":quit":
                break
            if not text.strip():
                continue
            try:
                print("DE> " + run([text]), end="")
            except ValueError as error:
                print(str(error), file=sys.stderr)
    else:
        texts = [args.text] if args.text is not None else Path(args.input).read_text(encoding="utf-8-sig").splitlines()
        output = run(texts)
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(output, encoding="utf-8")
        else:
            print(output, end="")


if __name__ == "__main__":
    main()
