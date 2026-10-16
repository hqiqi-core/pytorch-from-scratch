"""

"""
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from retrieval_metrics import recall_at_k
from simple_clip_train import encode_dataset, resolve_device,create_toy_dataset,ImageTextDataset,evaluate_alignment
from simple_clip_checkpoint import load_checkpoint
import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint-path",
        type=str,
        default="data/simple_clip/checkpoints/model.pt",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="data/simple_clip",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 3, 5],
    )
    return p.parse_args()




def main():
    args = parse_args()
    device = resolve_device(args.device)

    model, tokenizer = load_checkpoint(
        Path(args.checkpoint_path),
        device,
    )

    samples = create_toy_dataset(Path(args.data_dir))

    dataset = ImageTextDataset(
        samples,
        tokenizer,
        image_size=64,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    with torch.no_grad():
        image_features, text_features = encode_dataset(
            model, loader, device
        )

        similarities = image_features @ text_features.T
        print(f"Checkpoint: {args.checkpoint_path}")
        print(f"Device: {device}")
        print(f"Number of pairs: {len(dataset)}")
        print(f"Similarity shape: {tuple(similarities.shape)}")

        targets = torch.arange(
            similarities.shape[0],
            device=similarities.device,
        )

        for k in args.ks:
            i2t_recall = recall_at_k(
                similarities,
                targets,
                k,
            )
            t2i_recall = recall_at_k(
                similarities.T,
                targets,
                k,
            )

            print(
                f"R@{k}: "
                f"i2t={i2t_recall:.4f}, "
                f"t2i={t2i_recall:.4f}"
            )

    # metrics = evaluate_alignment(model, loader, device)

    # print(f"Checkpoint: {args.checkpoint_path}")
    # print(f"Device: {device}")
    # print(f"Number of pairs: {len(dataset)}")
    # print(f"Vocabulary size: {tokenizer.vocab_size}")
    # print(
    #     "Image-to-text accuracy: "
    #     f"{metrics['image_to_text_accuracy']:.4f}"
    # )
    # print(
    #     "Text-to-image accuracy: "
    #     f"{metrics['text_to_image_accuracy']:.4f}"
    # )


if __name__ == "__main__":
    main()