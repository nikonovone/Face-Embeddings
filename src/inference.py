from pathlib import Path
from typing import List

import click
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.transform import get_valid_transforms
from src.utils.logger import get_logger

logger = get_logger()


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.pair_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        self.pair_ids = [pair_dir.name for pair_dir in self.pair_dirs]

    def __len__(self):
        return len(self.pair_dirs)

    def __getitem__(self, idx):
        pair_dir = self.pair_dirs[idx]
        images = sorted(list(pair_dir.glob("*.jpg")))

        if len(images) != 2:
            raise ValueError(f"Expected 2 images in {pair_dir}, found {len(images)}")

        img1 = np.array(Image.open(images[0]).convert("RGB"))
        img2 = np.array(Image.open(images[1]).convert("RGB"))

        if self.transform:
            img1 = self.transform(image=img1)["image"].float()
            img2 = self.transform(image=img2)["image"].float()

        return img1, img2, self.pair_ids[idx]


def create_sample_sub(pair_ids: List[str], sim_scores: List[float]):
    sub_sim_column = "similarity"
    id_column = "pair_id"
    return pd.DataFrame({id_column: pair_ids, sub_sim_column: sim_scores})


@click.command()
@click.option(
    "--test-path",
    type=click.Path(exists=True),
    default="./data/original/test_public/images",
    help="Path to test images directory",
)
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    default="./checkpoints/model.onnx",
    help="Path to ONNX model file",
)
@click.option(
    "--output-path",
    type=click.Path(),
    default="./submission.csv",
    help="Path to save submission file",
)
@click.option("--batch-size", type=int, default=32, help="Batch size for inference")
@click.option(
    "--num-workers",
    type=int,
    default=4,
    help="Number of workers for data loading",
)
@click.option("--img-size", type=int, default=224, help="Image size for model input")
def main(test_path, model_path, output_path, batch_size, num_workers, img_size):
    """Run inference using ONNX model and create submission file."""
    # Check GPU availability
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")

    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Initialize ONNX Runtime session with GPU providers
    logger.info(f"Loading ONNX model from {model_path}")
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if torch.cuda.is_available()
        else ["CPUExecutionProvider"]
    )
    logger.info(f"Using providers: {providers}")
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name

    # Prepare dataset and dataloader
    transform = get_valid_transforms(img_size, img_size)
    test_dataset = TestDataset(test_path, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Run inference
    sim_scores = []
    pair_ids = []

    logger.info("Starting inference...")
    for img1_batch, img2_batch, batch_pair_ids in tqdm(
        test_loader,
        desc="Processing pairs",
    ):
        # Convert to numpy for ONNX Runtime
        img1_np = img1_batch.numpy()
        img2_np = img2_batch.numpy()

        # Get embeddings using ONNX Runtime
        emb1 = session.run(None, {input_name: img1_np})[0]
        emb2 = session.run(None, {input_name: img2_np})[0]

        # Convert to torch tensors for cosine similarity
        emb1_tensor = torch.from_numpy(emb1)
        emb2_tensor = torch.from_numpy(emb2)

        # Compute cosine similarity
        batch_similarities = torch.nn.functional.cosine_similarity(
            emb1_tensor,
            emb2_tensor,
        ).numpy()

        sim_scores.extend(batch_similarities.tolist())
        pair_ids.extend(batch_pair_ids)

    # Create and save submission file
    sub_df = create_sample_sub(pair_ids, sim_scores)
    sub_df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path} with {len(sub_df)} entries")


if __name__ == "__main__":
    main()
