from pathlib import Path

import click
import numpy as np
import onnxruntime
import torch

from src.models.face_module import FaceEmbeddingsModel
from src.utils.logger import get_logger

logger = get_logger()


@click.command()
@click.option(
    "--checkpoint",
    required=True,
    type=click.Path(exists=True),
    help="Path to the PyTorch model checkpoint",
)
@click.option(
    "--output",
    default="model.onnx",
    type=click.Path(),
    help="Path to save the ONNX model",
)
@click.option("--image-size", default=224, type=int, help="Input image size (square)")
@click.option(
    "--test-inference",
    is_flag=True,
    help="Test the exported ONNX model with random input",
)
def main(checkpoint, output, image_size, test_inference):
    """Convert PyTorch model to ONNX format."""
    # Create output directory if it doesn't exist
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load PyTorch model
    logger.info(f"Loading model from {checkpoint}")
    model = FaceEmbeddingsModel.load_from_checkpoint(checkpoint)
    model.eval()

    # Set example input array for the model
    model.example_input_array = torch.randn(1, 3, image_size, image_size)

    # Export to ONNX
    logger.info(f"Exporting model to {output}")
    model.to_onnx(
        output,
        export_params=True,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    logger.info(f"Model successfully exported to {output}")

    # Test inference with ONNX Runtime
    if test_inference:
        logger.info("Testing inference with ONNX Runtime...")

        # Create ONNX Runtime session
        ort_session = onnxruntime.InferenceSession(str(output_path))

        # Get input name
        input_name = ort_session.get_inputs()[0].name

        # Create random input
        random_input = np.random.randn(1, 3, image_size, image_size).astype(np.float32)

        # Run inference
        ort_inputs = {input_name: random_input}
        ort_outs = ort_session.run(None, ort_inputs)

        logger.info(f"Inference successful! Output shape: {ort_outs[0].shape}")

        # Performance testing
        logger.info("Running performance tests...")

        # Parameters for performance testing
        num_iterations = 100
        batch_size = 64

        # Prepare batch input
        batch_input = np.random.randn(batch_size, 3, image_size, image_size).astype(
            np.float32,
        )
        ort_inputs = {input_name: batch_input}

        # Warmup runs
        for _ in range(10):
            ort_session.run(None, ort_inputs)

        # Measure performance
        import time

        start_time = time.time()

        for _ in range(num_iterations):
            ort_session.run(None, ort_inputs)

        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        avg_time_per_inference = total_time / num_iterations * 1000  # in milliseconds
        throughput = num_iterations * batch_size / total_time  # images per second

        logger.info("Performance results:")
        logger.info(
            f"  Total time for {num_iterations} iterations: {total_time:.4f} seconds",
        )
        logger.info(f"  Average inference time: {avg_time_per_inference:.4f} ms")
        logger.info(f"  Throughput: {throughput:.2f} images/second")


if __name__ == "__main__":
    main()
