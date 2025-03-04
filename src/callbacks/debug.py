# src/callbacks/debug.py
from typing import Tuple

import cv2
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_only
from torchvision.utils import make_grid

from src.models import FaceEmbeddingsModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ImageVisualizer:
    """Helper class for image visualization tasks."""

    @staticmethod
    def add_label(
        img: torch.Tensor,
        label: str,
        position: Tuple[int, int] = (50, 50),
        color: Tuple[int, int, int] = (0, 255, 0),
        scale: float = 2.0,
        thickness: int = 2,
    ) -> torch.Tensor:
        """
        Add text label to an image.
        """
        # Проверка на None
        if img is None:
            raise ValueError("Input image is None")

        # Убедимся, что тензор на CPU и отсоединен от графа вычислений
        img_cpu = img.detach().cpu()

        # Проверка на NaN и Inf
        if torch.isnan(img_cpu).any() or torch.isinf(img_cpu).any():
            # Заменяем NaN и Inf на безопасные значения
            img_cpu = torch.nan_to_num(img_cpu, nan=0.0, posinf=1.0, neginf=0.0)

        # Нормализация в диапазон [0, 1]
        if img_cpu.max() > 1.0:
            img_cpu = img_cpu / 255.0

        # Преобразование в numpy массив для OpenCV
        img_np = img_cpu.permute(1, 2, 0).numpy()

        # Масштабирование в диапазон [0, 255]
        img_array = (img_np * 255).astype(np.uint8)

        # Проверка формы массива
        if len(img_array.shape) == 2 or img_array.shape[2] == 1:
            # Для одноканальных изображений
            if len(img_array.shape) == 3 and img_array.shape[2] == 1:
                img_array = img_array[:, :, 0]
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        # Проверка корректности массива перед вызовом putText
        if not img_array.flags["C_CONTIGUOUS"]:
            img_array = np.ascontiguousarray(img_array)

        # Добавление текста с помощью OpenCV
        cv2.putText(
            img_array,
            label,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
        )

        # Преобразование обратно в тензор PyTorch
        if img_cpu.shape[0] == 1:  # Если оригинал был одноканальным
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            img_array = np.expand_dims(img_array, axis=2)

        result = torch.from_numpy(img_array.transpose(2, 0, 1)).float() / 255.0

        # Возвращаем тензор того же типа, что и входной
        return result.to(dtype=img.dtype)

    @staticmethod
    def create_triplet_grid(
        triplets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        max_samples: int = 8,
    ) -> torch.Tensor:
        """
        Create a visualization grid for triplet samples.

        Args:
            triplets: Tuple of (anchor, positive, negative) batches
            max_samples: Maximum number of triplets to visualize

        Returns:
            Grid of visualized triplets
        """
        anchor = triplets["anchor"]
        positive = triplets["positive"]
        negative = triplets["negative"]

        batch_size = len(anchor)
        num_samples = min(batch_size, max_samples)

        # Select subset of samples
        indices = torch.randperm(batch_size)[:num_samples]
        visualizations = []

        for idx in indices:
            for img, label in zip(
                [anchor[idx], positive[idx], negative[idx]],
                ["Anchor", "Positive", "Negative"],
            ):
                labeled_img = ImageVisualizer.add_label(img, label)
                visualizations.append(labeled_img)

        return make_grid(visualizations, nrow=3, normalize=True, pad_value=1)


class LogModelSummary(Callback):
    """Callback to log model summary at the start of training."""

    def __init__(self, max_depth: int = 3):
        super().__init__()
        self.max_depth = max_depth

    @rank_zero_only
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log model summary when training starts."""
        try:
            # Get sample input
            batch = next(iter(trainer.train_dataloader))
            images = batch["anchor"].to(pl_module.device)

            from torchinfo import summary as ti_summary

            model_summary = ti_summary(
                pl_module.model,
                input_size=images.shape,
                col_names=["input_size", "output_size", "num_params"],
                depth=self.max_depth,
                verbose=0,
            )
            logger.info(f"Model Summary:\n{model_summary}")

        except Exception as e:
            logger.warning(f"Failed to log model summary: {str(e)}")
            import traceback

            logger.debug(f"Detailed error: {traceback.format_exc()}")


class VisualizeBatch(Callback):
    """Callback to periodically visualize training batches."""

    def __init__(
        self,
        every_n_epochs: int = 1,
        max_samples: int = 8,
        log_key: str = "batch_preview",
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.max_samples = max_samples
        self.log_key = log_key
        self.visualizer = ImageVisualizer()

    @rank_zero_only
    def on_train_epoch_start(
        self,
        trainer: Trainer,
        pl_module: FaceEmbeddingsModel,
    ) -> None:
        """Visualize batch at start of selected epochs."""
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        try:
            # Get random batch
            batch = next(iter(trainer.train_dataloader))

            # Create visualization grid
            grid = self.visualizer.create_triplet_grid(
                batch,
                max_samples=self.max_samples,
            )

            # Log to tensorboard
            trainer.logger.experiment.add_image(
                self.log_key,
                img_tensor=grid,
                global_step=trainer.global_step,
            )
        except Exception as e:
            logger.warning(f"Failed to visualize batch: {str(e)}")
