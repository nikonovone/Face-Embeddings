from typing import Dict, Optional

import torch
from ellzaf_ml.models import GhostFaceNetsV2
from lightning import LightningModule
from pytorch_metric_learning import losses
from torch import Tensor
from torch.optim import SGD, Adam, AdamW
from torchmetrics import MeanMetric

from src.utils.config import OptimizerConfig, SchedulerConfig
from src.utils.metrics import get_metrics
from src.utils.schedulers import get_cosine_schedule_with_warmup


class FaceEmbeddingsModel(LightningModule):
    def __init__(
        self,
        image_size: int = 512,
        width: float = 1.0,
        dropout: float = 0.0,
        optimizer_params: Optional[OptimizerConfig] = None,
        scheduler_params: Optional[SchedulerConfig] = None,
        margin: float = 0.5,
        embedding_size: int = 64,
        model_name: str = None,
    ):
        """
        Args:
            image_size: Input image size (default: 112 for GhostFaceNetsV2)
            width: Width multiplier for the model
            dropout: Dropout rate
            optimizer_params: Optimizer configuration
            scheduler_params: Learning rate scheduler configuration
            margin: Margin for triplet loss
        """
        super().__init__()

        # Initialize model
        self.model = GhostFaceNetsV2(
            image_size=image_size,
            width=width,
            dropout=dropout,
        )

        # Loss function
        self.loss = losses.TripletMarginLoss(margin=margin, swap=True)

        # Save parameters
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params

        # Metrics
        self._train_loss = MeanMetric()
        self._valid_loss = MeanMetric()

        metrics = get_metrics()
        self._valid_metrics = metrics.clone(prefix="valid_")
        self._test_metrics = metrics.clone(prefix="test_")

        self.save_hyperparameters(ignore=["model"])

    def forward(self, images: Tensor) -> Tensor:
        """Forward pass to get embeddings."""
        return self.model(images)

    def training_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Training step with triplet loss."""
        anchors = batch["anchor"]
        positives = batch["positive"]
        negatives = batch["negative"]

        # Get embeddings for all images
        anchor_embeddings = self(anchors)
        positive_embeddings = self(positives)
        negative_embeddings = self(negatives)

        # Concatenate all embeddings and create labels
        embeddings = torch.cat(
            [anchor_embeddings, positive_embeddings, negative_embeddings],
            dim=0,
        )

        # Create labels for triplet mining
        labels = torch.cat(
            [
                torch.arange(len(anchors), device=self.device),
                torch.arange(len(positives), device=self.device),
                torch.arange(len(negatives), device=self.device),
            ],
        )

        # Create indices for triplet selection
        indices_tuple = (
            torch.arange(len(anchors)),
            torch.arange(len(anchors), len(anchors) + len(positives)),
            torch.arange(len(anchors) + len(positives), len(embeddings)),
        )

        # Compute loss
        loss = self.loss(embeddings, labels, indices_tuple)

        # Log metrics
        self._train_loss(loss)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)

        return {"loss": loss}

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        """Validation step."""
        anchors = batch["anchor"]
        positives = batch["positive"]
        negatives = batch["negative"]

        # Get embeddings
        anchor_embeddings = self(anchors)
        positive_embeddings = self(positives)
        negative_embeddings = self(negatives)

        # Concatenate embeddings and create labels
        embeddings = torch.cat(
            [anchor_embeddings, positive_embeddings, negative_embeddings],
            dim=0,
        )

        labels = torch.cat(
            [
                torch.arange(len(anchors), device=self.device),
                torch.arange(len(positives), device=self.device),
                torch.arange(len(negatives), device=self.device),
            ],
        )

        # Create indices for triplet selection
        indices_tuple = (
            torch.arange(len(anchors)),
            torch.arange(len(anchors), len(anchors) + len(positives)),
            torch.arange(len(anchors) + len(positives), len(embeddings)),
        )

        # Compute validation loss
        loss = self.loss(embeddings, labels, indices_tuple)
        self._valid_loss(loss)

        # Compute similarity metrics for positive pairs
        positive_similarities = torch.nn.functional.cosine_similarity(
            anchor_embeddings,
            positive_embeddings,
            dim=1,
        )

        # Compute similarity metrics for negative pairs
        negative_similarities = torch.nn.functional.cosine_similarity(
            anchor_embeddings,
            negative_embeddings,
            dim=1,
        )

        # Создаем все сходства и соответствующие метки для EER
        all_similarities = torch.cat([positive_similarities, negative_similarities])
        all_labels = torch.cat(
            [
                torch.ones(len(positive_similarities), device=self.device),
                torch.zeros(len(negative_similarities), device=self.device),
            ],
        )

        # Проверяем, что у нас есть как положительные, так и отрицательные образцы
        if torch.sum(all_labels) > 0 and torch.sum(all_labels) < len(all_labels):
            # Update EER metric
            self._valid_metrics.update(all_similarities, all_labels)

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics."""
        self.log(
            "val_loss",
            self._valid_loss.compute(),
            prog_bar=True,
            on_epoch=True,
        )
        self._valid_loss.reset()

        metrics = self._valid_metrics.compute()
        for name, value in metrics.items():
            self.log(
                name,
                value,
                prog_bar=True,
                on_epoch=True,
            )
        self._valid_metrics.reset()

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        """Test step."""
        anchors = batch["anchor"]
        positives = batch["positive"]
        negatives = batch["negative"]

        # Get embeddings
        anchor_embeddings = self(anchors)
        positive_embeddings = self(positives)
        negative_embeddings = self(negatives)

        positive_similarities = torch.nn.functional.cosine_similarity(
            anchor_embeddings,
            positive_embeddings,
            dim=1,
        )

        # Compute similarity metrics for negative pairs
        negative_similarities = torch.nn.functional.cosine_similarity(
            anchor_embeddings,
            negative_embeddings,
            dim=1,
        )

        # Создаем все сходства и соответствующие метки для EER
        all_similarities = torch.cat([positive_similarities, negative_similarities])
        all_labels = torch.cat(
            [
                torch.ones(len(positive_similarities), device=self.device),
                torch.zeros(len(negative_similarities), device=self.device),
            ],
        )

        # Проверяем, что у нас есть как положительные, так и отрицательные образцы
        if torch.sum(all_labels) > 0 and torch.sum(all_labels) < len(all_labels):
            # Update EER metric
            self._test_metrics.update(all_similarities, all_labels)

    def on_test_epoch_end(self) -> None:
        """Log test metrics."""
        metrics = self._test_metrics.compute()
        for name, value in metrics.items():
            self.log(
                name,
                value,
                prog_bar=True,
                on_epoch=True,
            )
        self._test_metrics.reset()

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        opt_name = self.optimizer_params.name
        optimizer_params = self.optimizer_params.dict()
        optimizer_params.pop("name")

        # Get trainable parameters
        params = filter(lambda p: p.requires_grad, self.parameters())

        # Create optimizer
        optimizers = {
            "AdamW": AdamW,
            "Adam": Adam,
            "SGD": SGD,
        }

        if opt_name not in optimizers:
            raise ValueError(f'Unknown optimizer: "{opt_name}"')

        optimizer = optimizers[opt_name](params, **optimizer_params)

        # Create scheduler
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.scheduler_params.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_cycles=self.scheduler_params.num_cycles,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
