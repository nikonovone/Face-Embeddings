from typing import Dict, Optional

import torch
from lightning import LightningModule
from pytorch_metric_learning import losses
from torch import Tensor, nn
from torch.optim import SGD, Adam, AdamW
from torchmetrics import MeanMetric
from transformers import AutoImageProcessor, AutoModel

from src.utils.config import OptimizerConfig, SchedulerConfig
from src.utils.metrics import get_metrics
from src.utils.schedulers import get_cosine_schedule_with_warmup


class FaceEmbeddingsModel(LightningModule):
    def __init__(
        self,
        image_size: int = 224,  # DINOv2 default image size
        optimizer_params: Optional[OptimizerConfig] = None,
        scheduler_params: Optional[SchedulerConfig] = None,
        margin: float = 0.5,
        embedding_size: int = 768,  # DINOv2 default embedding size
        model_name: str = "facebook/dinov2-base",  # or small, large, giant
        use_projection: bool = True,
    ):
        """
        Args:
            image_size: Input image size for DINOv2
            optimizer_params: Optimizer configuration
            scheduler_params: Learning rate scheduler configuration
            margin: Margin for triplet loss
            embedding_size: Size of the face embeddings
            model_name: DINOv2 model variant to use
            freeze_backbone: Whether to freeze the backbone model
            use_projection: Whether to use a projection head
        """
        super().__init__()

        # Initialize DINOv2 model
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Get the actual embedding size from the model
        self.backbone_embedding_size = self.model.config.hidden_size

        # Add projection head if needed
        self.use_projection = use_projection
        if use_projection:
            self.projection = nn.Sequential(
                nn.Linear(self.backbone_embedding_size, self.backbone_embedding_size),
                nn.ReLU(),
                nn.Linear(self.backbone_embedding_size, embedding_size),
                nn.LayerNorm(embedding_size),
            )

        # Loss function
        self.loss = losses.TripletMarginLoss(margin=margin, swap=True)

        # Save parameters
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        self.embedding_size = embedding_size
        self.image_size = image_size

        # Metrics
        self._train_loss = MeanMetric()
        self._valid_loss = MeanMetric()

        metrics = get_metrics()
        self._valid_metrics = metrics.clone(prefix="valid_")
        self._test_metrics = metrics.clone(prefix="test_")

        self.example_input_array = torch.randn(1, 3, image_size, image_size)
        self.save_hyperparameters(ignore=["backbone", "processor"])

    def forward(self, images: Tensor) -> Tensor:
        """Forward pass to get embeddings."""
        # Get embeddings from DINOv2
        outputs = self.model(images)
        embeddings = outputs.last_hidden_state[:, 0]  # CLS token

        # Apply projection if needed
        if self.use_projection:
            embeddings = self.projection(embeddings)

        # Normalize embeddings
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

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
        batch_size = len(anchors)
        labels = torch.cat(
            [
                torch.arange(
                    batch_size,
                    device=self.device,
                ),  # Anchor labels: 0,1,2,...
                torch.arange(
                    batch_size,
                    device=self.device,
                ),  # Positive labels: 0,1,2,... (same as anchors)
                torch.arange(
                    batch_size,
                    device=self.device,
                ),  # Negative labels: n,n+1,n+2,... (different)
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

        # Create labels for triplet mining - should match the same pattern as in training_step
        batch_size = len(anchors)
        labels = torch.cat(
            [
                torch.arange(
                    batch_size,
                    device=self.device,
                ),  # Anchor labels: 0,1,2,...
                torch.arange(
                    batch_size,
                    device=self.device,
                ),  # Positive labels: 0,1,2,... (same as anchors)
                torch.arange(
                    batch_size,
                    device=self.device,
                ),  # Negative labels: 0,1,2,... (same as anchors)
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
