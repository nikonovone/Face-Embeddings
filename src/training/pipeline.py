from typing import List, Optional

import lightning
from lightning import Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from torch import set_float32_matmul_precision

from src.callbacks.debug import LogModelSummary, VisualizeBatch
from src.callbacks.experiment_tracking import ClearMLTracking
from src.callbacks.freeze import FeatureExtractorFreezeUnfreeze
from src.data.datamodule import DefaultDataModule
from src.models import FaceEmbeddingsModel
from src.utils import ExperimentConfig
from src.utils.constants import CHECKPOINTS_PATH, LOGS_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingPipeline:
    """Training pipeline for FaceEmbeddingsModel with modular setup and logging."""

    def __init__(self, cfg: ExperimentConfig):
        """
        Initialize the training pipeline with experiment configuration.

        Args:
            cfg: Experiment configuration object containing model, data, and trainer settings.
        """
        self.cfg = cfg
        self._setup_environment()
        self.logger = self._initialize_logger()
        self.callbacks = self._initialize_callbacks()
        self.model = self._initialize_model()
        self.datamodule = self._initialize_datamodule()
        self.trainer = self._initialize_trainer()

    def _setup_environment(self) -> None:
        """Configure the training environment, including seeds and precision."""
        lightning.seed_everything(self.cfg.data_config.seed, workers=True)
        set_float32_matmul_precision("medium")

    def _initialize_logger(self) -> pl_loggers.TensorBoardLogger:
        """Initialize TensorBoard logger with experiment-specific log directory."""
        return pl_loggers.TensorBoardLogger(save_dir=str(LOGS_PATH), name="training")

    def _initialize_callbacks(self) -> List[Callback]:
        """Initialize and return a list of training callbacks."""
        callbacks: List[Callback] = [
            LogModelSummary(),
            RichProgressBar(),
            VisualizeBatch(every_n_epochs=5),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=str(CHECKPOINTS_PATH),
                filename="{epoch}-{valid_eer:.4f}",
                save_top_k=3,
                monitor="valid_eer",
                mode="min",
                every_n_epochs=1,
                save_last=True,
                save_weights_only=False,  # Ensure full model state is saved
            ),
            FeatureExtractorFreezeUnfreeze(unfreeze_at_epoch=0),
        ]

        if self.cfg.track_in_clearml:
            callbacks.append(ClearMLTracking(self.cfg))

        return callbacks

    def _initialize_model(self) -> FaceEmbeddingsModel:
        """Initialize the FaceEmbeddingsModel with configuration."""
        return FaceEmbeddingsModel(
            image_size=self.cfg.data_config.img_size[0],
            embedding_size=self.cfg.model_params_config.embedding_size,
            optimizer_params=self.cfg.optimizer_config,
            scheduler_params=self.cfg.scheduler_config,
        )

    def _initialize_datamodule(self) -> DefaultDataModule:
        """Initialize the data module with configuration."""
        return DefaultDataModule(cfg=self.cfg.data_config)

    def _initialize_trainer(self) -> Trainer:
        """Initialize the Lightning Trainer with configuration and callbacks."""
        trainer_config = dict(self.cfg.trainer_config)  # Avoid modifying original dict
        trainer_config.update(
            {
                "callbacks": self.callbacks,
                "logger": self.logger,
            },
        )
        return Trainer(**trainer_config)

    def train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        """
        Execute the training process.

        Args:
            resume_from_checkpoint: Path to checkpoint for resuming training, if any.
        """
        try:
            self.trainer.fit(
                model=self.model,
                datamodule=self.datamodule,
                ckpt_path=resume_from_checkpoint,
            )
            best_model_path = self.trainer.checkpoint_callback.best_model_path
            logger.info(
                f"Training completed. Best model saved at: {best_model_path}",
            )

            if self.cfg.run_test:
                self.test(ckpt_path=best_model_path)

        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise

    def test(self, ckpt_path: str = "best") -> None:
        """
        Execute the testing process.

        Args:
            ckpt_path: Path to checkpoint for testing (default: 'best').
        """
        try:
            self.trainer.test(
                model=self.model,
                datamodule=self.datamodule,
                ckpt_path=ckpt_path,
            )
            logger.info(f"Testing completed using checkpoint: {ckpt_path}")
        except Exception as e:
            logger.error(f"Testing failed: {str(e)}", exc_info=True)
            raise
