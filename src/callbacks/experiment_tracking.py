# src/callbacks/experiment_tracking.py
from pathlib import Path
from typing import Dict, Optional, Union

from clearml import OutputModel, Task
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only

from src.utils.config import ExperimentConfig
from src.utils.logger import get_logger

logger = get_logger(__name__, rank_zero_only=True)


class ClearMLTracking(Callback):
    """
    Lightning callback for experiment tracking with ClearML.

    Features:
    - Automatic experiment tracking
    - Model checkpoint management
    - Configuration logging
    - Label enumeration support
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        label_enumeration: Optional[Dict[str, int]] = None,
        output_uri: Union[bool, str] = True,
        auto_connect_frameworks: bool = True,
    ) -> None:
        """
        Initialize ClearML tracking callback.

        Args:
            cfg: Experiment configuration
            label_enumeration: Optional mapping of labels to indices
            output_uri: ClearML output URI (True for default, str for custom URI)
            auto_connect_frameworks: Whether to automatically connect ML frameworks
        """
        super().__init__()
        self.cfg = cfg
        self.label_enumeration = label_enumeration
        self.output_uri = output_uri
        self.auto_connect_frameworks = auto_connect_frameworks

        self.task: Optional[Task] = None
        self.output_model: Optional[OutputModel] = None

    @rank_zero_only
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Setup ClearML task if not already initialized."""
        if self.task is None:
            self._initialize_task()

    @rank_zero_only
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Handle fit start event."""
        try:
            if self.task is None:
                self._initialize_task()

            # Log model hyperparameters
            if hasattr(pl_module, "hparams"):
                self.task.connect_configuration(
                    name="model_hyperparameters",
                    configuration=dict(pl_module.hparams),
                )
        except Exception as e:
            logger.error(f"Failed to initialize ClearML tracking: {str(e)}")
            raise

    @rank_zero_only
    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Handle test end event by uploading the best checkpoint."""
        try:
            checkpoint_path = self._get_best_checkpoint(trainer)
            if checkpoint_path and self.output_model is not None:
                logger.info(f'Uploading checkpoint "{checkpoint_path}" to ClearML')
                self.output_model.update_weights(
                    weights_filename=checkpoint_path,
                    auto_delete_file=True,
                    iteration=trainer.global_step,
                )
        except Exception as e:
            logger.error(f"Failed to upload checkpoint to ClearML: {str(e)}")

    def _initialize_task(self) -> None:
        """Initialize ClearML task and output model."""
        try:
            # Configure ClearML task
            Task.force_requirements_env_freeze()
            self.task = Task.init(
                project_name=self.cfg.project_name,
                task_name=self.cfg.experiment_name,
                output_uri=self.output_uri,
                auto_connect_frameworks=self.auto_connect_frameworks,
            )

            # Connect configuration
            self._connect_configuration()

            # Initialize output model
            self.output_model = OutputModel(
                task=self.task,
                label_enumeration=self.label_enumeration,
            )

            logger.info(
                f"Initialized ClearML tracking: "
                f"project='{self.cfg.project_name}', "
                f"experiment='{self.cfg.experiment_name}'",
            )
        except Exception as e:
            logger.error(f"Failed to initialize ClearML task: {str(e)}")
            raise

    def _connect_configuration(self) -> None:
        """Connect experiment configuration to ClearML."""
        if not self.task:
            return

        # Connect main configuration
        self.task.connect_configuration(
            name="experiment_config",
            configuration=self.cfg.model_dump(),
        )

        # Connect additional configurations if available
        if hasattr(self.cfg, "model_config"):
            self.task.connect_configuration(
                name="model_config",
                configuration=self.cfg.model_config.model_dump()
                if hasattr(self.cfg.model_config, "model_dump")
                else self.cfg.model_config,
            )

        if hasattr(self.cfg, "data_config"):
            self.task.connect_configuration(
                name="data_config",
                configuration=self.cfg.data_config.model_dump()
                if hasattr(self.cfg.data_config, "model_dump")
                else self.cfg.data_config,
            )

    def _get_best_checkpoint(self, trainer: Trainer) -> Optional[Path]:
        """Get the best checkpoint path from trainer."""
        checkpoint_callback = trainer.checkpoint_callback

        if isinstance(checkpoint_callback, ModelCheckpoint):
            if checkpoint_callback.best_model_path:
                best_path = Path(checkpoint_callback.best_model_path)
                if best_path.exists():
                    return best_path
                logger.warning(f"Best model path '{best_path}' does not exist")

        # Fallback to saving current model state
        fallback_path = Path(trainer.log_dir) / "last-checkpoint.ckpt"
        fallback_path.parent.mkdir(parents=True, exist_ok=True)

        trainer.save_checkpoint(str(fallback_path))
        logger.info(f"Saved fallback checkpoint to {fallback_path}")

        return fallback_path
