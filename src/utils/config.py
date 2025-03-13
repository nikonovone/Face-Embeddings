from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

T = TypeVar("T", bound="BaseConfig")


class OptimizerName(str, Enum):
    """Supported optimizer names"""

    SGD = "SGD"
    ADAM = "Adam"
    ADAMW = "AdamW"
    RMSPROP = "RMSprop"


class BaseConfig(BaseModel):
    """Base configuration class with common functionality"""

    model_config = ConfigDict(
        extra="forbid",  # Prevent extra fields
        frozen=True,  # Make configs immutable
        validate_assignment=True,  # Validate on attribute assignment
    )

    @classmethod
    @lru_cache(maxsize=32)
    def from_yaml(cls: Type[T], path: Union[str, Path]) -> T:
        """Load configuration from YAML file with caching"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        try:
            cfg = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
            return cls(**cast(Dict[str, Any], cfg))
        except Exception as e:
            raise ValueError(f"Failed to load config from {path}: {str(e)}")

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            yaml.safe_dump(
                self.model_dump(exclude_none=True),
                f,
                default_flow_style=False,
                sort_keys=False,
            )


class DataConfig(BaseConfig):
    """Data loading and processing configuration"""

    # Constants
    VALID_IMG_SIZES: ClassVar[Tuple[Tuple[int, int], ...]] = (
        (224, 224),
        (256, 256),
        (299, 299),
        (384, 384),
    )

    # Fields
    dataset_dir: Path = Field(
        default=Path("../data"),
        description="Directory containing the dataset",
    )
    train_dir: Path = Field(
        default=Path("train"),
        description="Directory containing training data",
    )
    val_dir: Path = Field(
        default=Path("val"),
        description="Directory containing validation data",
    )
    test_dir: Path = Field(
        default=Path("test"),
        description="Directory containing test data",
    )
    img_size: Tuple[int, int] = Field(
        default=(224, 224),
        description="Input image dimensions (height, width)",
    )
    batch_size: int = Field(default=32, ge=1, description="Number of samples per batch")
    data_split: Tuple[float, float, float] = Field(
        default=(0.7, 0.2, 0.1),
        description="Train/val/test split ratios",
    )
    num_workers: int = Field(
        default=0,
        ge=0,
        description="Number of data loading workers",
    )
    pin_memory: bool = Field(
        default=True,
        description="Pin memory for faster GPU transfer",
    )
    seed: int = Field(default=13, description="Random seed for reproducibility")

    @field_validator("img_size")
    @classmethod
    def validate_img_size(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        if v not in cls.VALID_IMG_SIZES:
            raise ValueError(
                f"img_size {v} not in supported sizes: {cls.VALID_IMG_SIZES}",
            )
        return v


class TrainerConfig(BaseConfig):
    """Training process configuration"""

    min_epochs: int = Field(
        default=7,
        ge=1,
        description="Minimum number of epochs to train",
    )
    max_epochs: int = Field(
        default=20,
        ge=1,
        description="Maximum number of epochs to train",
    )
    check_val_every_n_epoch: int = Field(
        default=3,
        ge=1,
        description="Validation frequency in epochs",
    )
    log_every_n_steps: int = Field(
        default=50,
        ge=1,
        description="Logging frequency in steps",
    )
    devices: int = Field(default=2, ge=1, description="Number of devices to use")
    accelerator: Literal["cpu", "gpu", "tpu", "ipu"] = Field(
        default="gpu",
        description="Hardware accelerator type",
    )
    gradient_clip_val: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Gradient clipping value",
    )
    gradient_clip_algorithm: Optional[Literal["norm", "value"]] = Field(
        default=None,
        description="Gradient clipping algorithm",
    )
    accumulate_grad_batches: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of batches to accumulate gradients",
    )
    deterministic: bool = Field(
        default=False,
        description="Enable deterministic training",
    )
    fast_dev_run: bool = Field(default=False, description="Run quick dev iteration")
    default_root_dir: Optional[Path] = Field(
        default=None,
        description="Root directory for logs/checkpoints",
    )

    @model_validator(mode="after")
    def validate_epochs(self) -> "TrainerConfig":
        if self.max_epochs < self.min_epochs:
            raise ValueError(
                f"max_epochs ({self.max_epochs}) must be >= "
                f"min_epochs ({self.min_epochs})",
            )
        return self


class OptimizerConfig(BaseConfig):
    """Optimization algorithm configuration"""

    name: OptimizerName = Field(
        default=OptimizerName.SGD,
        description="Optimizer algorithm name",
    )
    lr: float = Field(default=2e-3, gt=0.0, description="Learning rate")
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        description="Weight decay coefficient",
    )


class SchedulerConfig(BaseConfig):
    """Learning rate scheduler configuration"""

    warmup_steps: int = Field(default=200, ge=0, description="Number of warmup steps")
    num_cycles: int = Field(
        default=2,
        ge=1,
        description="Number of cycles for cyclic schedulers",
    )


class ModelArchitecture(str, Enum):
    """Supported model architectures"""

    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    RESNET50 = "resnet50"
    EFFICIENTNET = "efficientnet"
    GHOSTFACENETV2 = "GhostFaceNetsV2"
    DINOV2BASE = "facebook/dinov2-base"

    def __str__(self):
        return self.value


class ModelConfig(BaseConfig):
    """Learning rate scheduler configuration"""

    name_model: ModelArchitecture = Field(
        default=ModelArchitecture.GHOSTFACENETV2,
        description="Model architecture name",
    )
    unfreeze_epoch: int = Field(default=0, description="")
    pretrained: bool = Field(default=False, description="")
    embedding_size: int = Field(default=512, description="Embedding size")


class ExperimentConfig(BaseConfig):
    """Top-level experiment configuration"""

    project_name: str = Field(
        default="default_project_name",
        min_length=1,
        description="Project name for logging",
    )
    experiment_name: str = Field(
        default="default_experiment_name",
        min_length=1,
        description="Experiment name for logging",
    )
    track_in_clearml: bool = Field(default=True, description="Enable ClearML tracking")
    run_test: bool = Field(default=False, description="Running test stage")

    optimizer_config: OptimizerConfig = Field(
        default_factory=OptimizerConfig,
        description="Optimizer configuration",
    )
    model_params_config: ModelConfig = Field(
        default_factory=ModelConfig,
        description="Model configuration",
    )
    scheduler_config: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="Scheduler configuration",
    )
    trainer_config: TrainerConfig = Field(
        default_factory=TrainerConfig,
        description="Trainer configuration",
    )
    data_config: DataConfig = Field(
        default_factory=DataConfig,
        description="Data configuration",
    )
