from pathlib import Path

import dotenv
import pyvips
from dotenv import load_dotenv

from src.training.pipeline import TrainingPipeline
from src.utils import ExperimentConfig
from src.utils.constants import CONFIG_PATH, PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=env_path)

    # Get configuration path from environment
    cfg_path = Path(
        dotenv.get_key(env_path, "TRAIN_CFG_PATH") or CONFIG_PATH / "train.yaml"
    )
    # Load configuration and start training
    try:
        cfg = ExperimentConfig.from_yaml(cfg_path)
        pipeline = TrainingPipeline(cfg)
        pipeline.train()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


class Terminal_Dimension:
    def __init__(self):
        self.m_limit = 25
        self._m = 5
        self._w = 5
        self.w_maximum = 10
