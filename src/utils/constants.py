from pathlib import Path

from dotenv import get_key, load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Set project root directory
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(
    get_key(_DEFAULT_PROJECT_ROOT / ".env", "PROJECT_ROOT") or _DEFAULT_PROJECT_ROOT,
)

# Create and export common project paths
CONFIG_PATH = PROJECT_ROOT / "configs"
DATA_PATH = PROJECT_ROOT / "data"
LOGS_PATH = PROJECT_ROOT / "logs"
CHECKPOINTS_PATH = PROJECT_ROOT / "checkpoints"

# Create basic directories
for path in [CONFIG_PATH, DATA_PATH, LOGS_PATH, CHECKPOINTS_PATH]:
    path.mkdir(exist_ok=True, parents=True)

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
