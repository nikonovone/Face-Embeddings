from pathlib import Path

from dotenv import get_key, load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Set project root directory
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(get_key(".env", "PROJ_ROOT") or _DEFAULT_PROJECT_ROOT)

# Create and export common project paths
CONFIG_PATH = PROJECT_ROOT / "configs"
DATA_PATH = PROJECT_ROOT / "data"
LOGS_PATH = PROJECT_ROOT / "logs"
CHECKPOINTS_PATH = PROJECT_ROOT / "checkpoints"

# Create basic directories
for path in [CONFIG_PATH, DATA_PATH, LOGS_PATH, CHECKPOINTS_PATH]:
    path.mkdir(exist_ok=True, parents=True)
