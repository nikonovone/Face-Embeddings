from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.utils import DataConfig

from .dataset import TripletDataset
from .transform import get_train_transforms, get_valid_transforms


class DefaultDataModule(LightningDataModule):
    def __init__(self, cfg: DataConfig):
        super().__init__()
        self.cfg = cfg
        self._train_transforms = get_train_transforms(*cfg.img_size)
        self._valid_transforms = get_valid_transforms(*cfg.img_size)

        # Пути к данным
        self.train_dir = self.cfg.dataset_dir / self.cfg.train_dir
        self.val_dir = self.cfg.dataset_dir / self.cfg.val_dir
        self.test_dir = self.cfg.dataset_dir / self.cfg.test_dir

    def prepare_data(self):
        # Проверка наличия данных
        if not self.train_dir.exists():
            raise FileNotFoundError(
                f"Директория с обучающими данными не найдена: {self.train_dir}",
            )

        if self.val_dir and not self.val_dir.exists():
            raise FileNotFoundError(
                f"Директория с валидационными данными не найдена: {self.val_dir}",
            )

        if self.test_dir and not self.test_dir.exists():
            raise FileNotFoundError(
                f"Директория с тестовыми данными не найдена: {self.test_dir}",
            )

    def setup(self, stage: Optional[str] = None) -> None:
        """Настройка датасетов в зависимости от текущего этапа (train/validate/test)."""
        if stage == "fit" or stage is None:
            # Создаем обучающий датасет
            self.train_dataset = TripletDataset(
                data_dir=self.train_dir,
                transforms=self._train_transforms,
                triplet_cache_size=self.cfg.triplet_cache_size
                if hasattr(self.cfg, "triplet_cache_size")
                else None,
            )

            # Создаем валидационный датасет
            if self.val_dir:
                self.val_dataset = TripletDataset(
                    data_dir=self.val_dir,
                    transforms=self._valid_transforms,
                    triplet_cache_size=self.cfg.val_triplet_cache_size
                    if hasattr(self.cfg, "val_triplet_cache_size")
                    else None,
                )

        if stage == "test" or stage is None:
            # Создаем тестовый датасет
            if self.test_dir:
                self.test_dataset = TripletDataset(
                    data_dir=self.test_dir,
                    transforms=self._valid_transforms,
                    triplet_cache_size=self.cfg.test_triplet_cache_size
                    if hasattr(self.cfg, "test_triplet_cache_size")
                    else None,
                )

    def train_dataloader(self) -> DataLoader:
        """Возвращает DataLoader для обучающего датасета."""
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Возвращает DataLoader для валидационного датасета."""
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Возвращает DataLoader для тестового датасета."""
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
            shuffle=False,
        )
