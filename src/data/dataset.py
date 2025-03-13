import json
import random
from pathlib import Path
from typing import Tuple

import albumentations as albu
from numpy import uint8
from numpy.typing import NDArray

# Import vips configuration
from src.utils.vips_config import configure_vips

# Configure VIPS before importing pyvips
if configure_vips():
    import pyvips
else:
    raise ImportError("Failed to configure and import pyvips")

from typing import Dict

from torch import Tensor

# Import vips configuration
from src.utils.vips_config import configure_vips

# Configure VIPS before importing pyvips
if configure_vips():
    import pyvips
else:
    raise ImportError("Failed to configure and import pyvips")


class TripletDataset:
    def __init__(
        self,
        data_dir: Path,
        transforms: albu.Compose,
        meta_filename: str = "meta.json",
        images_dir: str = "images",
        triplet_cache_size: int = 10000,
    ):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / images_dir
        self.transforms = transforms

        # Load metadata
        with open(self.data_dir / meta_filename, "r") as f:
            self.metadata = json.load(f)

        # Group images by person and type (real/synthetic)
        self.person_to_real_images = {}
        self.person_to_synthetic_images = {}

        for img_path, is_synthetic in self.metadata.items():
            # Extract person ID from path
            person_id = img_path.split("/")[0]

            # Initialize lists if needed
            if person_id not in self.person_to_real_images:
                self.person_to_real_images[person_id] = []
                self.person_to_synthetic_images[person_id] = []

            # Add image to appropriate list based on type
            if is_synthetic == 0:  # Real image
                self.person_to_real_images[person_id].append(img_path)
            else:  # Synthetic image
                self.person_to_synthetic_images[person_id].append(img_path)

        # Find valid persons (those with real images)
        self.valid_persons = [
            p
            for p in self.person_to_real_images.keys()
            if len(self.person_to_real_images[p]) > 0
        ]

        if not self.valid_persons:
            raise ValueError("No people with real images in the dataset")

        # Triplet cache for faster operation
        self.triplet_cache = []
        self.triplet_cache_size = triplet_cache_size
        if triplet_cache_size:
            self._fill_triplet_cache()

    def _fill_triplet_cache(self):
        """Fill the triplet cache with random combinations"""
        self.triplet_cache = []
        for _ in range(self.triplet_cache_size):
            # Select a random person with real images
            person_id = random.choice(self.valid_persons)

            # Anchor - always a real image
            anchor_path = random.choice(self.person_to_real_images[person_id])

            # Choose strategy for positive and negative examples
            strategy = random.choice(
                ["same_person_real", "diff_person_real", "same_person_synthetic"],
            )

            if strategy == "same_person_real":
                # Positive - another real image of the same person
                real_images = [
                    img
                    for img in self.person_to_real_images[person_id]
                    if img != anchor_path
                ]
                positive_path = (
                    random.choice(real_images) if real_images else anchor_path
                )

                # Negative - real image of another person
                other_persons = [p for p in self.valid_persons if p != person_id]
                if other_persons:
                    other_person = random.choice(other_persons)
                    negative_path = random.choice(
                        self.person_to_real_images[other_person],
                    )
                else:
                    negative_path = (
                        random.choice(self.person_to_synthetic_images[person_id])
                        if self.person_to_synthetic_images[person_id]
                        else anchor_path
                    )

            elif strategy == "diff_person_real":
                # Positive - real image of the same person
                real_images = [
                    img
                    for img in self.person_to_real_images[person_id]
                    if img != anchor_path
                ]
                positive_path = (
                    random.choice(real_images) if real_images else anchor_path
                )

                # Negative - real image of another person
                other_persons = [p for p in self.valid_persons if p != person_id]
                if other_persons:
                    other_person = random.choice(other_persons)
                    negative_path = random.choice(
                        self.person_to_real_images[other_person],
                    )
                else:
                    negative_path = (
                        random.choice(self.person_to_synthetic_images[person_id])
                        if self.person_to_synthetic_images[person_id]
                        else anchor_path
                    )

            elif strategy == "same_person_synthetic":
                # Positive - real image of the same person
                real_images = [
                    img
                    for img in self.person_to_real_images[person_id]
                    if img != anchor_path
                ]
                positive_path = (
                    random.choice(real_images) if real_images else anchor_path
                )

                # Negative - synthetic image of the same person
                if self.person_to_synthetic_images[person_id]:
                    negative_path = random.choice(
                        self.person_to_synthetic_images[person_id],
                    )
                else:
                    other_persons = [p for p in self.valid_persons if p != person_id]
                    negative_path = (
                        random.choice(
                            self.person_to_real_images[random.choice(other_persons)],
                        )
                        if other_persons
                        else anchor_path
                    )

            self.triplet_cache.append((anchor_path, positive_path, negative_path))

    def _get_triplet(self, idx: int) -> Tuple[str, str, str]:
        """Returns a triplet (anchor, positive, negative)"""
        if self.triplet_cache_size and self.triplet_cache:
            # Use cache if available
            if idx >= len(self.triplet_cache):
                self._fill_triplet_cache()
            return self.triplet_cache[idx % len(self.triplet_cache)]

        # Generate triplet on the fly if no cache
        person_id = random.choice(self.valid_persons)
        anchor_path = random.choice(self.person_to_real_images[person_id])

        # Choose strategy
        strategy = random.choice(
            ["same_person_real", "diff_person_real", "same_person_synthetic"],
        )

        # Generate positive and negative examples based on strategy
        if strategy == "same_person_real":
            real_images = [
                img
                for img in self.person_to_real_images[person_id]
                if img != anchor_path
            ]
            positive_path = random.choice(real_images) if real_images else anchor_path

            other_persons = [p for p in self.valid_persons if p != person_id]
            if other_persons:
                other_person = random.choice(other_persons)
                negative_path = random.choice(self.person_to_real_images[other_person])
            else:
                negative_path = (
                    random.choice(self.person_to_synthetic_images[person_id])
                    if self.person_to_synthetic_images[person_id]
                    else anchor_path
                )

        elif strategy == "diff_person_real":
            real_images = [
                img
                for img in self.person_to_real_images[person_id]
                if img != anchor_path
            ]
            positive_path = random.choice(real_images) if real_images else anchor_path

            other_persons = [p for p in self.valid_persons if p != person_id]
            if other_persons:
                other_person = random.choice(other_persons)
                negative_path = random.choice(self.person_to_real_images[other_person])
            else:
                negative_path = (
                    random.choice(self.person_to_synthetic_images[person_id])
                    if self.person_to_synthetic_images[person_id]
                    else anchor_path
                )

        elif strategy == "same_person_synthetic":
            real_images = [
                img
                for img in self.person_to_real_images[person_id]
                if img != anchor_path
            ]
            positive_path = random.choice(real_images) if real_images else anchor_path

            if self.person_to_synthetic_images[person_id]:
                negative_path = random.choice(
                    self.person_to_synthetic_images[person_id],
                )
            else:
                other_persons = [p for p in self.valid_persons if p != person_id]
                negative_path = (
                    random.choice(
                        self.person_to_real_images[random.choice(other_persons)],
                    )
                    if other_persons
                    else anchor_path
                )

        return anchor_path, positive_path, negative_path

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        anchor_path, positive_path, negative_path = self._get_triplet(idx)

        # Convert to full paths
        anchor_full_path = self.images_dir / anchor_path
        positive_full_path = self.images_dir / positive_path
        negative_full_path = self.images_dir / negative_path

        # Read and transform images
        anchor = self._read_and_transform_image(anchor_full_path)
        positive = self._read_and_transform_image(positive_full_path)
        negative = self._read_and_transform_image(negative_full_path)

        # Return dictionary with separate keys for anchor, positive, and negative
        return {
            "anchor": anchor,
            "positive": positive,
            "negative": negative,
        }

    def _read_and_transform_image(self, path):
        image = self._read_image(path)
        if self.transforms:
            image = self.transforms(image=image)["image"]
        return image.float()

    def __len__(self):
        if self.triplet_cache_size:
            return self.triplet_cache_size
        return len(self.valid_persons) * 10  # Arbitrary multiplier for dataset size

    @staticmethod
    def _read_image(image_path: Path) -> NDArray[uint8]:
        """Read image using PyVIPS and convert to RGB numpy array"""
        image = pyvips.Image.new_from_file(str(image_path), access="sequential")
        image = image.colourspace("srgb")
        image_np = image.numpy()

        if image_np.shape[-1] == 4:
            image_np = image_np[..., :3]

        return image_np
