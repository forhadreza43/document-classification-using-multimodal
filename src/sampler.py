import random
from collections import defaultdict
from typing import Iterator, List, Dict

from torch.utils.data import Sampler

class MinPerClassBatchSampler(Sampler[List[int]]):
    """
    Ensures each batch contains at least `min_per_class` samples for each selected class.
    Batch size must be divisible by min_per_class.

    Paper intent: ensure enough positives in each minibatch for contrastive loss.
    """
    def __init__(self, labels: List[int], batch_size: int, min_per_class: int, seed: int = 42):
        assert batch_size % min_per_class == 0, "batch_size must be divisible by min_per_class"
        self.labels = labels
        self.batch_size = batch_size
        self.min_per_class = min_per_class
        self.classes_per_batch = batch_size // min_per_class
        self.rng = random.Random(seed)

        self.class_to_indices: Dict[int, List[int]] = defaultdict(list)
        for i, y in enumerate(labels):
            self.class_to_indices[int(y)].append(i)

        # keep only classes with enough samples
        self.valid_classes = [c for c, idxs in self.class_to_indices.items() if len(idxs) >= min_per_class]
        if len(self.valid_classes) == 0:
            raise ValueError("No class has enough samples for min_per_class.")

        # shuffle per class pools
        for c in self.valid_classes:
            self.rng.shuffle(self.class_to_indices[c])

    def __iter__(self) -> Iterator[List[int]]:
        ptr = {c: 0 for c in self.valid_classes}

        total = sum(len(self.class_to_indices[c]) for c in self.valid_classes)
        num_batches = max(1, total // self.batch_size)

        for _ in range(num_batches):
            batch: List[int] = []

            # IMPORTANT: allow class repetition so we do not require more distinct classes than exist.
            chosen_classes = [self.rng.choice(self.valid_classes) for _ in range(self.classes_per_batch)]

            for c in chosen_classes:
                idxs = self.class_to_indices[c]
                start = ptr[c]
                end = start + self.min_per_class

                if end > len(idxs):
                    self.rng.shuffle(idxs)
                    start = 0
                    end = self.min_per_class
                    ptr[c] = 0

                batch.extend(idxs[start:end])
                ptr[c] = end

            # Fill if short (should be rare now)
            while len(batch) < self.batch_size:
                c = self.rng.choice(self.valid_classes)
                idxs = self.class_to_indices[c]
                start = ptr[c]
                end = start + 1
                if end > len(idxs):
                    self.rng.shuffle(idxs)
                    start = 0
                    end = 1
                    ptr[c] = 0
                batch.append(idxs[start])
                ptr[c] = end

            yield batch[:self.batch_size]

    def __len__(self) -> int:
        total = sum(len(self.class_to_indices[c]) for c in self.valid_classes)
        return max(1, total // self.batch_size)