import math
from typing import Tuple
from mindspore import nn, ops
from mindspore import Tensor
import numpy as np

def one_hot(labels, depth):
    one_hot = np.eye(depth)[labels]

    return one_hot.astype(int)  # 可能需要转换为 int 类型，取决于应用场景


class RandomMixup:
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.
    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def __call__(self, batch: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )
        Returns:
            Tensor: Randomly transformed batch.
        """

        # batch = ms.tensor(batch)
        # target = ms.tensor(target)

        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.dtype == np.float32:
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}. {type(batch)} {batch.shape}")
        if target.dtype != np.int64:
            raise TypeError(f"Target dtype should be ms.float32. Got {target.dtype}")

        if not self.inplace:
            batch = batch.copy()
            target = target.copy()

        if target.ndim == 1:
            target = one_hot(target, depth=self.num_classes)

        if np.random.rand(1).item() >= self.p:
            return batch, target.astype(np.float16)

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = np.roll(batch, shift=1, axis=0)
        target_rolled = np.roll(target, shift=1, axis=0)

        # Implemented as on mixup paper, page 3.
        lambda_param = np.random.dirichlet([self.alpha, self.alpha])[0]
        batch_rolled = batch_rolled * (1.0 - lambda_param)
        batch = batch * lambda_param + batch_rolled

        target_rolled = target_rolled * (1.0 - lambda_param)
        target = target * lambda_param + target_rolled

        return batch, target.astype(np.float16)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_classes={num_classes}"
        s += ", p={p}"
        s += ", alpha={alpha}"
        s += ", inplace={inplace}"
        s += ")"
        return s.format(**self.__dict__)


class RandomCutmix:
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.
    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def __call__(self, batch: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )
        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.dtype == np.float32:
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != np.int64:
            raise TypeError(f"Target dtype should be ms.float32. Got {target.dtype}")

        if not self.inplace:
            batch = batch.copy()
            target = target.copy()

        if target.ndim == 1:
            target = one_hot(target, depth=self.num_classes)

        if np.random.rand(1).item() >= self.p:
            return batch, target.astype(np.float16)

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = np.roll(batch, shift=1, axis=0)
        target_rolled = np.roll(target, shift=1, axis=0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = np.random.dirichlet([self.alpha, self.alpha])[0]
        W, H = batch.shape[2:]

        r_x = np.random.randint(low=0, high=W, size=(1,))
        r_y = np.random.randint(low=0, high=H, size=(1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(np.clip(r_x - r_w_half, min=0))
        y1 = int(np.clip(r_y - r_h_half, min=0))
        x2 = int(np.clip(r_x + r_w_half, max=W))
        y2 = int(np.clip(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled = target_rolled * (1.0 - lambda_param)
        target = target * lambda_param + target_rolled

        return batch, target.astype(np.float16)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_classes={num_classes}"
        s += ", p={p}"
        s += ", alpha={alpha}"
        s += ", inplace={inplace}"
        s += ")"
        return s.format(**self.__dict__)
