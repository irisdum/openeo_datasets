import logging.config

import torch
import torchvision.transforms
from einops import rearrange
from torch import Tensor

from openeo_mmdc.constant.torch_dataloader import S2_BAND

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)
my_logger = logging.getLogger(__name__)


class Clip(torch.nn.Module):
    """
    Clip transform for data, clip each band between two values qmin and qmax
    """

    def __init__(
        self, qmin: list, qmax: list, inplace=False, s2_partial: bool = False
    ):
        super().__init__()

        self.qmin = rearrange(
            torch.Tensor(qmin), "c -> c 1 1 1"
        )  # reshape so that it is broadcastable
        self.qmax = rearrange(torch.Tensor(qmax), "c ->  c  1 1 1 ")
        my_logger.info(
            f"Load transform clipping data to qmin {qmin} qmax {qmax}"
        )
        self.s2_partial = s2_partial

    def forward(self, tensor: Tensor) -> Tensor:
        my_logger.debug(tensor.shape)
        if self.s2_partial:
            tmp_tensor = tensor[: -(tensor.shape[0] - len(S2_BAND)), ...]
        else:
            tmp_tensor = tensor
        assert len(tensor.shape) == 4
        tmp_tensor = torch.min(
            torch.max(tmp_tensor, self.qmin), self.qmax
        )  # clip values on the quantile
        if self.s2_partial:
            tensor[: -(tensor.shape[0] - len(S2_BAND)), ...] = (
                tmp_tensor  # so cloud mask not normalized
            )
            return tensor
        return tmp_tensor

    def __repr__(self):
        return self.__class__.__name__ + "( qmin={} , qmax={})".format(
            self.qmin, self.qmax
        )


class S2Normalize(torch.nn.Module):
    def __init__(
        self,
        med: list | tuple,
        scale: list | tuple,
        s2_partial: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.med = med  # reshape so that it is broadcastable
        self.scale = scale
        my_logger.info(
            f"Load transform clipping data to qmin {med} qmax {scale}"
        )
        self.transform = torchvision.transforms.Normalize(mean=med, std=scale)
        self.s2_partial = s2_partial

    def forward(self, tensor: Tensor):
        tensor = rearrange(tensor, "c t h w -> t c h w")
        # print(tensor.shape)
        if self.s2_partial:
            tmp_tensor = tensor[:, : -(tensor.shape[1] - len(S2_BAND)), ...]
        else:
            tmp_tensor = tensor
        assert len(tensor.shape) == 4
        my_logger.debug(tmp_tensor.shape, self.scale)
        tmp_tensor = self.transform(tmp_tensor)  # clip values on the quantile
        if self.s2_partial:
            tensor[:, : -(tensor.shape[1] - len(S2_BAND)), ...] = (
                tmp_tensor  # so cloud mask not normalized
            )
            return rearrange(tensor, "t c h w -> c t h w")
        return rearrange(tmp_tensor, "t c h w -> c t h w")

    def __repr__(self):
        return self.__class__.__name__ + "( med={} , scale={})".format(
            self.med, self.scale
        )
