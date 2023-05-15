import logging.config

import torch
from einops import rearrange
from torch import Tensor

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

    def __init__(self, qmin: list, qmax: list, inplace=False):
        super().__init__()

        self.qmin = rearrange(
            torch.Tensor(qmin), "c -> 1 c 1 1"
        )  # reshape so that it is broadcastable
        self.qmax = rearrange(torch.Tensor(qmax), "c -> 1 c 1 1 ")
        my_logger.info(
            f"Load transform clipping data to qmin {qmin} qmax {qmax}"
        )

    def forward(self, tensor: Tensor):
        if len(tensor.shape) == 4:
            return torch.min(
                torch.max(tensor, self.qmin), self.qmax
            )  # clip values on the quantile
        elif len(tensor.shape) == 3:
            qmin = torch.squeeze(self.qmin, dim=0)
            qmax = torch.squeeze(self.qmax, dim=0)
            return torch.min(
                torch.max(tensor, qmin), qmax
            )  # clip values on the quantile
        else:
            raise TypeError

    def __repr__(self):
        return self.__class__.__name__ + "( qmin={} , qmax={})".format(
            self.qmin, self.qmax
        )
