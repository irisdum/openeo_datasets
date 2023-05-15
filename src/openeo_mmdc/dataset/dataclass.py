import logging.config
from dataclasses import dataclass

import pandas as pd
from torch import Tensor

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)
my_logger = logging.getLogger(__name__)


@dataclass
class OneMod:
    sits: Tensor
    doy: Tensor


@dataclass
class ItemTensorMMDC:
    s2: OneMod | None = None
    s1_asc: OneMod | None = None
    s1_desc: OneMod | None = None
    dem: OneMod | None = None
    agera5: OneMod | None = None


@dataclass
class MMDCDF:
    s2: pd.DataFrame | None = None
    s1_asc: pd.DataFrame | None = None
    s1_desc: pd.DataFrame | None = None
    dem: pd.DataFrame | None = None
    dew_temp: pd.DataFrame | None = None
    prec: pd.DataFrame | None = None
    sol_rad: pd.DataFrame | None = None
    temp_max: pd.DataFrame | None = None
    temp_mean: pd.DataFrame | None = None
    temp_min: pd.DataFrame | None = None
    val_press: pd.DataFrame | None = None
    wind_speed: pd.DataFrame | None = None

    def __len__(self):
        if self.s2 is not None:
            return len(self.s2)
        elif self.s1_asc is not None:
            return len(self.s1_asc)
        elif self.s1_desc is not None:
            return len(self.s1_desc)
        elif self.dem is not None:
            return len(self.dem)
        else:
            my_logger.info(
                "No len found maybe add the computation of the agera5 data"
            )
            return None
