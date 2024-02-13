import logging.config
from dataclasses import dataclass

import pandas as pd
import torch.nn
from einops import rearrange
from torch import Tensor

from openeo_mmdc.dataset.padding import apply_padding

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)
my_logger = logging.getLogger(__name__)


@dataclass
class MaskMod:
    mask_cld: Tensor | None = None
    mask_nan: Tensor | None = None
    mask_slc: Tensor | None = None
    mask_clp: Tensor | None = None
    padd_mask: Tensor | None = None  # 1 if the date has been padded


@dataclass
class PaddingMMDC:
    max_len_s2: int | None = None
    max_len_s1_asc: int | None = None
    max_len_s1_desc: int | None = None
    max_len_agera5: int | None = None


@dataclass
class OneMod:
    sits: Tensor
    doy: Tensor
    mask: MaskMod = MaskMod()
    true_doy: None | Tensor = None

    def apply_padding(self, max_len: int, allow_padd=True):
        sits = rearrange(self.sits, "c t h w -> t c h w")
        t = sits.shape[0]
        sits, doy, padd_index = apply_padding(
            allow_padd, max_len, t, sits, self.doy
        )
        if self.true_doy is not None:
            raise NotImplementedError
        if self.mask.mask_cld is not None:
            raise NotImplementedError
        return OneMod(sits=sits, doy=doy, mask=MaskMod(padd_mask=padd_index))


@dataclass
class ItemTensorMMDC:
    s2: OneMod | None = None
    s1_asc: OneMod | None = None
    s1_desc: OneMod | None = None
    dem: OneMod | None = None
    agera5: OneMod | None = None

    def apply_padding(self, paddmmdc: PaddingMMDC):
        return ItemTensorMMDC(
            self.s2.apply_padding(paddmmdc.max_len_s2),
            s1_asc=self.s1_asc.apply_padding(paddmmdc.max_len_s1_asc),
            s1_desc=self.s1_desc.apply_padding(paddmmdc.max_len_s1_desc),
            dem=self.dem,
            agera5=self.agera5.apply_padding(paddmmdc.max_len_agera5),
        )


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


@dataclass
class PT_MMDC_DF:
    s2: pd.DataFrame | None = None
    s1_asc: pd.DataFrame | None = None
    s1_desc: pd.DataFrame | None = None
    dem: pd.DataFrame | None = None
    agera5: pd.DataFrame | None = None

    def __len__(self):
        if self.s2 is not None:
            return len(self.s2)
        elif self.s1_asc is not None:
            return len(self.s1_asc)
        elif self.s1_desc is not None:
            return len(self.s1_desc)
        elif self.dem is not None:
            return len(self.dem)
        elif self.agera5 is not None:
            return len(self.agera5)
        else:
            my_logger.info("No dataframe found ")
            return None


@dataclass
class Stats:
    median: list
    qmin: list
    qmax: list


@dataclass
class MMDC_MAXLEN:
    s2: int | None = None
    s1_asc: int | None = None
    s1_desc: int | None = None
    agera5: int | None = None


@dataclass
class OneTransform:
    transform: torch.nn.Module | None = None
    stats: Stats | None = None


@dataclass
class ModTransform:
    s2: OneTransform | None = OneTransform()
    s1_asc: OneTransform | None = OneTransform()
    s1_desc: OneTransform | None = OneTransform()
    dem: OneTransform | None = OneTransform()
    agera5: OneTransform | None = OneTransform()
