import torch
from torch.nn import functional as F


def apply_padding(allow_padd, max_len, t, sits, doy):
    if allow_padd:
        padd_tensor = (0, 0, 0, 0, 0, 0, 0, max_len - t)
        padd_doy = (0, max_len - t)
        # padd_label = (0, 0, 0, 0, 0, self.max_len - t)
        sits = F.pad(sits, padd_tensor)
        # print(f"before {doy.shape}")
        doy = F.pad(doy, padd_doy)
        # print(f"after {doy.shape}")
        padd_index = torch.zeros(max_len)
        padd_index[t:] = 1
        padd_index = padd_index.bool()
    else:
        padd_index = None

    return sits, doy, padd_index
