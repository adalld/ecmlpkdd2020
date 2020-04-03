import torch
import numpy as np


def convert_to_torch_input(list_x: list,
                           dense_y: np.ndarray,
                           mask: np.ndarray,
                           multilabel: bool):
    x = torch.cat(
        [
            torch.as_tensor(
                cur_tensor[mask], dtype=torch.float
            ).unsqueeze(0) for cur_tensor in list_x
        ],
        dim=0
    )
    y = None
    if dense_y is not None:
        if multilabel:
            y = torch.as_tensor(dense_y[mask], dtype=torch.float)
        else:
            y = torch.as_tensor(dense_y[mask].nonzero()[1], dtype=torch.long)
    return x, y

