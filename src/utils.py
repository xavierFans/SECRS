from typing import List, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def padded_tensor(
    items: List[Union[List[int], torch.LongTensor]],
    pad_idx: int = 0,
    pad_tail: bool = True,
    max_len: Optional[int] = None,
    debug: bool = False,
    device: torch.device = torch.device('cpu'),
    use_amp: bool = False
) -> torch.LongTensor:
    """Create a padded matrix from an uneven list of lists.

    Returns padded matrix.

    Matrix is right-padded (filled to the right) by default, but can be
    left padded if the flag is set to True.

    Matrix can also be placed on cuda automatically.

    :param list[iter[int]] items: List of items
    :param int pad_idx: the value to use for padding
    :param bool pad_tail:
    :param int max_len: if None, the max length is the maximum item length

    :returns: padded tensor.
    :rtype: Tensor[int64]

    """
    # number of items
    n = len(items)
    # length of each item
    lens: List[int] = [len(item) for item in items]
    # max in time dimension
    t = max(lens)
    # if input tensors are empty, we should expand to nulls
    t = max(t, 1)
    if debug and max_len is not None:
        t = max(t, max_len)

    if use_amp:
        t = t // 8 * 8

    output = torch.full((n, t), fill_value=pad_idx, dtype=torch.long, device=device)

    for i, (item, length) in enumerate(zip(items, lens)):
        if length == 0:
            continue
        if not isinstance(item, torch.Tensor):
            item = torch.tensor(item, dtype=torch.long, device=device)
        if pad_tail:
            output[i, :length] = item
        else:
            output[i, t - length:] = item

    return output


class LSLoss(nn.Module):
    def __init__(self, label_smooth=0.3, class_num=31162):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred, target):
        ''' 
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        '''
        eps = 1e-12
        
        if self.label_smooth is not None:
            # cross entropy loss with label smoothing
            logprobs = F.log_softmax(pred, dim=1)	
            target = F.one_hot(target, self.class_num)	
            target = torch.clamp(target.float(), min=self.label_smooth/(self.class_num-1), max=1.0-self.label_smooth)
            loss = -1*torch.sum(target*logprobs, 1)
        else:
            print("standard cross entropy loss*****")
            # standard cross entropy loss
            loss = -1.*pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred+eps).sum(dim=1))

        return loss.mean()