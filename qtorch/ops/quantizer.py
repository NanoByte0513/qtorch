import torch
import torch.nn as nn
import torch.nn.functional as F

class Quantizer(nn.Module):
    def __init__(self, bits:int=8, mode=None, signed:bool=True, symmetry:bool=True):
        super(Quantizer, self).__init__()

        if not signed: # unsigned quant range [0, 2^bits-1]
            self.register_buffer('quant_min', torch.tensor(0, dtype=torch.float32))
            self.register_buffer('quant_max', torch.tensor(2**bits - 1, dtype=torch.float32))