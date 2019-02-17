import torch
from flops_counter import get_model_complexity_info
import models as mymodels

nets = [
    ('CifarResNetBasic', [1, 1, 1]),
    ('CifarResNetBasic', [2, 3, 1]),
]

with torch.cuda.device(0):
  for net_type, num_blocks in nets:
      net = getattr(mymodels, net_type)(num_blocks)
      print(net_type, num_blocks)
      flops, params = get_model_complexity_info(net, (32, 32), as_strings=True, print_per_layer_stat=False)
      print('Flops:  {}'.format(flops))
      print('Params: ' + params)