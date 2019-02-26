import torch
from flops_counter import get_model_complexity_info
import models as mymodels

nets = [
    # pruning from 48-48-48
    ('CifarResNetBasic', [1, 4, 8], 91.98),
    ('CifarResNetBasic', [4, 13, 10], 93.43),
    ('CifarResNetBasic', [5, 14, 26], 93.61),
    ('CifarResNetBasic', [13, 26, 46], 94.37),
    # pruning from 24-24-24
    ('CifarResNetBasic', [1, 2, 4], 91.50),
    ('CifarResNetBasic', [4, 3, 5], 92.70),
    ('CifarResNetBasic', [4, 5, 7], 93.20),
    ('CifarResNetBasic', [10, 5, 16], 93.80),
    ('CifarResNetBasic', [13, 11, 23], 93.89),
    ('CifarResNetBasic', [17, 10, 24], 93.96),
    # manual search
    ('CifarResNetBasic', [3, 3, 3], 92.96),
    ('CifarResNetBasic', [5, 5, 5], 93.44),
    ('CifarResNetBasic', [7, 7, 7], 93.68),
    ('CifarResNetBasic', [9, 9, 9], 93.88),
    ('CifarResNetBasic', [11, 11, 11], 93.72),
    ('CifarResNetBasic', [13, 13, 13], 93.91),
    ('CifarResNetBasic', [15, 15, 15], 93.90),
    ('CifarResNetBasic', [18, 18, 18], 93.79),
    ('CifarResNetBasic', [24, 24, 24], 94.26),
    ('CifarResNetBasic', [48, 48, 48], 94.54),
    # growing
    ('CifarResNetBasic', [42, 42, 41], 94.31),

]

with torch.cuda.device(0):
  print('Net, flops, params, accuracy:')
  for net_type, num_blocks, accu in nets:
      net = getattr(mymodels, net_type)(num_blocks)
      flops, params = get_model_complexity_info(net, (32, 32), as_strings=False, print_per_layer_stat=False)
      print('{}-{}\t{}\t{}\t{}'.format(net_type, num_blocks, flops, params, accu))