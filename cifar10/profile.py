import torch
from flops_counter import get_model_complexity_info
import models as mymodels

nets = [
    # pruning from 96-96-96
    ('CifarResNetBasic', [1, 3, 4], 91.09),
    ('CifarResNetBasic', [1, 6, 7], 92.52),
    ('CifarResNetBasic', [2, 11, 17], 92.96),
    ('CifarResNetBasic', [4, 17, 41], 93.59),
#    ('CifarResNetBasic', [2, 22, 63], 93.72),
    ('CifarResNetBasic', [3, 24, 36], 93.74),
    ('CifarResNetBasic', [7, 36, 56], 93.88),
    ('CifarResNetBasic', [7, 45, 64], 93.96),

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
    # growing with gaussian
#    ('CifarResNetBasic', [5, 5, 4], 92.68),
    ('CifarResNetBasic', [4, 6, 3], 93.11),
    ('CifarResNetBasic', [10, 10, 10], 93.34),
    ('CifarResNetBasic', [11, 8, 11], 93.41),
#    ('CifarResNetBasic', [9, 16, 16], 93.34),
    ('CifarResNetBasic', [24, 23, 23], 93.48),
    ('CifarResNetBasic', [33, 32, 32], 94.15),
    ('CifarResNetBasic', [92, 91, 91], 94.41),

    # growing with zero
#    ('CifarResNetBasic', [6, 3, 5], 92.28),
#    ('CifarResNetBasic', [3, 3, 6], 92.56),
    ('CifarResNetBasic', [5, 5, 4], 92.99),
    ('CifarResNetBasic', [22, 6, 6], 93.18),
    ('CifarResNetBasic', [49, 18, 18], 93.33),
    ('CifarResNetBasic', [32, 31, 31], 93.86),
    ('CifarResNetBasic', [42, 42, 41], 94.31),

]

with torch.cuda.device(0):
  print('Net, flops, params, accuracy:')
  for net_type, num_blocks, accu in nets:
      net = getattr(mymodels, net_type)(num_blocks)
      flops, params = get_model_complexity_info(net, (32, 32), as_strings=False, print_per_layer_stat=False)
      print('{}-{}\t{}\t{}\t{}'.format(net_type, num_blocks, flops, params, accu))