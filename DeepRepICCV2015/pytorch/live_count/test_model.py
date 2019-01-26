# pytorch モデルに適当な入力を与えて出力を確認， ios上での coreml モデルとの出力と比較する

import numpy as np
import pdb
import sys

import torch
from torch.autograd import Variable

sys.path.append('..')

from layers import RepetitionCountingNet

# モデルを初期化
D_model = RepetitionCountingNet('./weights.save')

# 入力を初期化
input = np.zeros([1, 20, 50, 50])
for i in range(20):
    input[:, i, :, :] = i

pdb.set_trace()

input_variable = Variable(torch.FloatTensor(input))

output_label, pYgivenX = D_model.get_output_labels(input_variable)

print(output_label)
print('\n')
print(pYgivenX)
