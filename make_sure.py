import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

dim = 256

u = F.normalize(torch.randn([dim]), p = 2, dim = -1)

results = []
for i in range(100000):
    v = F.normalize(torch.randn([dim]), p = 2, dim = -1)
    #results.append(torch.dot(u,v))
    #results.append(torch.einsum("bd,bd->b", u.unsqueeze(0), v.unsqueeze(0)).squeeze(0))
    results.append(torch.einsum("d,d->", u,v) * dim**(0.5))
    #results.append(torch.abs(u - v).mean())
plt.hist(results, bins = 10)
plt.show()