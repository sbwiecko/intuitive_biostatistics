import numpy as np

arr = np.random.binomial(10, .3333, 10)

N=10000
bts = np.empty(N)

for _ in range(N):
	bts[_] = np.mean(np.random.binomial(1, .3, 1000))
	
print(np.percentile(bts, [2.5, 97.5]))