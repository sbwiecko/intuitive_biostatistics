"""
Cumulative P values and ad hoc sample size decisions.
Here we simulate the cumulative influence of the addition of samples to the
previous set on the P value. The two populations are from a Gaussian 
distribution with the same mean and SD.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

mean = 0
SD   = 1
N    = 75

pop_1 = np.random.normal(loc=mean, scale=SD, size=N)
pop_2 = np.random.normal(loc=mean, scale=SD, size=N)

p_values = []

for _ in range(5, N):
	subset_1 = pop_1[:_]
	subset_2 = pop_2[:_]
	
	stat, p_value = stats.ttest_ind(subset_1, subset_2)
	p_values.append(p_value)
	
plt.plot([_ for _ in range(5, N)], p_values)
plt.xlim((0,75))
plt.yscale('log')
plt.ylim((.01, 1))
plt.hlines(.05, 0, 75, colors='r', linestyles='dashed')
