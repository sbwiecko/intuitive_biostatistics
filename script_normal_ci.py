import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

sample=[np.random.normal(0.25, 0.1, size=20) for i in range(20)]

def W_array(array, conf=0.95):
    t=stats.t(df=len(array)-1).ppf((1+conf)/2)
    return t*stats.sem(array)

mean_list=[np.mean(sample[i]) for i in range(20)]
W_list=[W_array(sample[i]) for i in range(20)]

plt.errorbar(x=mean_list, y=range(20), xerr=W_list, fmt='o', color='k')
plt.axvline(.25, ls='--')
plt.yticks([])

plt.show()
