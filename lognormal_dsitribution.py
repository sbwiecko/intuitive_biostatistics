import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#data come from Intuitive Biostatistics Chapter 25 - Outliers, Figure 25.2
array = np.array([47, 23, 19, 16, 15, 11,
				  10, 10, 9, 9, 8, 7, 1, 2, 2])

sns.stripplot(y=array)

distribution = [np.random.choice(array) for _ in range(10000)]
plt.subplot(1,2,1)
sns.violinplot(y=distribution)
plt.subplot(1,2,2)
sns.violinplot(y=np.log10(distribution))

plt.show()
