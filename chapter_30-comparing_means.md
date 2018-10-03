### Importing libraries

```python
import numpy as np
from scipy.stats import ttest_ind
from scipy import stats
import matplotlib.pyplot as plt
%matplotlib inline
```

### Getting data from the book


```python
old = np.array([20.8, 2.8, 50, 33.3, 29.4, 38.9, 29.4, 52.6, 14.3])
young=np.array([45.5, 55, 60.7, 61.5, 61.1, 65.5, 42.9, 37.5])
```

### Testing normality and equal variance


```python
stats.normaltest(old), stats.normaltest(young)
# H0 = sets randomly sampled from populations with identical means
# same variance and from a normal distribution (large P values --> cannot reject H0)
```

(NormaltestResult(statistic=0.10954414579072048, **pvalue**=0.9467009021600232),
 NormaltestResult(statistic=1.74534981122529, **pvalue**=0.41783239088593094))


```python
plt.boxplot([old, young]);
```


![png](output_3_0.png)

```python
stats.describe(old)
```

DescribeResult(nobs=9, minmax=(2.8, 52.6), mean=30.16666666666667, **variance=259.0375**, skewness=-0.17569385392311104, kurtosis=-0.8080848898945967)


```python
stats.describe(young)
```

DescribeResult(nobs=8, minmax=(37.5, 65.5), mean=53.7125, **variance=107.4069642857143**, skewness=-0.4421628669270582, kurtosis=-1.3632352105713499)

### Student t test

```
print(ttest_ind(old, young).pvalue)
```

0.0030218492023012695

### F ratio


```python
F_ratio = (np.std(old, ddof=1) / np.std(young, ddof=1))**2
print(f"F ratio = {F_ratio:5.3f}")
```

F ratio = 2.412

### t ratio

```python
mean_diff = np.mean(young) - np.mean(old)
print(f"mean difference= {mean_diff:5.2f}")
```

mean difference= 23.55
$$
SEM\substack{mean\_diff} = \sqrt{SEM²_a + SEM²_b}
$$


```python
SEM_mean_diff = np.sqrt(stats.sem(old)**2 + stats.sem(young)**2)
```


```python
t_ratio = mean_diff / SEM_mean_diff
print(f"t ratio = {t_ratio:4.2f}")
```

t ratio = 3.62

### Confidence interval of the mean difference

```python
df = len(old) + len(young) - 2
t_ = stats.t(df=df).ppf((1+.95)/2) # two-tailed
CI_diff_mean = (mean_diff - t_*SEM_mean_diff, mean_diff + t_*SEM_mean_diff)
```


```python
print(f"CI of the mean difference = {CI_diff_mean}")
```

CI of the mean difference = (9.6983295715685, 37.393337095098154)

### Computing CI and P value with bootstraping

```python
bs_old = np.array([np.mean(np.random.choice(old, size=len(old))) for _ in range(10000)])
bs_young= np.array([np.mean(np.random.choice(young,size=len(young))) for _ in range(10000)])

bs_mean_diff = bs_young - bs_old

CI_mean_diff_bs = np.percentile(bs_mean_diff, [2.5, 97.5])
print(f"mean difference using bootstraping = {np.mean(bs_mean_diff):5.2f}")
print(f"CI of the mean difference using bootstraping = {CI_mean_diff_bs}")
```

mean difference using bootstraping = 23.61
CI of the mean difference using bootstraping = [11.57413194 35.66055556]

```python
# now we test the H0 hypothesis that both sets comes from the same population, same mean
combined_mean = np.mean(np.concatenate([old, young]))

young_shifted = young - np.mean(young) + combined_mean
old_shifted = old - np.mean(old) + combined_mean

bs_shifted_old = np.array([np.mean(np.random.choice(
    old_shifted, size=len(old))) for _ in range(10000)])
bs_shifted_young = np.array([np.mean(np.random.choice(
    young_shifted, size=len(old))) for _ in range(10000)])

bs_diff_shifted = bs_shifted_young - bs_shifter_old

P_value_bs = np.sum(bs_diff_shifted >= mean_diff) / len(bs_diff_shifted)
print(f"P value obtained using bootstraping = {P_value_bs:7.6f}")
```

P value obtained using bootstraping = 0.000000

```python
plt.hist(bs_diff_shifted, density=True, bins=100)
#plt.vlines(mean_diff, 0, .01, color='red', linestyles='--')
plt.annotate('mean_diff', xy=(mean_diff, 0), xytext=(mean_diff-7, .015),
            arrowprops={'arrowstyle': '->', 'color': 'red'});
```

![](C:\Users\Sébastien\Downloads\chapter_30-comparing_means (1)\output_21_0.png)