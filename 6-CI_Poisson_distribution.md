```python
import numpy as np
```


```python
a = np.random.poisson(1.6, size=10000)
b = np.random.poisson(7.5, size=10000)
```


```python
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
_ = plt.hist(a, bins=8, normed=True)
_ = plt.hist(b, bins=20, normed=True);
```


![png](output_3_0.png)


# raisins in bagels


```python
raisins = 10
```


```python
dist = np.random.poisson(raisins, size=10000)
conf_int = np.percentile(dist, [2.5, 97.5])
print(f'95% confidence interval for 10 raisins in bagel is {conf_int}')
```

    95% confidence interval for 10 raisins in bagel is [ 4. 17.]

```python
_ = plt.hist(dist, bins=14, normed=True);
```


![png](output_7_0.png)

```python
from scipy.stats import poisson
```


```python
poisson.interval(.95, [10])
```


    (array([4.]), array([17.]))


```python
# method used for large count values (i.e. >=25), but looks like it also work fine with count = 10
raisins - 1.96 * raisins**.5 , raisins + 1.96 * raisins**.5
```


    (3.8019357860699765, 16.198064213930024)

# radioactive count


```python
desintegr = 120
```


```python
radio_dist = np.random.poisson(desintegr, 10000)
ci_desintegr = np.percentile(radio_dist, [2.5, 97.5])
```


```python
ci_desintegr
```


    array([ 99., 142.])


```python
poisson.interval(.95, desintegr)
```


    (99.0, 142.0)


```python
# method used for large count values (i.e. >=25) because it approximates Gaussian distribution
desintegr - 1.96 * desintegr**.5 , desintegr + 1.96 * desintegr**.5
```


    (98.52927574579749, 141.4707242542025)

# person-years


```python
poisson.interval(.95, 1.6)
# count 1.6 event for 1000
```


    (0.0, 4.0)


```python
poisson.interval(.95, 16)
# count 16 event for 10000
```


    (9.0, 24.0)


```python
poisson.interval(.95, 160)
# count 160 event for 100000
```


    (136.0, 185.0)


```python
# important to compute the CI using the actual counts, and then divide by the 'volume'
# counting for a longer time interval (or volume) narrows the CI
```


```python
a_counts = np.random.poisson(700, 20)
```


```python
b_counts = np.random.poisson(7000, 20)
```


```python
_ = plt.boxplot([a_counts, b_counts/10])
plt.xticks((1, 2), ('One min.', 'Ten min.'));
```


![png](output_24_0.png)


## counting raisins in 7 bagels


```python
np.percentile([9, 7, 13, 12, 10, 9, 10], [2.5, 97.5])
```


    array([ 7.3 , 12.85])


```python
poisson.interval(.95, 10)
```


    (4.0, 17.0)

# when observed number is 0?


```python
poisson.interval(.95, 0.1)
```


    (0.0, 1.0)


```python
zero = np.random.poisson(0.1, 100000)
```


```python
_ = plt.hist(zero, range=(-.5, 5), normed=True);
```


![png](output_31_1.png)

```python
np.percentile(zero, [2.5, 97.5])
```


    array([0., 0.])

