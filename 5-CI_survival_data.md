```python
import pandas as pd
from lifelines import KaplanMeierFitter
```


```python
years = [4.07, 6.54, 1.39, 6.17, 5.89, 4.76, 3.67]
code  = [  1 ,  0  ,  1  ,  0  ,  1  ,  1  ,  0  ]
```


```python
data = pd.DataFrame({'YEARS': years, 'CODE': code},)
```


```python
data
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CODE</th>
      <th>YEARS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>4.07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>6.54</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1.39</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>6.17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5.89</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>4.76</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>3.67</td>
    </tr>
  </tbody>
</table>
censored subjects are encoded 0, death events 1


```python
from statsmodels.stats import proportion
conf_int = proportion.proportion_confint(8, 9, method='binom_test')
conf_int
```


    (0.5565115333605972, 0.9943169550119749)


```python
kmf = KaplanMeierFitter()
```


```python
kmf.fit(years, code)
```


    <lifelines.KaplanMeierFitter: fitted with 7 observations, 3 censored>


```python
%pylab inline
figsize(7,5)
kmf.plot(show_censors=True, legend=False,
        lw=4, c='red')
plt.xlabel('Years', fontsize=18)
plt.ylabel('Suvival', fontsize=18);
```

    Populating the interactive namespace from numpy and matplotlib

![png](output_8_1.png)

```python
kmf.median_
```


    5.89


```python
kmf.survival_function_
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>KM_estimate</th>
    </tr>
    <tr>
      <th>timeline</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.00</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1.39</th>
      <td>0.857143</td>
    </tr>
    <tr>
      <th>3.67</th>
      <td>0.857143</td>
    </tr>
    <tr>
      <th>4.07</th>
      <td>0.685714</td>
    </tr>
    <tr>
      <th>4.76</th>
      <td>0.514286</td>
    </tr>
    <tr>
      <th>5.89</th>
      <td>0.342857</td>
    </tr>
    <tr>
      <th>6.17</th>
      <td>0.342857</td>
    </tr>
    <tr>
      <th>6.54</th>
      <td>0.342857</td>
    </tr>
  </tbody>
</table>

```python
print(6/7)
print(5/7)
print(5/6)
```

    0.8571428571428571
    0.7142857142857143
    0.8333333333333334

```python
# calculating the 95% CI using the Wald method
p = (6+2)/(7+2)
```


```python
W = 2 * (p*(1-p)/(7+4))**.5
```


```python
p-W
```


    0.6993771410172351


```python
p+W
```


    1.0784006367605425
