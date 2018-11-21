
# Comparing survival curves
This example come from the SNO18 poster presented by VAXIMM in PD-L1lo and PD-L1hi groups.


```python
import lifelines
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
from lifelines import KaplanMeierFitter
```


```python
data=pd.DataFrame({'T': [111,861,778,839,227,250,161,566],
                  'E': [1,0,1,0,1,1,1,0],
                 'pat': [2601,2702,2704,2605,2706,2607,2708,2714],
                 'PD-L1_increase': [True,False,False,False,True,True,True,True]})
```


```python
data
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>T</th>
      <th>E</th>
      <th>pat</th>
      <th>PD-L1_increase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>111</td>
      <td>1</td>
      <td>2601</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>861</td>
      <td>0</td>
      <td>2702</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>778</td>
      <td>1</td>
      <td>2704</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>839</td>
      <td>0</td>
      <td>2605</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>227</td>
      <td>1</td>
      <td>2706</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>250</td>
      <td>1</td>
      <td>2607</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>161</td>
      <td>1</td>
      <td>2708</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>566</td>
      <td>0</td>
      <td>2714</td>
      <td>True</td>
    </tr>
  </tbody>
</table>

```python
ix = data['PD-L1_increase'] == True

kmf_PDL1 = KaplanMeierFitter()

plt.rcParams["figure.figsize"] = (5,5)

kmf_PDL1.fit(data['T'][~ix], data['E'][~ix])
ax=kmf_PDL1.plot(legend=None, show_censors=True, ci_alpha=.1, lw=5, c="#CC00CC",
              censor_styles={'marker':'|', 'ms': 15, 'mew':2, 'markerfacecolor':'red', 
                             'markeredgecolor': 'red'})

kmf_PDL1.fit(data['T'][ix], data['E'][ix])
kmf_PDL1.plot(ax=ax, legend=None, show_censors=True, ci_alpha=.1, lw=5, c="#0F99B2",
              censor_styles={'marker':'|', 'ms': 15, 'mew':2, 'markerfacecolor':'red', 
                             'markeredgecolor': 'red'},)

plt.ylabel('Proportion survival', fontdict={'fontname':'arial', 'size': 18, 'weight': 'bold'})
plt.ylim([0,1.05])
plt.yticks(fontsize=16)
plt.xlim([0, 900])
plt.xticks([0,150, 300, 450, 600, 750, 900], fontsize=16)
plt.setp(ax.spines.values(), linewidth=2)
plt.xlabel('Days from start of treatment', fontdict={'fontname':'arial',
                                                     'size': 18, 'weight': 'bold'})

plt.tight_layout()
plt.savefig('survival_pdl1.svg');
```


![png](output_5_0.png)

```python
from lifelines.statistics import logrank_test
```


```python
results = logrank_test(data['T'][ix], data['T'][~ix], data['E'][ix], data['E'][~ix], alpha=.95)

results.print_summary()
```


```tex
t_0=-1, alpha=0.95, null_distribution=chi squared, df=1

test_statistic      p   
        3.7385 0.0532  .
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1 
```

