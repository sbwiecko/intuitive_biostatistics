
# Meta-analysis using Python
We use the data extracted from Table 43.1 on page 455


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
mean = np.array([2.5, 1.4, 1.1, 1.2, .8, .85, 1.05, 1.2])
lower= np.array([1.7, .8, .8, .85, .5, .4, .6, .9])
upper= np.array([4.8, 2.2, 1.8, 1.9, 1.4, 1.6, 2, 1.6])
W = (upper-lower)/2
```


```python
plt.errorbar(np.flip(mean), y=np.arange(len(mean)), xerr=W, fmt='o')
plt.xscale('log')
```


![png](output_3_0.png)


Doesn't look optimal, to be enhanced further with python, or waiting for a library dedicated to meta-analysis. For the time better to stick to R.


```python
from statsmodels.stats.contingency_tables import StratifiedTable
```


```python
data=pd.read_csv("../data/catheter.csv", index_col=0, usecols=[1,2,3,4,5])
```


```python
data
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n.trt</th>
      <th>n.ctrl</th>
      <th>col.trt</th>
      <th>col.ctrl</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ciresi</th>
      <td>124</td>
      <td>127</td>
      <td>15.0</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>George</th>
      <td>44</td>
      <td>35</td>
      <td>10.0</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>Hannan</th>
      <td>68</td>
      <td>60</td>
      <td>22.0</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>Heard</th>
      <td>151</td>
      <td>157</td>
      <td>60.0</td>
      <td>82.0</td>
    </tr>
    <tr>
      <th>vanHeerden</th>
      <td>28</td>
      <td>26</td>
      <td>4.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>Maki</th>
      <td>208</td>
      <td>195</td>
      <td>28.0</td>
      <td>47.0</td>
    </tr>
    <tr>
      <th>Bach(a)</th>
      <td>14</td>
      <td>12</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Ramsay</th>
      <td>199</td>
      <td>189</td>
      <td>45.0</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>Appavu</th>
      <td>12</td>
      <td>7</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Trazzera</th>
      <td>123</td>
      <td>99</td>
      <td>16.0</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>Collins</th>
      <td>98</td>
      <td>139</td>
      <td>2.0</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>Bach(b)</th>
      <td>116</td>
      <td>117</td>
      <td>2.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>Tennenberg</th>
      <td>137</td>
      <td>145</td>
      <td>8.0</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>Pemberton</th>
      <td>32</td>
      <td>40</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Logghe</th>
      <td>338</td>
      <td>342</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>

```python
data.iloc[[6,5,3,12,4,11,1,8,10,2]]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n.trt</th>
      <th>n.ctrl</th>
      <th>col.trt</th>
      <th>col.ctrl</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bach(a)</th>
      <td>14</td>
      <td>12</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>Maki</th>
      <td>208</td>
      <td>195</td>
      <td>28.0</td>
      <td>47.0</td>
    </tr>
    <tr>
      <th>Heard</th>
      <td>151</td>
      <td>157</td>
      <td>60.0</td>
      <td>82.0</td>
    </tr>
    <tr>
      <th>Tennenberg</th>
      <td>137</td>
      <td>145</td>
      <td>8.0</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>vanHeerden</th>
      <td>28</td>
      <td>26</td>
      <td>4.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>Bach(b)</th>
      <td>116</td>
      <td>117</td>
      <td>2.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>George</th>
      <td>44</td>
      <td>35</td>
      <td>10.0</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>Appavu</th>
      <td>12</td>
      <td>7</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Collins</th>
      <td>98</td>
      <td>139</td>
      <td>2.0</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>Hannan</th>
      <td>68</td>
      <td>60</td>
      <td>22.0</td>
      <td>22.0</td>
    </tr>
  </tbody>
</table>

```python
mat = np.asarray(data.iloc[[6,5,3,12,4,11,1,8,10,2]])
tables = [np.reshape(x.tolist(), (2, 2)) for x in mat]
```


```python
mat
```


    array([[ 14.,  12.,   0.,   4.],
           [208., 195.,  28.,  47.],
           [151., 157.,  60.,  82.],
           [137., 145.,   8.,  32.],
           [ 28.,  26.,   4.,  10.],
           [116., 117.,   2.,  16.],
           [ 44.,  35.,  10.,  25.],
           [ 12.,   7.,   1.,   1.],
           [ 98., 139.,   2.,  25.],
           [ 68.,  60.,  22.,  22.]])


```python
st = sm.stats.StratifiedTable(tables)
print(st.summary())
```

                       Estimate   LCB    UCB 
    -----------------------------------------
    Pooled odds           2.041   1.612 2.583
    Pooled log odds       0.713   0.478 0.949
    Pooled risk ratio     1.512              
                                             
                     Statistic P-value 
    -----------------------------------
    Test of OR=1        35.866   0.000 
    Test constant OR    21.523   0.011 
                           
    -----------------------
    Number of tables   10  
    Min n              21  
    Max n             478  
    Avg n             217  
    Total n          2170  
    -----------------------

