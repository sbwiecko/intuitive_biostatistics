# Comparing models

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

# Linear regression


```python
# data from chapter 32
insulin_sensitiv = np.array([250, 220, 145, 115, 230, 200, 330, 400, 370, 260, 270, 530, 375])
C2022_fatacids = np.array([17.9, 18.3, 18.3, 18.4, 18.4, 20.2, 20.3, 21.8, 21.9, 22.1, 23.1, 24.2, 24.4])
```

## Graph


```python
res = stats.linregress(x=C2022_fatacids, y=insulin_sensitiv)

plt.figure(figsize=(8,4))
plt.subplot(1,2,2)
plt.plot(C2022_fatacids, insulin_sensitiv, 'bo', linestyle='None')
plt.ylim(0,600)

x=np.array([17, 25])
plt.plot(x, res.slope*x + res.intercept, 'r-', lw=4)

plt.xlabel('%C20-22 fatty acids')
plt.ylabel('Insulin sensitivity (mg/m²/min)')
plt.title("Linear regression")

plt.subplot(1,2,1)
plt.plot(C2022_fatacids, insulin_sensitiv, 'bo', linestyle='None')
plt.ylim(0,600)

x=np.array([17, 25])
plt.plot(x, [np.mean(insulin_sensitiv)]*2, 'r-', lw=4)

plt.xlabel('%C20-22 fatty acids')
plt.ylabel('Insulin sensitivity (mg/m²/min)')
plt.title("Null hypothesis")

plt.tight_layout();
```


![png](output_4_0.png)

## R²


```python
def compute_rss(y_estimate, y): 
  return sum(np.power(y-y_estimate, 2)) 

def estimate_y(x, b_0, b_1): 
  return b_0 + b_1 * x

y = insulin_sensitiv
x = C2022_fatacids

beta_0 = res.intercept
beta_1 = res.slope

rss = compute_rss(estimate_y(x, beta_0, beta_1), y)

tss = np.sum(np.power(y - np.mean(y), 2))
print(f"R² using TSS and RSS: {(tss-rss)/tss}")
```

    R² using TSS and RSS: 0.5929039159757119

```python
print(f"scatter from regression line: {rss:6.0f}")
```

    scatter from regression line:  63361

```python
print(f"scatter from horizontal line (H0): {compute_rss(y, np.mean(y)):6.0f}")
```

    scatter from horizontal line (H0): 155642

```python
print(f"percentage of variation for the regression line : {rss/compute_rss(y, np.mean(y)):3.1%}")
```

    percentage of variation for the regression line : 40.7%


Comparing the fit of a horizontal line vs. best-fit linear regression line:

Hypothesis | Scatter from | Sum of squares | Percentage of variation | R² 
---|---|---|---|---
Null | Horizontal line | 155642 | 100.0% |
Alternative | Regression line | 63361 | 40.7% |
Difference | Improvement | 92281 | 59.3% | 0.593

Scatter around the regression line accounts for 40.7% of the variation. Therefore the linear regression model itself accounts for $100\% - 40.7\% = 59.3\% $ of the variation. This is the definition of $R²$, which equals 0.593.

## P value

Comparaing the fit of a horizontal line vs the best-fit linear regression line (ANOVA table):

Source of variation | Hypothesis | Scatter from | Sum of squares | DF | MS | F ratio
---|---|---|---|---|---|---
Regression (model) | Difference | Improvement | 92281 | 1 | 92281 | 16.0
Random (residues) | Alternative | Regression line | 63361 | 11 | 5760.1 |
Total (grand mean) | Null | Horizontal line | 155642 | 12

1. 'Total'shows the sum of squares of the distances from the fit of the null hypothesis, 13 data points $-$ one parameter (the mean) leaves $DF=12$.
2. 'Random' shows the sum of squares from the linear regression, two parameters are fitted (slope and intercept) leaving 11 df.
3. 'Regression' shows the difference.

Mean square (MS) also called *variances* is the RSS divided by DF


```python
F = 92281/5760.1
print(f"F ratio : {F:3.1f}")
```

    F ratio : 16.0


The distribution of the *F ratio* is known when the null hypothesis is true


```python
dfn, dfd = 1, 11
p_value = 1 - stats.f(dfn, dfd).cdf(F)
print(f"P values computed from the F ratio distribution: {p_value:5.4f}")
```

    P values computed from the F ratio distribution: 0.0021


# Unpaired t test


```python
# data come from chapter 30
old = np.array([20.8, 2.8, 50, 33.3, 29.4, 38.9, 29.4, 52.6, 14.3])
young=np.array([45.5, 55, 60.7, 61.5, 61.1, 65.5, 42.9, 37.5])
```

To view the unpaired t test as a comparison of the fits of two models, consider it a special case of linear regression.


```python
data = pd.DataFrame({'X': [0]*9+[1]*8, 'y':list(old) + list(young)})
```

## Graph


```python
sns.lmplot('X', 'y', data=data)
plt.xticks([0,1], ['Old ($X=0$)', 'Young ($X=1$)'])
plt.xlim([-.5, 1.5])
plt.ylim([0, 100])
plt.ylabel("$\%E_{max}$")
plt.title("Comparing two groups by linear regression");
```

![png](output_23_1.png)


The slope of the best-fit regression line equals the difference between the means, because X values are one unit apart. The slope is 23.55 with 95% CI ranging from 9.338% to 37.75%, matching the results reported by the unpaired t test. Same for the P value.


```python
import statsmodels.api as sm
X = data['X'].values
y = data['y'].values

X = sm.add_constant(X)
results = sm.OLS(y, X).fit()
print(results.summary2())
```

                     Results: Ordinary least squares
    =================================================================
    Model:              OLS              Adj. R-squared:     0.418   
    Dependent Variable: y                AIC:                139.1606
    Date:               2018-10-15 15:34 BIC:                140.8271
    No. Observations:   17               Log-Likelihood:     -67.580 
    Df Model:           1                F-statistic:        12.47   
    Df Residuals:       15               Prob (F-statistic): 0.00302 
    R-squared:          0.454            Scale:              188.28  
    -------------------------------------------------------------------
                Coef.    Std.Err.     t      P>|t|     [0.025    0.975]
    -------------------------------------------------------------------
    const      30.1667     4.5738   6.5955   0.0000   20.4178   39.9155
    x1         23.5458     6.6674   3.5315   0.0030    9.3346   37.7571
    -----------------------------------------------------------------
    Omnibus:               0.260        Durbin-Watson:          2.044
    Prob(Omnibus):         0.878        Jarque-Bera (JB):       0.358
    Skew:                  -0.240       Prob(JB):               0.836
    Kurtosis:              2.476        Condition No.:          3    
    =================================================================


## Goodness of fit and R²

The t test recast as a comparison of models:

Hypothesis | Scatter from | Sum of squares | Percentage of variation | R² 
---|---|---|---|---
Null | Grand mean | 5172 | 100.0% |
Alternative | Group mean | 2824 | 54.6% |
Difference | Improvement | 2348 | 45.4% | 0.454


```python
print(f"scatter from grand mean (H0): {compute_rss(y, np.mean(y)):6.0f}")
```

    scatter from grand mean (H0):   5172

```python
beta_0_2 = results.params[0]
beta_1_2 = results.params[1]

X = data['X'].values
y = data['y'].values

rss_2 = compute_rss(estimate_y(X, beta_0_2, beta_1_2), y)
print(f"scatter from group means: {rss_2:6.0f}")
```

    scatter from group means:   2824

```python
data.groupby('X').mean()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
    </tr>
    <tr>
      <th>X</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30.166667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>53.712500</td>
    </tr>
  </tbody>
</table>

```python
data['dist_grand'] = data['y'] - data['y'].mean()
```


```python
data['dist_group'] = data.groupby('X')['y'].transform(lambda x: x - x.mean())
```


```python
data.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X</th>
      <th>y</th>
      <th>dist_grand</th>
      <th>dist_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>20.8</td>
      <td>-20.447059</td>
      <td>-9.366667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2.8</td>
      <td>-38.447059</td>
      <td>-27.366667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>50.0</td>
      <td>8.752941</td>
      <td>19.833333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>33.3</td>
      <td>-7.947059</td>
      <td>3.133333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>29.4</td>
      <td>-11.847059</td>
      <td>-0.766667</td>
    </tr>
  </tbody>
</table>

```python
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
data['dist_group'].plot(marker='o', ms=10, linestyle='None')
plt.xticks([])
plt.ylim(-60, 40)
plt.ylabel('Distance from mean')
plt.title("Null")
plt.hlines(y=0, xmin=0, xmax=16, lw=3, colors='gray')

plt.subplot(1,2,2)
data['dist_grand'].plot(marker='+', ms=10, linestyle='None', c='r')
plt.xticks([])
plt.ylim(-60, 40)
plt.ylabel('Distance from mean')
plt.title("Alternative")
plt.hlines(y=0, xmin=0, xmax=16, lw=3, colors='gray')
plt.tight_layout();
```


![png](output_34_0.png)

```python
data['dist_grand'].apply(lambda x: x**2).sum() # sum of squares in H0
```


    5172.222352941178


```python
data.groupby('X')['dist_group'].transform(lambda x: x**2).sum()
# sum of squares in the alternative hypothesis
```


    2824.14875

## P value

The t test recast as a comparison of models:

Source of variation | Hypothesis | Scatter from | Sum of squares | DF | MS | F ratio
---|---|---|---|---|---|---
Between groups | Difference | Improvement | 2348 | 1 | 2348.0 | 12.47
Within groups | Alternative | Group mean | 2824 | 15 | 188.3 |
Total | Null |Grand mean | 5172 | 16 |

1. 'Total'shows the fit of the null hypothesis, 17 data points $-$ one parameter (grand mean) leaves $DF=16$.
2. 'Within groups' shows the fit of the alternative model, two parameters are fitted (the mean of each group) leaving 15 df.
3. 'Regression' shows the difference.

Mean square (MS) also called *variances* is the RSS divided by DF


```python
F = 2348/188.3
dfn, dfd = 1, 15
p_value= 1 - stats.f(dfn, dfd).cdf(F)
print(f"P values computed from the F ratio distribution: {p_value:5.4f}")
```

    P values computed from the F ratio distribution: 0.0030


Viewed as linear regression, the slope is not zero. The goodness of fit of the two models is compared to see whether there is substantial evidence to reject the simpler (null hypothesis) model and accept the mode complicated alternative model.

### statsmodels.api.anova_lm()

There is a function in scipy.stats that generate the ANOVA table for the different models generated.


```python
import statsmodels.formula.api as smf
```


```python
data
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>20.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>33.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>29.4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>38.9</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>29.4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>52.6</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>14.3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>45.5</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>55.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>60.7</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>61.5</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>61.1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>65.5</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>42.9</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>37.5</td>
    </tr>
  </tbody>
</table>

```python
grandmean = smf.ols(formula='y ~ 1', data=data).fit()
```


```python
groupmean = smf.ols(formula='y ~ X', data=data).fit()
```


```python
plt.scatter(data['X'], grandmean.resid, label='grand mean')
plt.scatter(data['X'], groupmean.resid, label='group mean')
plt.axhline(0, color='r', linestyle='--')
plt.legend();
```


![png](output_48_0.png)

```python
for model in [grandmean, groupmean]:
    print(model.mse_resid)
    print(model.rsquared)
    print(model.fvalue)
    print('---')
```

    323.2638970588236
    0.0
    nan
    ---
    188.27658333333335
    0.45397769908440777
    12.471405425835894
    ---

```python
grandmean.summary()
```

<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.000</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.000</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>     nan</td>
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 08 Nov 2018</td> <th>  Prob (F-statistic):</th>  <td>   nan</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>11:13:06</td>     <th>  Log-Likelihood:    </th> <td> -72.724</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    17</td>      <th>  AIC:               </th> <td>   147.4</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    16</td>      <th>  BIC:               </th> <td>   148.3</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     0</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   41.2471</td> <td>    4.361</td> <td>    9.459</td> <td> 0.000</td> <td>   32.003</td> <td>   50.491</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 1.156</td> <th>  Durbin-Watson:     </th> <td>   1.293</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.561</td> <th>  Jarque-Bera (JB):  </th> <td>   0.987</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.513</td> <th>  Prob(JB):          </th> <td>   0.611</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.416</td> <th>  Cond. No.          </th> <td>    1.00</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


```python
groupmean.summary()
```

<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.454</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.418</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   12.47</td>
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 08 Nov 2018</td> <th>  Prob (F-statistic):</th>  <td>0.00302</td>
</tr>
<tr>
  <th>Time:</th>                 <td>11:13:32</td>     <th>  Log-Likelihood:    </th> <td> -67.580</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    17</td>      <th>  AIC:               </th> <td>   139.2</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    15</td>      <th>  BIC:               </th> <td>   140.8</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   30.1667</td> <td>    4.574</td> <td>    6.596</td> <td> 0.000</td> <td>   20.418</td> <td>   39.915</td>
</tr>
<tr>
  <th>X</th>         <td>   23.5458</td> <td>    6.667</td> <td>    3.531</td> <td> 0.003</td> <td>    9.335</td> <td>   37.757</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.260</td> <th>  Durbin-Watson:     </th> <td>   2.044</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.878</td> <th>  Jarque-Bera (JB):  </th> <td>   0.358</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.240</td> <th>  Prob(JB):          </th> <td>   0.836</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.476</td> <th>  Cond. No.          </th> <td>    2.55</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```python
import statsmodels.api as sm
```


```python
sm.stats.anova_lm(grandmean)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>df</th>
      <th>sum_sq</th>
      <th>mean_sq</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Residual</th>
      <td>16.0</td>
      <td>5172.222353</td>
      <td>323.263897</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>

```python
sm.stats.anova_lm(groupmean)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>df</th>
      <th>sum_sq</th>
      <th>mean_sq</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>X</th>
      <td>1.0</td>
      <td>2348.073603</td>
      <td>2348.073603</td>
      <td>12.471405</td>
      <td>0.003022</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>15.0</td>
      <td>2824.148750</td>
      <td>188.276583</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
## ANOVA
This is the analysis of the variance, using sum of squares (SS), between the groups `SSB` and the variance within the group `SSW`. With $SSW + SSB = SST$.
We can compute the F statistic:
$$
F=\frac{SSB/(m-1)}{SSW/(n-m)}
$$
with m the number of groups and n the total number of points.  
When F increase, then the variance between the group (SSB) is bigger than within the group (SSW).
We then compare the F statistic to the critical F* using df1 and df2.

One example of one-way ANOVA comparing 3 groups:  

Hypothesis | Scatter from | Sum of squares | Percentage of variation | $R²$
---|---|---|---|---
Null | Grand mean | 17.38 | 100.0% |
Alternative | Group meand | 16.45 | 94.7% |
Difference |  | .93 | 5.3% | 0.0053

Of all the variation, 94.7% is the result of variation within the groups, leaving 5.3% of the total variation as the result of differences between the group means.  
**The sum of squares resulting from the treatment and the sum of sqaures within the groups always add up to the total sum of squares.**

Source of variation | Sum of squares | DF | MS | F ratio | P value
---|---|---|---|---|---
Between groups | .93 | 2 | .46 | 5.69 | 0.0039
+ Within groups (replicates, residual)| 16.45 | 202 | 0.081 | 
= Total | 17.38 | 204 |

For the total, there are 205 values and only one parameter (grand mean) estimated, so df=204.  
For the SSW, 3 parameters were fit (the mean of each group), so df = 205 - 3 = 202.  
If the null hypothesis were true, F would be likely to have a value close to 1.


```python
dfn, dfd = 2, 202
F = .46 / .081
p_value= 1 - stats.f(dfn, dfd).cdf(F)
print(f"P values computed from the F ratio distribution: {p_value:.4f}")
```

    P values computed from the F ratio distribution: 0.0040


The low P value means that the differences among group means would be very unlikely if in fact all the population means were equal.  
The low R² means that the differences among group means are only a tiny fraction of the overall variability.

# Two-way ANOVA

The data are divided in two ways because each data point is either from an animal given either an inactive or active treatment (one factor) given for either a short or long duration (second factor). If male and female animals were both included you'd need three-way ANOVA.

Two-way ANOVA simultaneously tests 3 null hypotheses and so computes 3 P values:
1. there is no interaction between the two factors (treatment and duration);
2. the population means are identical for animals given placebo and active treatment (treatment), pooling short+log duration;
3. the population means are identical for animals given a treatment for short vs. long duration (duration).

Source of variation | Sum of squares | DF | MS
---|---|---|---
Interaction | 12896 | 1 | 12896 
+ Between rows | 5764 | 1 | 5764
+ Between columns | 3710 | 1 | 3710
+ Among replicates (residual) | 928 | 8 | 116
= Total | 23298 | 11
