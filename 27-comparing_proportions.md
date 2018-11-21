
# case study (see page 264)


```python
import numpy as np
from scipy import stats
```


```python
table = np.array([[73, 756], [14, 826]])
# contingency table (rows = alternative treatments, cols = alternative outcomes)
```


```python
table
```


    array([[ 73, 756],
           [ 14, 826]])

##  Fisher's test


```python
# (use chi-square for large sample sizes or more oucomes)
odds_ratio, p_value = stats.fisher_exact(table)
```


```python
print(odds_ratio)
print(p_value)
```

    5.697089947089947
    1.3340996598862038e-11


## confidence interval

To calculate the confidence interval, we use the log odds ratio, log(or) = log(a*d/b*c), and calculate its standard error:
$ \sqrt{1/n_{00} + 1/n_{01} + 1/n_{10} + 1/n_{11}}$;  
and then ci = $\exp{\log(OR) \pm Z\alpha/2*\log(OR)}$


```python
conf=0.95
se_or = (1/73 + 1/756 + 1/14 + 1/826)**.5
t = stats.t(df = (73+756+14+826) - 1).ppf((1 + conf) /2)
W = t*se_or
ci = np.exp(np.log(odds_ratio) - W), np.exp(np.log(odds_ratio) + W)
print(ci)
```

    (3.1875006695657295, 10.182533975641583)


## statsmodels Table()


```python
import statsmodels.api as sm
```


```python
table = sm.stats.Table2x2(np.array([[73, 756], [14, 826]]))
```


```python
table.summary(method='normal')
```


<table class="simpletable">
<tr>
         <td></td>        <th>Estimate</th>  <th>SE</th>    <th>LCB</th>    <th>UCB</th>  <th>p-value</th>
</tr>
<tr>
  <th>Odds ratio</th>        <td>5.697</td>      <td></td> <td>3.189</td> <td>10.178</td>   <td>0.000</td>
</tr>
<tr>
  <th>Log odds ratio</th>    <td>1.740</td> <td>0.296</td> <td>1.160</td>  <td>2.320</td>   <td>0.000</td>
</tr>
<tr>
  <th>Risk ratio</th>        <td>5.283</td>      <td></td> <td>3.007</td>  <td>9.284</td>   <td>0.000</td>
</tr>
<tr>
  <th>Log risk ratio</th>    <td>1.665</td> <td>0.288</td> <td>1.101</td>  <td>2.228</td>   <td>0.000</td>
</tr>
</table>
Risk ratio (relative risk) = $\frac{\frac{EE}{EE+EN}}{\frac{CE}{CE+CN}}$, is different from odds_ratio (used for case studies with Event in Experimental group smaller than Non-Event, 
same in the control group: odds_ratio = $ \frac{EE.CN}{EN.CE}$


```python
risk_ratio = (73/(73+756)) / (14/(14+826))
```

subjects treated with placebo 5.2 times more likely than with apixaban to have recurrent thromboembolism.


```python
from statsmodels.stats import proportion
print(proportion.proportion_confint(73, 73+756, method='binom_test'))
print(proportion.proportion_confint(14, 14+826, method='binom_test'))
```

    (0.07041488462935341, 0.10961261456629558)
    (0.009701518613066794, 0.027789354812008435)

```python
reduc_relativ_risk = 1 - 1/risk_ratio
print(reduc_relativ_risk)
```

    0.8107305936073059


drug reduced the relative risk by 81% (from 8.8% to 1.7%)


```python
# attribuable risk = difference between the two proportions
diff_frac = 73/(73+756) - 14/(14+826)
print(diff_frac)
# drug lowers the absolute risk by 7.1%
```

    0.0713912344189787

```python
bootstrap_diff_frac = [np.mean(np.random.binomial(1, 73/(73+756) - 14/(14+826), 1669)) for _ in range(10000)]
ci_diff_frac = np.percentile(bootstrap_diff_frac, [2.5, 97.5])
print(ci_diff_frac)
```

    [0.05931696 0.08388256]

```python
# NNT = number needed to treat
# how many patients'd require treatment to reduce the expected number of cases by one
nnt = 1/diff_frac
ci_nnt = np.flip(1/ci_diff_frac) # tuple flipped
print(ci_nnt)
```

    [11.92142857 16.85858586]


## Chi²


```python
# with low we reject H0 that data follow the theoritical distribution
chi2, p, dof, expected = stats.chi2_contingency([[73, 756], [14, 826]])
print(chi2)
print(p)
```

    41.605457103156986
    1.1168109964257731e-10

```python
# for the Mendel's peas experiment
stats.chisquare(f_obs=[315, 108, 101, 32], f_exp=[312.75, 104.25, 104.25,34.75])
```


    Power_divergenceResult(statistic=0.4700239808153477, pvalue=0.925425895103616)
