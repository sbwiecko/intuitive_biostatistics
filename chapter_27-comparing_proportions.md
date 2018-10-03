# case study (see page 264)
```python
import numpy as np
from scipy import stats

table = np.array([[73, 756], [14, 826]]) # contingency table (rows = alternative treatments, cols = alternative outcomes)
```

## ci (bootstrap)
```python
bootstrap = np.random.binomial((14+826), 14/(14+826), size=10000)
ci = np.percentile(bootstrap, [2.5, 97.5])
```

## Fisher's test (use chi-square for large sample sizes or more oucomes)
```python
odds_ratio, p_value = stats.fisher_exact(table)
chi2, p, dof, expected = stats.chi2_contingency([[73, 756], [14, 826]])
```

with low we reject H0 that data follow the theoritical distribution
$$
Riskratio (relativerisk) = EE/(EE+EN) / CE/(CE+CN)
$$
is different from odds_ratio (used for case studies with Event in Experimental group smaller than Non-Event, same in the control group: 
$$
oddsratio = EE.CN/EN.CE
$$

```python
risk_ratio = (73/(73+756)) / (14/(14+826))
```

subjects treated with placebo 5.2 times more likely than with apixaban to have recurrent thromboembolism

```python
reduc_relativ_risk = 1 - 1/risk_ratio
```

drug reduced the relative risk by 81% (from 8.8% to 1.7%)

## attribuable risk = difference between the two proportions

```python
diff_frac = 73/(73+756) - 14/(14+826)
```

drug lowers the absolute risk by 7.1%

```python
bootstrap_diff_frac = [np.mean(np.random.binomial(1,
	73/(73+756) - 14/(14+826), 1669)) for _ in range(10000)] # also works this way
ci_diff_frac = np.percentile(bootstrap_diff_frac, [2.5, 97.5])
```

## NNT = number needed to treat

how many patients'd require treatment to reduce the expected number of cases by one

```python
nnt = 1/diff_frac
ci_nnt = np.flip(1/ci_diff_frac) # tuple flipped
```

## for the Mendel's peas experiment
```python
stats.chisquare(f_obs=[315, 108, 101, 32], f_exp=[312.75, 104.25, 104.25,
34.75])
```

