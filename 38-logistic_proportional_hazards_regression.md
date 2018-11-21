
# Logistic regression

Logistic regression is used when the outcome (dependent variable) has two possible values or categories. Because it is almost always used with two or more independent variables, is should be called multiple logistic regression.

Type of regression|Dependent var (Y)|Examples
---|---|---|
Linear|Continuous (interval, ratio)|Enzyme activity, renal function, weight etc.
Logistic|Binary (dichotomous)|Death during surgery, graduation, recurrence of cancer etc.
Proportional hazards|Elapsed time to event|Months until death, quarters in school before graduation etc.

The odds of an event occuring equals the probability that that event will occur divided by the probability that it will not occur. Every probability can be expressed as odds, every odds can be expressed as a probability.

The logistic regression model computes the odds from baseline odds and from odds ratios computed for each independent variable:
$ Odds = (Baseline Odds).OR_1.OR_2.(...)OR_n $
The baseline odds answers this question, if every single independent X variables equaled 0 what are te odds of a particular category? To make the baseline odds meaningful, we can encode that variable as $Age - 20 $ so that $X=0$ would encode people who are 20.

For each continuous variable such as the age, the corresponding odds ratio answers the question, for each additional year of age, by how much do the odds increase or decrease? If the OR associated with age equals 1.0, then the age is not related. If $OR>1$ then the odds increase by a set percentage for each additional year of age. For instance, an OR of 1.03 would mean that the odds of Y increase by 3% as a person grows older by one year.

$$
Y_i = \beta_0 + \beta_1.X_{i,1} + \beta_2.X_{i,2} + ... + \beta_n.X_{i,n}
$$
with $Y_i$ the natural log of the odds for a particular participant, $\beta_0$ the natural log of the baseline odds, $\beta_2$ the natural log of the odds ratio for the first independent variable etc. For example:
$$
\log\left[\frac{prob(HR_{suicide=oui})}{1-prob(HR_{suicide=oui})}\right] = a + b \times duree + c \times a
$$

The Y value is the natural log of odds, which can be transformed to a probability. Since it implicitly embodies uncertainty, there is no need to explicitly add a random term to the model. Because there is no Gaussian distribution, the method of least squares is not used, instead logistic regression finds the values of the odds ratios using what is called a _maximum likelihood method_ (MLE).

## Fitting a simple model


```python
# let's try the simplest example taken from the MOOC 'Introduction à la statistique avec R'
import pandas as pd
import numpy as np
import statsmodels.api as sm
```


```python
data = pd.read_csv('../data/smp2.csv', delimiter=';')
data.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>prof</th>
      <th>duree</th>
      <th>discip</th>
      <th>n.enfant</th>
      <th>n.fratrie</th>
      <th>ecole</th>
      <th>separation</th>
      <th>juge.enfant</th>
      <th>place</th>
      <th>...</th>
      <th>subst.cons</th>
      <th>scz.cons</th>
      <th>char</th>
      <th>rs</th>
      <th>ed</th>
      <th>dr</th>
      <th>suicide.s</th>
      <th>suicide.hr</th>
      <th>suicide.past</th>
      <th>dur.interv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>31.0</td>
      <td>autre</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>4</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>3</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>50.0</td>
      <td>prof.intermediaire</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47.0</td>
      <td>ouvrier</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>105.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>23.0</td>
      <td>sans emploi</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>

```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 799 entries, 0 to 798
    Data columns (total 26 columns):
    age             797 non-null float64
    prof            793 non-null object
    duree           576 non-null float64
    discip          793 non-null float64
    n.enfant        773 non-null float64
    n.fratrie       799 non-null int64
    ecole           794 non-null float64
    separation      788 non-null float64
    juge.enfant     794 non-null float64
    place           792 non-null float64
    abus            792 non-null float64
    grav.cons       795 non-null float64
    dep.cons        799 non-null int64
    ago.cons        799 non-null int64
    ptsd.cons       799 non-null int64
    alc.cons        799 non-null int64
    subst.cons      799 non-null int64
    scz.cons        799 non-null int64
    char            703 non-null float64
    rs              696 non-null float64
    ed              692 non-null float64
    dr              688 non-null float64
    suicide.s       758 non-null float64
    suicide.hr      760 non-null float64
    suicide.past    785 non-null float64
    dur.interv      749 non-null float64
    dtypes: float64(18), int64(7), object(1)
    memory usage: 162.4+ KB

```python
data_ = data.dropna(subset=['suicide.hr', 'abus'], how='any')
y = data_['suicide.hr']
X = data_['abus']
X = sm.add_constant(X)
model = sm.Logit(y, X)
result = model.fit()
```

```python
result.summary()
```


<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>suicide.hr</td>    <th>  No. Observations:  </th>  <td>   753</td>  
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>  <td>   751</td>  
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>  <td>     1</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Thu, 08 Nov 2018</td> <th>  Pseudo R-squ.:     </th>  <td>0.02098</td> 
</tr>
<tr>
  <th>Time:</th>              <td>13:01:57</td>     <th>  Log-Likelihood:    </th> <td> -372.13</td> 
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td> -380.11</td> 
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th> <td>6.494e-05</td>
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>   -1.6161</td> <td>    0.115</td> <td>  -14.003</td> <td> 0.000</td> <td>   -1.842</td> <td>   -1.390</td>
</tr>
<tr>
  <th>abus</th>  <td>    0.7688</td> <td>    0.190</td> <td>    4.052</td> <td> 0.000</td> <td>    0.397</td> <td>    1.141</td>
</tr>
</table>
```python
result.params
```


    const   -1.616082
    abus     0.768785
    dtype: float64


```python
result.conf_int()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>const</th>
      <td>-1.842275</td>
      <td>-1.389890</td>
    </tr>
    <tr>
      <th>abus</th>
      <td>0.396937</td>
      <td>1.140632</td>
    </tr>
  </tbody>
</table>

```python
OR = np.exp(result.params[1])
CI_OR=np.round(np.exp(np.array([.396937, 1.140632])), 2)
print(f"Odds ratio: {OR:3.2f}")
print(f"CI of the OR: {CI_OR}")
```

    Odds ratio: 2.16
    CI of the OR: [1.49 3.13]

```python
data_['abus'].unique()
```


    array([0., 1.])

The independent variable has two possible values. The corresponding odds ratio is 1.0 for `abus=0`and 2.16 for `abus=1`, meaning that participant with `abus=1` has a bit more than twice the odds of being `suicide.hr`, with CI ranging from 1.49 to 3.13.
If the H0 were true, the OR would equal 1.0, but the CI above doesn't include 1.0, so the corresponding P value must be less than 0.05.

### Analysis of the corresponding contingency table


```python
from scipy import stats
```


```python
table = pd.crosstab(data_['suicide.hr'], data_['abus'])
OR, pval = stats.fisher_exact(table)
print(table)
print(f"Fischer's odds ratio: {OR}")
print(f"Fischer's P value: {pval}")
```

    abus        0.0  1.0
    suicide.hr          
    0.0         453  147
    1.0          90   63
    Fischer's odds ratio: 2.157142857142857
    Fischer's P value: 7.357964669107829e-05


## Fitting a more complex model


```python
data_ = data.dropna(
    subset=['suicide.hr', 'abus', 'discip', 'duree', 'age'], how='any')
```


```python
y = data_['suicide.hr']
X = data_[['abus', 'discip', 'duree', 'age']]
# duree is graduated from 1 to 5 --> better to get dummy
X = sm.add_constant(X)
model2 = sm.Logit(y, X)
result2 = model2.fit()
print(result2.summary2())
```

    Optimization terminated successfully.
             Current function value: 0.484332
             Iterations 6
                             Results: Logit
    =================================================================
    Model:              Logit            Pseudo R-squared: 0.042     
    Dependent Variable: suicide.hr       AIC:              542.7654  
    Date:               2018-11-08 13:02 BIC:              564.3150  
    No. Observations:   550              Log-Likelihood:   -266.38   
    Df Model:           4                LL-Null:          -277.97   
    Df Residuals:       545              LLR p-value:      0.00011671
    Converged:          1.0000           Scale:            1.0000    
    No. Iterations:     6.0000                                       
    -------------------------------------------------------------------
               Coef.    Std.Err.      z      P>|z|     [0.025    0.975]
    -------------------------------------------------------------------
    const      0.1058     0.5304    0.1995   0.8418   -0.9338    1.1454
    abus       0.6161     0.2279    2.7031   0.0069    0.1694    1.0628
    discip     0.4713     0.2506    1.8810   0.0600   -0.0198    0.9624
    duree     -0.3640     0.1271   -2.8643   0.0042   -0.6130   -0.1149
    age       -0.0065     0.0093   -0.7002   0.4838   -0.0247    0.0117
    =================================================================


```python
ORs = np.exp(result2.params)
ORs
```


    const     1.111641
    abus      1.851698
    discip    1.602105
    duree     0.694905
    age       0.993530
    dtype: float64


```python
.99**15
```


    0.8600583546412884

The OR for `age` is 0.99. Every year, the odds ratio of `suicide.hr` goes down about 1%. The OR for a 40-y compared to a 25-y is $0.99^{15}$, with the exponent equals $40-15$. That means a 40-y persone has about 14% lower odds of being `suicide.hr` than a 25-y.

The same assumptions and attentions as for multiple regression applied to logistic regression (e.g. at least 5-10 events per variable, not too many independent variables in the model, etc.)

## Using R-style formula


```python
import statsmodels.formula.api as smf
```


```python
model2 = smf.logit(formula="Q('suicide.hr') ~ abus + discip + duree + age", 
                 data=data_)
```


```python
model2.fit().summary()
```

<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>  <td>Q('suicide.hr')</td> <th>  No. Observations:  </th>  <td>   550</td>  
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>  <td>   545</td>  
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>  <td>     4</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Thu, 08 Nov 2018</td> <th>  Pseudo R-squ.:     </th>  <td>0.04169</td> 
</tr>
<tr>
  <th>Time:</th>              <td>13:03:46</td>     <th>  Log-Likelihood:    </th> <td> -266.38</td> 
</tr>
<tr>
  <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td> -277.97</td> 
</tr>
<tr>
  <th> </th>                      <td> </td>        <th>  LLR p-value:       </th> <td>0.0001167</td>
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>    0.1058</td> <td>    0.530</td> <td>    0.200</td> <td> 0.842</td> <td>   -0.934</td> <td>    1.145</td>
</tr>
<tr>
  <th>abus</th>      <td>    0.6161</td> <td>    0.228</td> <td>    2.703</td> <td> 0.007</td> <td>    0.169</td> <td>    1.063</td>
</tr>
<tr>
  <th>discip</th>    <td>    0.4713</td> <td>    0.251</td> <td>    1.881</td> <td> 0.060</td> <td>   -0.020</td> <td>    0.962</td>
</tr>
<tr>
  <th>duree</th>     <td>   -0.3640</td> <td>    0.127</td> <td>   -2.864</td> <td> 0.004</td> <td>   -0.613</td> <td>   -0.115</td>
</tr>
<tr>
  <th>age</th>       <td>   -0.0065</td> <td>    0.009</td> <td>   -0.700</td> <td> 0.484</td> <td>   -0.025</td> <td>    0.012</td>
</tr>
</table>

## Interactions


```python
data_ = data.dropna(
    subset=['suicide.hr', 'abus', 'discip', 'duree'], how='any')
data_['duree*discip'] = data_['duree'] * data_['discip']

y = data_['suicide.hr']
X = data_[['abus', 'duree*discip']]

X = sm.add_constant(X)
model3 = sm.GLM(y, X, family=sm.families.Binomial())
result3 = model3.fit()
print(result3.summary2())
```

                   Results: Generalized linear model
    ===============================================================
    Model:              GLM              AIC:            551.7420  
    Link Function:      logit            BIC:            -2905.7833
    Dependent Variable: suicide.hr       Log-Likelihood: -272.87   
    Date:               2018-11-08 13:02 LL-Null:        -277.97   
    No. Observations:   550              Deviance:       545.74    
    Df Model:           2                Pearson chi2:   550.      
    Df Residuals:       547              Scale:          1.0000    
    Method:             IRLS                                       
    ---------------------------------------------------------------
                    Coef.  Std.Err.    z     P>|z|   [0.025  0.975]
    ---------------------------------------------------------------
    const          -1.6547   0.1466 -11.2844 0.0000 -1.9421 -1.3673
    abus            0.5841   0.2239   2.6089 0.0091  0.1453  1.0229
    duree*discip    0.0856   0.0524   1.6347 0.1021 -0.0170  0.1883
    ===============================================================


# Proportional hazards regression

is similar to logistic regression but is used when the outcome is elapsed time to an event and is often used for analyses of survival times.


```python
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
surv = pd.read_csv('../data/alcool.csv', delimiter=';')
```


```python
surv.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>t</th>
      <th>SEVRE</th>
      <th>AGE</th>
      <th>SEXE</th>
      <th>EDVNEG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>121</td>
      <td>0</td>
      <td>53</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>121</td>
      <td>0</td>
      <td>52</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>40</td>
      <td>0</td>
      <td>45</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>39</td>
      <td>0</td>
      <td>48</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>66</td>
      <td>0</td>
      <td>45</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
## logrank test (categories)


```python
surv_ = surv[['t', 'SEVRE', 'SEXE', 'AGE']]
```


```python
# survival analysis as a function of sex
ix = surv_['SEXE'] == 1

kmf2 = KaplanMeierFitter()

#plt.rcParams["figure.figsize"] = (6,4)

kmf2.fit(surv_['t'][ix], surv_['SEVRE'][ix])
ax=kmf2.plot(legend=None, show_censors=True, ci_alpha=.1, lw=2, c="#CC00CC",
              censor_styles={'marker':'|', 'ms': 15, 'mew':1, 'markerfacecolor':'red', 
                             'markeredgecolor': 'red'})

kmf2.fit(surv_['t'][~ix], surv_['SEVRE'][~ix])
kmf2.plot(ax=ax, legend=None, show_censors=True, ci_alpha=.1, lw=2, c="#0F99B2",
              censor_styles={'marker':'|', 'ms': 15, 'mew':1, 'markerfacecolor':'red', 
                             'markeredgecolor': 'red'},)

plt.ylabel('Proportion survival', fontdict={'fontname':'arial', 'size': 18, 'weight': 'bold'})
plt.ylim([0,1.05])
#plt.yticks(fontsize=16)

#plt.setp(ax.spines.values(), linewidth=2)
plt.xlabel('Time', fontdict={'fontname':'arial','size': 18, 'weight': 'bold'})

plt.tight_layout()

from lifelines.statistics import logrank_test
results = logrank_test(surv_['t'][ix], surv_['t'][~ix],
                      surv_['SEVRE'][ix], surv_['SEVRE'][~ix])
print(results.summary)
```

       test_statistic         p
    0        0.023537  0.878069

![png](output_42_1.png)


The P value was 0.878, so the difference between the two survival curves was not considered to be statistically significant. Next we do a more sophisticated analysis that adjusts for differences in age, sexe etc.

## Cox regression (proportional hazards regression)

Uses regression methods to fit the relative risk associated with each independent variable, along with a CI and P value testing the H0 that the population relative risk is 1.0.

the name implies we regress covariates (e.g., year elected, country, etc.) against a another variable – in this case durations and lifetimes. Similar to the logic in the first part of this tutorial, we cannot use traditional methods like linear regression.

There are two popular competing techniques in survival regression: Cox’s model and Aalen’s additive model. Both models attempt to represent the hazard rate λ(t|x) as a function of t and some covariates x. In Cox’s model, the relationship is defined:
$$
λ(t|x)=b_0(t)exp(b_1x_1+...+b_dx_d)
$$

Lifelines has an implementation of the Cox propotional hazards regression model (implemented in R under coxph). The idea behind the model is that the log-hazard of an individual is a linear function of their static covariates and a population-level baseline hazard that changes over time. 


```python
from lifelines import CoxPHFitter
cph = CoxPHFitter()
cph.fit(surv, duration_col='t', event_col='SEVRE')
cph.print_summary()
```

    n=125, number of events=27
    
              coef  exp(coef)  se(coef)       z      p  lower 0.95  upper 0.95   
    AGE    -0.0473     0.9538    0.0237 -1.9993 0.0456     -0.0937     -0.0009  *
    SEXE   -0.0151     0.9850    0.6206 -0.0243 0.9806     -1.2314      1.2012   
    EDVNEG -0.4428     0.6422    1.0240 -0.4324 0.6655     -2.4499      1.5643   
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1 
    
    Concordance = 0.628
    Likelihood ratio test = 4.305 on 3 df, p=0.23034

```python
np.exp(np.array([-.0473, -.0937, -.0009]))
```


    array([0.95380121, 0.91055589, 0.9991004 ])


```python
cph.plot()
plt.axvline(0, linestyle='--', color='red');
```


![png](output_50_0.png)


The slope of the survival curve, called _hazard_ is the rate of death (event) in a short time interval. When comparing two groups, we often assume that the ratio of hazard functions is constant over time. The hazards (death rates) may change over the course of the study, but at any time point the group A risk of dying is xx% the risk of the group B. The two hazard functions are proportional to one another, hence the name _proportional hazards regression_. If the assumption is true, then the difference between the survival curves can be quantified by a single number, a **relative risk**. If the ratio is 0.5 the relative risk of dying in one group is half the risk of dying in the other group.


```python
# After fitting, we can plot what the survival curves look like as we vary a single covarite 
# while holding everything else equal. This is useful to understand the impact of a covariate, 
# given the model. 

cph.plot_covariate_groups('AGE', [0, 20, 40, 60])
```


![png](output_52_1.png)

```python
cph.plot_covariate_groups('SEXE', [1,2])
# CI for SEXE and EDVNEG, therefore not much difference
```


![png](output_53_1.png)
