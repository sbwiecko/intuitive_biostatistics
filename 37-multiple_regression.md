
# Definitions

**Multiple regression** extends simple linear regression to allow for multiple independent (X) variables:  
- to assess the impact of one variable after accounting for others,
- to create an equation for making useful predictions,
- to understand scientifically how much changes in each of several variables contribute to explaining an outcome of interest.  

Multiple linear regression --> the outcome variable is continuous.  
Logistic regression --> binary outcome.  
Proportional hazards regression --> survival times.

A regression model predicts one variable Y from one or more other variables X.  
The Y variable is called *dependent variable*, *the response variable*, or *the outcome variable*.  
The X variables are called *independent variables*, *explanatory variables*, or *predictor variables*.  
Each independent variable can be:
+ continuous (e.g. age, blood pressure, weight)
+ binary/dummy variable (e.g. code for gender)
+ categorical with 3 or more categories (e.g. school classes, countries)

The multiple regression model defines the dependent variables as a function of the independent variables and a set of parameters or *regression coefficients*. Regression methods find the values of each parameter that make the model predictions come as close as possible to the data. This approach is analoguous to linear regression.  
Multiple regression is also called *multivariable regression* refering to models with 2 or more X variables.

Methods exist that can simultaneously analyze several outcomes (Y variables) at once = *multivariate methods*, e.g. factor analysis, cluster analysis, PCA and MANOVA. *Univariate methods* deal witha single Y.

# Data

We will use the data from the MOOC 'Introducton à la statistique avec R' which deals with multiple regression.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.api as sm
```


```python
data = pd.read_csv('./data/smp2.csv', delimiter=';')
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
      <th>abus</th>
      <th>grav.cons</th>
      <th>dep.cons</th>
      <th>ago.cons</th>
      <th>ptsd.cons</th>
      <th>alc.cons</th>
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
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
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
      <td>0.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0.0</td>
      <td>2.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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


## Simple linear regression


```python
plt.figure(figsize=(5,5)).gca()
plt.plot(data['dur.interv'], data['age'], 'bo', markersize=2, linestyle='None')
plt.ylabel('Age')
plt.xlabel("Durée d'intervention");
```


![png](output_11_0.png)

```python
data_ = data.dropna(subset=['age', 'dur.interv']) # drop any Nan in the indicated subset
y = data_['age']
X = data_['dur.interv']

X = sm.add_constant(X)
mod = sm.OLS(y, X)
est = mod.fit()
print(est.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                    age   R-squared:                       0.007
    Model:                            OLS   Adj. R-squared:                  0.006
    Method:                 Least Squares   F-statistic:                     5.516
    Date:                Fri, 19 Oct 2018   Prob (F-statistic):             0.0191
    Time:                        20:57:19   Log-Likelihood:                -2991.2
    No. Observations:                 747   AIC:                             5986.
    Df Residuals:                     745   BIC:                             5996.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         35.4908      1.611     22.026      0.000      32.328      38.654
    dur.interv     0.0582      0.025      2.349      0.019       0.010       0.107
    ==============================================================================
    Omnibus:                       43.795   Durbin-Watson:                   1.591
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               50.685
    Skew:                           0.637   Prob(JB):                     9.86e-12
    Kurtosis:                       2.925   Cond. No.                         216.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


## Multiple linear regression

Let's compute of the parameters for independent variables *age*, *dep.cons*, *subst.cons* and *scz.cons* using the model:
$$
Y_i = \beta_0 + \beta_1.X_{i, age} + \beta_2.X_{i, dep.cons} + \beta_3.X_{i, subst.cons} + \beta_4.X_{i, scz.cons} + \epsilon_i
$$
with subscript _i_ refering to the particular participant, and with the 3 last variables designated as **dummy variables** because those values were chosen arbitrarily to designate two groups (yes or no).

Each regression coefficient represents the average change in Y when we change the corresponding X value by 1.0. For example, $\beta_4$ is the average difference in Y between those who have _scz.cons_ and those who don't.  
The intercept $\beta_0$ is the predicted average value of Y when all the X values are zero, and might only have a mathematical meaning.


```python
data_ = data.dropna(subset=['dur.interv', 'age', 'dep.cons', 'subst.cons', 'scz.cons'])
y = data_['dur.interv']
X = data_[['age', 'dep.cons', 'subst.cons', 'scz.cons']]

X = sm.add_constant(X)
mod = sm.OLS(y, X)
est = mod.fit()
print(est.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:             dur.interv   R-squared:                       0.058
    Model:                            OLS   Adj. R-squared:                  0.053
    Method:                 Least Squares   F-statistic:                     11.49
    Date:                Fri, 19 Oct 2018   Prob (F-statistic):           4.69e-09
    Time:                        16:33:56   Log-Likelihood:                -3260.7
    No. Observations:                 747   AIC:                             6531.
    Df Residuals:                     742   BIC:                             6554.
    Df Model:                           4                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         48.9011      2.622     18.649      0.000      43.753      54.049
    age            0.2210      0.057      3.871      0.000       0.109       0.333
    dep.cons       7.3893      1.448      5.104      0.000       4.547      10.232
    subst.cons     5.2516      1.743      3.013      0.003       1.829       8.674
    scz.cons       2.2726      2.523      0.901      0.368      -2.681       7.226
    ==============================================================================
    Omnibus:                       28.567   Durbin-Watson:                   1.072
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               31.557
    Skew:                           0.452   Prob(JB):                     1.40e-07
    Kurtosis:                       3.445   Cond. No.                         167.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

The investigator's goal was to answer the question, after adjusting for effects of the other variables, is there a substantial linear relationship between the *age* and *dur.interv*.  
The best-fit value of $\beta_1$ was 0.2210. This means that, on average, and after accounting for differences in the other variables, an increase in *age* of one unit is associated with an increase of *dur.interv* of 0.2210. The 95% CI ranges from 0.109 to 0.333.  
For the dummy variables, for example *dep.cons*, $\beta_2$ was 7.3893. That variable is coded as zero if the participant had no *dep.cons* and 1 if he had. Therefore, on average, participant who had *dep.cons* had an *dur.interv* that was 7.389 units higher that of participants who had not.

### Statistical significance

Multiple regression computes a P value for each parameter in the model testing the null hypothesis that the true value of that parameter is zero. Why zero? When a regression coefficient equals zero, then the corresponding independent variable has **no effect in the model**.  
The CI of $\beta_4$ runs from a negative to a positive number, including the value defining H0, then the P value must be greater thant 0.05.

$R²$ equals 0.058, this means that only 5.8% of the variability in Y is explained by the model.  
With multiple regression models which have more than two independent variables, we cannot visualize the best-fit line superimposed on the data. A way to visualize how well a multiple regression model fit the data is presented below.  
For each point/participant, the prediction is computed from the other variables for that participant and the best-fit parameter values.


```python
data_['pred'] = est.predict(X)
```

```python
import seaborn as sns
sns.regplot('dur.interv', 'pred', data=data_, ci=None, marker='+', 
            x_jitter=3, y_jitter=2, color='brown');
```


![png](output_22_0.png)

```python
plt.hist(est.resid, bins=10); # one of the validity condition for the multiple regression is
# the absence of evident correlative structure in the distribution of the noise (residues)
```


![png](output_23_0.png)

```python
from scipy import stats
```


```python
Rsq = stats.pearsonr(data_['dur.interv'], data_['pred'])[0]**2
print(Rsq)
```

    0.058329849335087426


The predicted and actual data values are correlated, with $R²$ equal to 0.058. By definition this is identical to the overall $R²$ computed by multiple regression!

Even if the independent variables are completely unable to predict the dependent variable, $R²$ will be greater than zero, limiting the usefulness of $R²$ as a way to quantify goodness of fit, especially with small sample sizes.  
Adjusted $R²$ estimates how well the model is expected to fit new data. This measure accounts for the number od independent variables and is always smaller than $R²$.
$$
R^2_{adj} = 1 - \left[\frac{(1-R^2)(n-1)}{n-k-1}\right]
$$


```python
n = est.nobs
k = len(est.params) - 1 # minus 'const'
Rsq_adj = 1 - ((1-Rsq)*(n-1)) / (n-k-1)
print(f"R² adjusted = {Rsq_adj:4.3f}")
```

    R² adjusted = 0.053


Rules of thumb specify that the number of participants (n) should be somewhere between 10-40 times the number of variables (k). With ca. 800 participants, we may analyze 20 independent variables.

### Selection of the variable?

The authors of the example presented in the chapter 37 of Intuitive Biostatistics 4th stated that the collected data for more variables for each participant and that the fit of the model was not improved when the model also account for smoking, mean blood pressure, residence in urban vs. rural etc. Consequently, the omitted these variables from the model whose fit they reported. In other words they computed a P value for each independent variable in the model, removed variables for which P values were greater than 0.05 and then reran the model wothout those variables (backward-stepwise selection of step-down).  

Deciding how to construct models is a difficult problem...


```python
data.columns
```


    Index(['age', 'prof', 'duree', 'discip', 'n.enfant', 'n.fratrie', 'ecole',
           'separation', 'juge.enfant', 'place', 'abus', 'grav.cons', 'dep.cons',
           'ago.cons', 'ptsd.cons', 'alc.cons', 'subst.cons', 'scz.cons', 'char',
           'rs', 'ed', 'dr', 'suicide.s', 'suicide.hr', 'suicide.past',
           'dur.interv'],
          dtype='object')


```python
# let's try to fit the data with more variables
var = ['age', 'n.enfant', 'grav.cons', 'dep.cons', 'ago.cons', 'alc.cons', 
       'subst.cons', 'scz.cons']
data_ = data.dropna(subset=['dur.interv']+var)

y = data_['dur.interv']
X = data_[var]

X = sm.add_constant(X)
mod = sm.OLS(y, X)
est = mod.fit()
print(est.summary2())
```

                     Results: Ordinary least squares
    ==================================================================
    Model:              OLS              Adj. R-squared:     0.071    
    Dependent Variable: dur.interv       AIC:                6310.4284
    Date:               2018-10-19 15:40 BIC:                6351.6667
    No. Observations:   722              Log-Likelihood:     -3146.2  
    Df Model:           8                F-statistic:        7.891    
    Df Residuals:       713              Prob (F-statistic): 3.33e-10 
    R-squared:          0.081            Scale:              361.38   
    -------------------------------------------------------------------
                    Coef.   Std.Err.     t     P>|t|    [0.025   0.975]
    -------------------------------------------------------------------
    const          44.2043    3.1398  14.0788  0.0000  38.0400  50.3687
    age             0.1943    0.0636   3.0545  0.0023   0.0694   0.3192
    n.enfant        0.8929    0.4232   2.1100  0.0352   0.0621   1.7236
    grav.cons       1.3273    0.5282   2.5126  0.0122   0.2902   2.3644
    dep.cons        5.3050    1.6507   3.2138  0.0014   2.0642   8.5458
    ago.cons       -2.0663    2.0192  -1.0233  0.3065  -6.0305   1.8980
    alc.cons        4.6952    1.9905   2.3588  0.0186   0.7873   8.6032
    subst.cons      4.2152    1.8435   2.2865  0.0225   0.5958   7.8346
    scz.cons        0.4231    2.6815   0.1578  0.8747  -4.8415   5.6878
    ------------------------------------------------------------------
    Omnibus:               26.097       Durbin-Watson:          1.077 
    Prob(Omnibus):         0.000        Jarque-Bera (JB):       28.298
    Skew:                  0.446        Prob(JB):               0.000 
    Kurtosis:              3.381        Condition No.:          191   
    ==================================================================

## Interactions among independent variables

What if the effects of one variable matter more with another variable? To include interaction between 2 variables, add a new term in the model equation with a new parameter multiplyied by the product of e.g. dep.cons ($X_2$) times subst.cons ($X_3$):
$$
Y_i = \beta_0 + \beta_1.X_{i, age} + \beta_2.X_{i, dep.cons} + \beta_3.X_{i, subst.cons} + \beta_4.X_{i, scz.cons} + \beta_{2,3}.X_{i, dep.cons}.X_{i, subst.cons} + \epsilon_i
$$


```python
# let's try to fit the data with more variables
var = ['age', 'dep.cons', 'subst.cons', 'scz.cons']
data_ = data.dropna(subset=['dur.interv']+var)
data_['interaction'] = data['dep.cons']*data['subst.cons'] # math multiplication

y = data_['dur.interv']
X = data_[var+['interaction']]

X = sm.add_constant(X)
mod = sm.OLS(y, X)
est = mod.fit()
print(est.summary2())
```

                     Results: Ordinary least squares
    ==================================================================
    Model:              OLS              Adj. R-squared:     0.054    
    Dependent Variable: dur.interv       AIC:                6531.3868
    Date:               2018-10-19 16:56 BIC:                6559.0831
    No. Observations:   747              Log-Likelihood:     -3259.7  
    Df Model:           5                F-statistic:        9.588    
    Df Residuals:       741              Prob (F-statistic): 7.02e-09 
    R-squared:          0.061            Scale:              364.17   
    -------------------------------------------------------------------
                    Coef.   Std.Err.     t     P>|t|    [0.025   0.975]
    -------------------------------------------------------------------
    const          49.5169    2.6579  18.6303  0.0000  44.2991  54.7348
    age             0.2173    0.0571   3.8045  0.0002   0.1052   0.3294
    dep.cons        6.1578    1.6977   3.6270  0.0003   2.8248   9.4908
    subst.cons      3.1724    2.2985   1.3802  0.1679  -1.3399   7.6848
    scz.cons        1.9723    2.5309   0.7793  0.4361  -2.9963   6.9410
    interaction     4.4969    3.2430   1.3867  0.1660  -1.8696  10.8634
    ------------------------------------------------------------------
    Omnibus:               29.591       Durbin-Watson:          1.078 
    Prob(Omnibus):         0.000        Jarque-Bera (JB):       32.889
    Skew:                  0.459        Prob(JB):               0.000 
    Kurtosis:              3.462        Condition No.:          232   
    ==================================================================


​    ATTENTION, the effects of the single variables of the interaction cannont be interpreted...

### patsy formula (R-style)


```python
import statsmodels.formula.api as smf
#statsmodels allows users to fit statistical models using R-style formulas. Internally, statsmodels uses the patsy 
#package to convert formulas and data to the matrices that are used in model fitting.

model = smf.ols(formula=                                            # V
                "Q('dur.interv') ~ age + Q('scz.cons') + Q('dep.cons'):Q('subst.cons')", 
                data=data_)
# Q() is a way to ‘quote’ variable names, especially ones that do not otherwise 
# meet Python’s variable name rules, such as with a dot in the variable name

#“:” adds a new column to the design matrix with the product of the other two columns.
#“*” will also include the individual columns that were multiplied together:

results = model.fit()
print(results.summary2()) # R-style formula just works the same!
```

                           Results: Ordinary least squares
    =============================================================================
    Model:                   OLS                 Adj. R-squared:        0.040    
    Dependent Variable:      Q('dur.interv')     AIC:                   6540.6674
    Date:                    2018-10-20 14:24    BIC:                   6559.1317
    No. Observations:        747                 Log-Likelihood:        -3266.3  
    Df Model:                3                   F-statistic:           11.38    
    Df Residuals:            743                 Prob (F-statistic):    2.66e-07 
    R-squared:               0.044               Scale:                 369.70   
    -----------------------------------------------------------------------------
                                   Coef.  Std.Err.    t    P>|t|   [0.025  0.975]
    -----------------------------------------------------------------------------
    Intercept                     52.6832   2.3309 22.6022 0.0000 48.1073 57.2591
    age                            0.1966   0.0545  3.6083 0.0003  0.0896  0.3036
    Q('scz.cons')                  2.0508   2.5486  0.8047 0.4213 -2.9526  7.0541
    Q('dep.cons'):Q('subst.cons') 11.2797   2.1915  5.1469 0.0000  6.9774 15.5821
    -----------------------------------------------------------------------------
    Omnibus:                  29.466           Durbin-Watson:              1.088 
    Prob(Omnibus):            0.000            Jarque-Bera (JB):           32.326
    Skew:                     0.468            Prob(JB):                   0.000 
    Kurtosis:                 3.402            Condition No.:              154   
    =============================================================================


```python
model = smf.ols(formula=                                            # V
                "Q('dur.interv') ~ age + Q('scz.cons') + Q('dep.cons')*Q('subst.cons')", 
                data=data_)

#“:” adds a new column to the design matrix with the product of the other two columns.
#“*” will also include the individual columns that were multiplied together:

results = model.fit()
print(results.summary2())
```

                           Results: Ordinary least squares
    =============================================================================
    Model:                   OLS                 Adj. R-squared:        0.054    
    Dependent Variable:      Q('dur.interv')     AIC:                   6531.3868
    Date:                    2018-10-20 14:24    BIC:                   6559.0831
    No. Observations:        747                 Log-Likelihood:        -3259.7  
    Df Model:                5                   F-statistic:           9.588    
    Df Residuals:            741                 Prob (F-statistic):    7.02e-09 
    R-squared:               0.061               Scale:                 364.17   
    -----------------------------------------------------------------------------
                                   Coef.  Std.Err.    t    P>|t|   [0.025  0.975]
    -----------------------------------------------------------------------------
    Intercept                     49.5169   2.6579 18.6303 0.0000 44.2991 54.7348
    age                            0.2173   0.0571  3.8045 0.0002  0.1052  0.3294
    Q('scz.cons')                  1.9723   2.5309  0.7793 0.4361 -2.9963  6.9410
    Q('dep.cons')                  6.1578   1.6977  3.6270 0.0003  2.8248  9.4908
    Q('subst.cons')                3.1724   2.2985  1.3802 0.1679 -1.3399  7.6848
    Q('dep.cons'):Q('subst.cons')  4.4969   3.2430  1.3867 0.1660 -1.8696 10.8634
    -----------------------------------------------------------------------------
    Omnibus:                  29.591           Durbin-Watson:              1.078 
    Prob(Omnibus):            0.000            Jarque-Bera (JB):           32.889
    Skew:                     0.459            Prob(JB):                   0.000 
    Kurtosis:                 3.462            Condition No.:              232   
    =============================================================================

#### removal of the intercept parameter


```python
# intercept automatically added using patsy formula, can be removed usng -1
model = smf.ols(formula=
                "Q('dur.interv') ~ age + Q('scz.cons') + Q('dep.cons'):Q('subst.cons') - 1", 
                data=data_)
results = model.fit()
print(results.summary2())
```

                           Results: Ordinary least squares
    =============================================================================
    Model:                   OLS                 Adj. R-squared:        0.853    
    Dependent Variable:      Q('dur.interv')     AIC:                   6929.5606
    Date:                    2018-10-20 14:23    BIC:                   6943.4087
    No. Observations:        747                 Log-Likelihood:        -3461.8  
    Df Model:                3                   F-statistic:           1441.    
    Df Residuals:            744                 Prob (F-statistic):    2.40e-309
    R-squared:               0.853               Scale:                 623.06   
    -----------------------------------------------------------------------------
                                   Coef.  Std.Err.    t    P>|t|   [0.025  0.975]
    -----------------------------------------------------------------------------
    age                            1.3579   0.0235 57.7363 0.0000  1.3118  1.4041
    Q('scz.cons')                  6.6538   3.2980  2.0175 0.0440  0.1793 13.1283
    Q('dep.cons'):Q('subst.cons') 27.6992   2.6842 10.3195 0.0000 22.4298 32.9687
    -----------------------------------------------------------------------------
    Omnibus:                  2.927            Durbin-Watson:               1.402
    Prob(Omnibus):            0.231            Jarque-Bera (JB):            2.769
    Skew:                     0.120            Prob(JB):                    0.250
    Kurtosis:                 3.178            Condition No.:               152  
    =============================================================================  

We can also use functions like in $y$ ~ $a + a:b + np.log(x)$

### Categorical variable > 2 classes


```python
# let's introduce the categorical variable 'prof'
data['prof'].describe()
```


    count         793
    unique          8
    top       ouvrier
    freq          227
    Name: prof, dtype: object

#### using pd.get_dummies


```python
# preparation of the subset, i.e. selection of the variables and dropna
var = ['age', 'dep.cons', 'subst.cons', 'scz.cons', 'prof']
df = data.dropna(subset=['dur.interv']+var)

# getting dummies
dummies = pd.get_dummies(df['prof'], drop_first=True)
data_ = pd.concat([df, dummies], axis=1)
```


```python
data_.columns
```


    Index(['age', 'prof', 'duree', 'discip', 'n.enfant', 'n.fratrie', 'ecole',
           'separation', 'juge.enfant', 'place', 'abus', 'grav.cons', 'dep.cons',
           'ago.cons', 'ptsd.cons', 'alc.cons', 'subst.cons', 'scz.cons', 'char',
           'rs', 'ed', 'dr', 'suicide.s', 'suicide.hr', 'suicide.past',
           'dur.interv', 'artisan', 'autre', 'cadre', 'employe', 'ouvrier',
           'prof.intermediaire', 'sans emploi'],
          dtype='object')


```python
var = ['age', 'dep.cons', 'subst.cons', 'scz.cons', 
        'artisan', 'autre', 'cadre', 'employe', 'ouvrier',
       'prof.intermediaire', 'sans emploi']

y = data_['dur.interv']
X = data_[var]

X = sm.add_constant(X)
mod = sm.OLS(y, X)
est = mod.fit()
print(est.summary2())
```

                      Results: Ordinary least squares
    ====================================================================
    Model:                OLS              Adj. R-squared:     0.052    
    Dependent Variable:   dur.interv       AIC:                6504.1550
    Date:                 2018-10-20 14:39 BIC:                6559.4834
    No. Observations:     743              Log-Likelihood:     -3240.1  
    Df Model:             11               F-statistic:        4.692    
    Df Residuals:         731              Prob (F-statistic): 5.82e-07 
    R-squared:            0.066            Scale:              365.04   
    --------------------------------------------------------------------
                        Coef.   Std.Err.    t    P>|t|   [0.025   0.975]
    --------------------------------------------------------------------
    const               62.7920  10.2078  6.1514 0.0000  42.7519 82.8321
    age                  0.2129   0.0588  3.6180 0.0003   0.0974  0.3284
    dep.cons             7.3679   1.4584  5.0521 0.0000   4.5048 10.2311
    subst.cons           5.3459   1.7690  3.0220 0.0026   1.8729  8.8188
    scz.cons             2.5044   2.5473  0.9831 0.3259  -2.4966  7.5054
    artisan            -11.4851   9.8294 -1.1685 0.2430 -30.7823  7.8120
    autre              -10.2875  10.3348 -0.9954 0.3199 -30.5770 10.0020
    cadre              -19.2964  10.3857 -1.8580 0.0636 -39.6857  1.0930
    employe            -13.5581   9.7634 -1.3887 0.1654 -32.7257  5.6096
    ouvrier            -14.0127   9.7211 -1.4415 0.1499 -33.0973  5.0719
    prof.intermediaire -13.0193   9.9691 -1.3060 0.1920 -32.5908  6.5522
    sans emploi        -14.2787   9.7178 -1.4693 0.1422 -33.3568  4.7995
    --------------------------------------------------------------------
    Omnibus:               29.873        Durbin-Watson:           1.083 
    Prob(Omnibus):         0.000         Jarque-Bera (JB):        33.335
    Skew:                  0.461         Prob(JB):                0.000 
    Kurtosis:              3.475         Condition No.:           1615  
    ====================================================================
    * The condition number is large (2e+03). This might indicate
    strong multicollinearity or other numerical problems.


#### using patsy


```python
var = ['age', 'dep.cons', 'subst.cons', 'scz.cons', 'prof']
data_ = data.dropna(subset=['dur.interv']+var)

y = data_['dur.interv']
X = data_[var]

model = smf.ols(formula="Q('dur.interv') ~ age + Q('dep.cons') + Q('subst.cons') + Q(\
                'scz.cons') + prof", data=data_)
#patsy determined that elements of 'prof' were text strings, so it treated Prof as a categorical 
#variable. patsy’s default is also to include an intercept, so we automatically dropped one 
# of the Prof categories

est = model.fit()
print(est.summary2())
```

                          Results: Ordinary least squares
    ============================================================================
    Model:                  OLS                 Adj. R-squared:        0.052    
    Dependent Variable:     Q('dur.interv')     AIC:                   6504.1550
    Date:                   2018-10-19 21:12    BIC:                   6559.4834
    No. Observations:       743                 Log-Likelihood:        -3240.1  
    Df Model:               11                  F-statistic:           4.692    
    Df Residuals:           731                 Prob (F-statistic):    5.82e-07 
    R-squared:              0.066               Scale:                 365.04   
    ----------------------------------------------------------------------------
                                Coef.   Std.Err.    t    P>|t|   [0.025   0.975]
    ----------------------------------------------------------------------------
    Intercept                   62.7920  10.2078  6.1514 0.0000  42.7519 82.8321
    prof[T.artisan]            -11.4851   9.8294 -1.1685 0.2430 -30.7823  7.8120
    prof[T.autre]              -10.2875  10.3348 -0.9954 0.3199 -30.5770 10.0020
    prof[T.cadre]              -19.2964  10.3857 -1.8580 0.0636 -39.6857  1.0930
    prof[T.employe]            -13.5581   9.7634 -1.3887 0.1654 -32.7257  5.6096
    prof[T.ouvrier]            -14.0127   9.7211 -1.4415 0.1499 -33.0973  5.0719
    prof[T.prof.intermediaire] -13.0193   9.9691 -1.3060 0.1920 -32.5908  6.5522
    prof[T.sans emploi]        -14.2787   9.7178 -1.4693 0.1422 -33.3568  4.7995
    age                          0.2129   0.0588  3.6180 0.0003   0.0974  0.3284
    Q('dep.cons')                7.3679   1.4584  5.0521 0.0000   4.5048 10.2311
    Q('subst.cons')              5.3459   1.7690  3.0220 0.0026   1.8729  8.8188
    Q('scz.cons')                2.5044   2.5473  0.9831 0.3259  -2.4966  7.5054
    ----------------------------------------------------------------------------
    Omnibus:                 29.873           Durbin-Watson:              1.083 
    Prob(Omnibus):           0.000            Jarque-Bera (JB):           33.335
    Skew:                    0.461            Prob(JB):                   0.000 
    Kurtosis:                3.475            Condition No.:              1615  
    ============================================================================
    * The condition number is large (2e+03). This might indicate
    strong multicollinearity or other numerical problems.


#### relevel

Using pd.get_dummies it is possible to not drop_first and then drop the new reference.


```python
model = smf.ols(formula="Q('dur.interv') ~ age + Q('dep.cons') + Q('subst.cons') + Q(\
                'scz.cons') + C(prof, Treatment(reference='ouvrier'))", data=data_)
# If prof had been an integer variable that we wanted to treat explicitly as categorical, we 
# could have done so by using the C() operator, which by the way utilizes the Treatment relevel()

est = model.fit()
print(est.summary2())
```

                                           Results: Ordinary least squares
    ==============================================================================================================
    Model:                              OLS                            Adj. R-squared:                   0.052    
    Dependent Variable:                 Q('dur.interv')                AIC:                              6504.1550
    Date:                               2018-10-20 14:45               BIC:                              6559.4834
    No. Observations:                   743                            Log-Likelihood:                   -3240.1  
    Df Model:                           11                             F-statistic:                      4.692    
    Df Residuals:                       731                            Prob (F-statistic):               5.82e-07 
    R-squared:                          0.066                          Scale:                            365.04   
    --------------------------------------------------------------------------------------------------------------
                                                                   Coef.  Std.Err.    t    P>|t|   [0.025   0.975]
    --------------------------------------------------------------------------------------------------------------
    Intercept                                                     48.7793   2.8394 17.1796 0.0000  43.2050 54.3536
    C(prof, Treatment(reference='ouvrier'))[T.agriculteur]        14.0127   9.7211  1.4415 0.1499  -5.0719 33.0973
    C(prof, Treatment(reference='ouvrier'))[T.artisan]             2.5275   2.4899  1.0151 0.3104  -2.3606  7.4157
    C(prof, Treatment(reference='ouvrier'))[T.autre]               3.7252   3.9964  0.9321 0.3516  -4.1205 11.5710
    C(prof, Treatment(reference='ouvrier'))[T.cadre]              -5.2837   4.2557 -1.2416 0.2148 -13.6384  3.0711
    C(prof, Treatment(reference='ouvrier'))[T.employe]             0.4546   2.1266  0.2138 0.8308  -3.7203  4.6296
    C(prof, Treatment(reference='ouvrier'))[T.prof.intermediaire]  0.9934   2.9581  0.3358 0.7371  -4.8139  6.8008
    C(prof, Treatment(reference='ouvrier'))[T.sans emploi]        -0.2660   1.8773 -0.1417 0.8874  -3.9514  3.4195
    age                                                            0.2129   0.0588  3.6180 0.0003   0.0974  0.3284
    Q('dep.cons')                                                  7.3679   1.4584  5.0521 0.0000   4.5048 10.2311
    Q('subst.cons')                                                5.3459   1.7690  3.0220 0.0026   1.8729  8.8188
    Q('scz.cons')                                                  2.5044   2.5473  0.9831 0.3259  -2.4966  7.5054
    --------------------------------------------------------------------------------------------------------------
    Omnibus:                             29.873                      Durbin-Watson:                         1.083 
    Prob(Omnibus):                       0.000                       Jarque-Bera (JB):                      33.335
    Skew:                                0.461                       Prob(JB):                              0.000 
    Kurtosis:                            3.475                       Condition No.:                         574   
    ==============================================================================================================
