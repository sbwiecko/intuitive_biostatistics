
# Meta-analysis (forest plot)
In a fixed effect analysis we assume that all the included studies share a common effect size, μ. The observed effects will be distributed about μ, with a variance σ² that depends primarily on the sample size for each study. We use the data from https://www.meta-analysis.com/downloads/Meta%20Analysis%20Fixed%20vs%20Random%20effects.pdf

## foresplot lib


```R
library('forestplot')
```

    Loading required package: grid
    Loading required package: magrittr
    Loading required package: checkmate


We use the data extracted from Table 43.1 on page 455.


```R
data = structure(list(
mean = c(NA, 2.5, 1.4, 1.1, 1.2, .8, .85, 1.05, NA, 1.2),
lower= c(NA, 1.7, .8, .8, .85, .5, .4, .6, NA, .9),
upper= c(NA, 4.8, 2.2, 1.8, 1.9, 1.4, 1.6, 2, NA, 1.6)),
                .Names = c('mean', 'lower', 'upper'),
                row.names = c('NA', -9L),
                class = 'data.frame')
```


```R
tabletext = cbind(
c("Trial ID", "14", "15", "46", "47", "50", "45", "49", NA, "Total"),
c("OR", "2.5", "1.4", "1.1", "1.2", "0.8", "0.85", "1.05", NA, "1.2"))
```


```R
forestplot(tabletext, data, is.summary = c(TRUE, TRUE, rep(FALSE,8, TRUE)),
          clip=c(.1, 2.5))
```


    Error in forestplot(tabletext, data, is.summary = c(TRUE, TRUE, rep(FALSE, : could not find function "forestplot"
    Traceback:

This package doesn't work well in a jupyter notebook and should be reserved for R studio.

## metaplot lib


```R
library('rmeta')
options(repr.plot.width = 7, repr.plot.height = 5) # plot size
options(jupyter.plot_mimetypes = "image/svg+xml")  # figure resolution
```

### data sample from the package


```R
data(catheter)
```


```R
catheter
```


<table>
<thead><tr><th scope=col>Name</th><th scope=col>n.trt</th><th scope=col>n.ctrl</th><th scope=col>col.trt</th><th scope=col>col.ctrl</th><th scope=col>inf.trt</th><th scope=col>inf.ctrl</th></tr></thead>
<tbody>
	<tr><td>Ciresi    </td><td>124       </td><td>127       </td><td>15        </td><td>21        </td><td>13        </td><td>14        </td></tr>
	<tr><td>George    </td><td> 44       </td><td> 35       </td><td>10        </td><td>25        </td><td> 1        </td><td> 3        </td></tr>
	<tr><td>Hannan    </td><td> 68       </td><td> 60       </td><td>22        </td><td>22        </td><td> 5        </td><td> 7        </td></tr>
	<tr><td>Heard     </td><td>151       </td><td>157       </td><td>60        </td><td>82        </td><td> 5        </td><td> 6        </td></tr>
	<tr><td>vanHeerden</td><td> 28       </td><td> 26       </td><td> 4        </td><td>10        </td><td>NA        </td><td>NA        </td></tr>
	<tr><td>Maki      </td><td>208       </td><td>195       </td><td>28        </td><td>47        </td><td> 2        </td><td> 9        </td></tr>
	<tr><td>Bach(a)   </td><td> 14       </td><td> 12       </td><td> 0        </td><td> 4        </td><td>NA        </td><td>NA        </td></tr>
	<tr><td>Ramsay    </td><td>199       </td><td>189       </td><td>45        </td><td>63        </td><td> 1        </td><td> 4        </td></tr>
	<tr><td>Appavu    </td><td> 12       </td><td>  7       </td><td> 1        </td><td> 1        </td><td>NA        </td><td>NA        </td></tr>
	<tr><td>Trazzera  </td><td>123       </td><td> 99       </td><td>16        </td><td>24        </td><td> 4        </td><td> 5        </td></tr>
	<tr><td>Collins   </td><td> 98       </td><td>139       </td><td> 2        </td><td>25        </td><td> 1        </td><td> 4        </td></tr>
	<tr><td>Bach(b)   </td><td>116       </td><td>117       </td><td> 2        </td><td>16        </td><td> 0        </td><td> 3        </td></tr>
	<tr><td>Tennenberg</td><td>137       </td><td>145       </td><td> 8        </td><td>32        </td><td> 5        </td><td> 9        </td></tr>
	<tr><td>Pemberton </td><td> 32       </td><td> 40       </td><td>NA        </td><td>NA        </td><td> 2        </td><td> 3        </td></tr>
	<tr><td>Logghe    </td><td>338       </td><td>342       </td><td>NA        </td><td>NA        </td><td>17        </td><td>15        </td></tr>
</tbody>
</table>
```R
write.csv(catheter, file="../data/catheter.csv", )
```

Fixed effects (Mantel-Haenszel) meta-analysis (2x2) with `n.trt` and `n.ctrl` the number of subjects , and `col.trt` and `col.ctrl` the number of events in the _treatment_ and _control_ groups respectively.


```R
help(meta.MH)
```


```R
a <- meta.MH(n.trt, n.ctrl, col.trt, col.ctrl, data=catheter,
             names=Name, subset=c(13,6,5,3,7,12,4,11,1,8,10,2))
```


```R
summary(a)
```


    Fixed effects ( Mantel-Haenszel ) meta-analysis
    Call: meta.MH(ntrt = n.trt, nctrl = n.ctrl, ptrt = col.trt, pctrl = col.ctrl, 
        names = Name, data = catheter, subset = c(13, 6, 5, 3, 7, 
            12, 4, 11, 1, 8, 10, 2))
    ------------------------------------
                 OR (lower  95% upper)
    Tennenberg 0.22    0.10       0.49
    Maki       0.49    0.29       0.82
    vanHeerden 0.27    0.07       1.00
    Hannan     0.83    0.40       1.72
    Bach(a)     NaN    0.00        NaN
    Bach(b)    0.11    0.02       0.49
    Heard      0.60    0.38       0.95
    Collins    0.10    0.02       0.41
    Ciresi     0.69    0.34       1.42
    Ramsay     0.58    0.37       0.92
    Trazzera   0.47    0.23       0.94
    George     0.12    0.04       0.33
    ------------------------------------
    Mantel-Haenszel OR =0.44 95% CI ( 0.36,0.54 )
    Test for heterogeneity: X^2( 10 ) = 25.36 ( p-value 0.0047 )

```R
print(a)
```

    Fixed effects ( Mantel-Haenszel ) Meta-Analysis
    Call: meta.MH(ntrt = n.trt, nctrl = n.ctrl, ptrt = col.trt, pctrl = col.ctrl, 
        names = Name, data = catheter, subset = c(13, 6, 5, 3, 7, 
            12, 4, 11, 1, 8, 10, 2))
    Mantel-Haenszel OR =0.44    95% CI ( 0.36, 0.54 )
    Test for heterogeneity: X^2( 10 ) = 25.36 ( p-value 0.0047 )

```R
str(a)
```

    List of 10
     $ logOR     : num [1:12] -1.519 -0.714 -1.322 -0.191 -Inf ...
     $ selogOR   : num [1:12] 0.416 0.263 0.674 0.373 Inf ...
     $ logMH     : num -0.824
     $ selogMH   : num 0.102
     $ MHtest    : num [1:2] 6.70e+01 2.22e-16
     $ het       : num [1:3] 25.3627 10 0.0047
     $ call      : language meta.MH(ntrt = n.trt, nctrl = n.ctrl, ptrt = col.trt, pctrl = col.ctrl,      names = Name, data = catheter, subse| __truncated__ ...
     $ names     : chr [1:12] "Tennenberg" "Maki" "vanHeerden" "Hannan" ...
     $ conf.level: num 0.95
     $ statistic : chr "OR"
     - attr(*, "class")= chr [1:2] "meta.MH.OR" "meta.MH"

```R
plot(a, summlabel='Total',)
```


![svg](output_20_0.svg)

```R
funnelplot(a)
```


![svg](output_21_0.svg)


A funnel plot is a scatterplot of treatment effect against a measure of study precision. It is used primarily as a visual aid for detecting bias or systematic heterogeneity. A symmetric inverted funnel shape arises from a ‘well-behaved’ data set, in which publication bias is unlikely. An asymmetric funnel indicates a relationship between treatment effect estimate and study precision. This suggests the possibility of either publication bias or a systematic difference between studies of higher and lower precision (typically ‘small study effects’). Asymmetry can also arise from use of an inappropriate effect measure. Whatever the cause, an asymmetric funnel plot leads to doubts over the appropriateness of a simple meta-analysis and suggests that there needs to be investigation of possible causes.


```R
metaplot(a$logOR, a$selogOR, nn=a$selogOR^-2, a$names, # equiv 1/a$selogOR^2
         summn=a$logMH, sumse=a$selogMH, sumnn=a$selogMH^-2,
         logeffect=TRUE, colors=meta.colors(box="magenta",
             lines="blue", zero="red", summary="orange",
             text="forestgreen"))
```


![svg](output_23_0.svg)


### step-by-step (Woolf method/inverse variance)

We assign **weights** based on the inverse of the variance rather than sample size. The inverse variance is roughly proportional to sample size, but is a more nuanced measure, and serves to minimize the variance of the combined effect: $w_i=\frac{1}{v_i}$ with $v_i$ the within-study variance for studi _i_.  
The weighted mean is then computed as the sum of the products $w_iT_i$ divided by the sum of the weights:
$$
\bar{T.}=\frac{\sum_{i=1}^kw_iT_i}{\sum_{i=1}^kw_i}
$$
The variance of the combined effect is $v.=\frac{1}{\sum_{i=1}^kw_i}$, and $SE(\bar{T.})=\sqrt{v_i}$.  
The 95% confidence interval for the combined effect would be $\bar{T.} \pm 1.96*SE(\bar{T.})$  
And $Z=\frac{\bar{T.}}{SE(\bar{T.})}$, so for a two-tailed test: $p=2[1-(\phi(|Z|))]$ with $\phi$ the standard normal cumulative function.

**Reminder**: To calculate the confidence interval, we use the log odds ratio, log(or) = log(a*d/b*c), and calculate its standard error:
$ SE(\log{OR})=\sqrt{\frac{1}{n_{00}}+ \frac{1}{n_{01}} + \frac{1}{n_{10}} + \frac{1}{n_{11}}}$ and then ci for the log(OR) is $\log{OR}\pm Z\alpha/2*SE(\log{OR})$


```R
# for example in the Tennenberg study
se_log_or = (1/8 + 1/(137-8) + 1/32 + 1/(145-32))**.5
se_log_or
```


0.415754128670564

```R
v_log_or = se_log_or^2
v_log_or
```


0.17285149550662

```R
w = 1/v_log_or
w
```


5.78531297672053


And so on for the other studies included in the analysis:


```R
# we drop the study with OR = NA
a <- meta.MH(n.trt, n.ctrl, col.trt, col.ctrl, data=catheter,
             names=Name, subset=c(13,6,5,3,12,4,11,1,8,10,2))
```


```R
weights = a$selogOR^-2 # actually the nn option of forest plot metaplot function
```


```R
weights
```


<ol class=list-inline>
	<li>5.78531297672053</li>
	<li>14.4293494366262</li>
	<li>2.20183486238532</li>
	<li>7.19610778443114</li>
	<li>1.7206582855435</li>
	<li>18.802600413244</li>
	<li>1.78830542374211</li>
	<li>7.52479484937013</li>
	<li>19.0384615384615</li>
	<li>7.8835881377786</li>
	<li>3.7117903930131</li>
</ol>
```R
sum(weights)
```


90.0828041013161

```R
means = a$logOR
means
```


<ol class=list-inline>
	<li>-1.51871894676922</li>
	<li>-0.71368766866095</li>
	<li>-1.32175583998232</li>
	<li>-0.191055236762709</li>
	<li>-2.20051947323307</li>
	<li>-0.505746078022692</li>
	<li>-2.3538783873816</li>
	<li>-0.364381024738289</li>
	<li>-0.537142932083364</li>
	<li>-0.76080582903376</li>
	<li>-2.14006616349627</li>
</ol>
```R
weighted_means = means * weights
weighted_means
```


<ol class=list-inline>
	<li>-8.78626443073532</li>
	<li>-10.29804875972</li>
	<li>-2.91028808803446</li>
	<li>-1.37485407652447</li>
	<li>-3.7863420641183</li>
	<li>-9.50934141562599</li>
	<li>-4.20945348698383</li>
	<li>-2.74189245815889</li>
	<li>-10.2263750531256</li>
	<li>-5.99787980892336</li>
	<li>-7.94347702607786</li>
</ol>
```R
# here we compute the summary OR
t_bar = sum(weighted_means) / sum(weights)
t_bar
exp(t_bar)
```


-0.752465660280636

0.471203291986275

```R
exp(log(100))
```


100

```R
v_bar = 1 / sum(weights)
v_bar
```


0.0111008977792843

```R
se_bar = v_bar**.5
se_bar
```


0.105360798114309

```R
LOL = t_bar - 1.96 * se_bar
UPL = t_bar + 1.96 * se_bar
exp(LOL)
exp(UPL)
```


0.383286386196564

0.579286273592915

```R
Z = t_bar / se_bar
Z
```


-7.14179916769671

```R
p = 2*pnorm(-abs(Z))
p
```


9.21171237474655e-13

```R
summary(a)
```


    Fixed effects ( Mantel-Haenszel ) meta-analysis
    Call: meta.MH(ntrt = n.trt, nctrl = n.ctrl, ptrt = col.trt, pctrl = col.ctrl, 
        names = Name, data = catheter, subset = c(13, 6, 5, 3, 12, 
            4, 11, 1, 8, 10, 2))
    ------------------------------------
                 OR (lower  95% upper)
    Tennenberg 0.22    0.10       0.49
    Maki       0.49    0.29       0.82
    vanHeerden 0.27    0.07       1.00
    Hannan     0.83    0.40       1.72
    Bach(b)    0.11    0.02       0.49
    Heard      0.60    0.38       0.95
    Collins    0.10    0.02       0.41
    Ciresi     0.69    0.34       1.42
    Ramsay     0.58    0.37       0.92
    Trazzera   0.47    0.23       0.94
    George     0.12    0.04       0.33
    ------------------------------------
    Mantel-Haenszel OR =0.45 95% CI ( 0.36,0.54 )
    Test for heterogeneity: X^2( 10 ) = 25.19 ( p-value 0.005 )

The results are not totally identical. Alternative methods, such **Woolf and inverse variance**, can be used to estimate the pooled odds ratio with fixed effects but the **Mantel-Haenszel** method is generally the _most robust_ (see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5287121/).  
A confidence interval for the Mantel-Haenszel odds ratio in StatsDirect is calculated using the Robins, Breslow and Greenland variance formula (Robins et al., 1986) or by the method of Sato (1990) if the estimate of the odds ratio can not be determined. A chi-square test statistic is given with its associated probability that the pooled odds ratio is equal to one.  
There are at least two other ways to do a fixed effects meta-analysis of binary data. **Peto's method** is a computationally simpler _approximation_ to the Mantel-Haenszel approach. It is also possible to weight the individual odds ratios according to their estimated variances. The Mantel-Haenszel method is superior if there are trials with small numbers of events (less than 5 or so in either group)