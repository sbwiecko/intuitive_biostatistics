
# Getting the data


```R
data <- read.csv2("../data/PlantGrowth.csv", sep = ',', dec='.')
```


```R
head(data)
```


<table>
<thead><tr><th scope=col>X</th><th scope=col>weight</th><th scope=col>group</th></tr></thead>
<tbody>
	<tr><td>1   </td><td>4.17</td><td>ctrl</td></tr>
	<tr><td>2   </td><td>5.58</td><td>ctrl</td></tr>
	<tr><td>3   </td><td>5.18</td><td>ctrl</td></tr>
	<tr><td>4   </td><td>6.11</td><td>ctrl</td></tr>
	<tr><td>5   </td><td>4.50</td><td>ctrl</td></tr>
	<tr><td>6   </td><td>4.61</td><td>ctrl</td></tr>
</tbody>
</table>
```R
data <- data[-c(1)] # drop the 'X' (index) column
```


```R
summary(data)
```


         weight       group   
     Min.   :3.590   ctrl:10  
     1st Qu.:4.550   trt1:10  
     Median :5.155   trt2:10  
     Mean   :5.073            
     3rd Qu.:5.530            
     Max.   :6.310            

```R
str(data)
```

    'data.frame':	30 obs. of  2 variables:
     $ weight: num  4.17 5.58 5.18 6.11 4.5 4.61 5.17 4.53 5.33 5.14 ...
     $ group : Factor w/ 3 levels "ctrl","trt1",..: 1 1 1 1 1 1 1 1 1 1 ...

```R
levels(data$group) # check if 'group' col is properly encoded
```


<ol class=list-inline>
	<li>'ctrl'</li>
	<li>'trt1'</li>
	<li>'trt2'</li>
</ol>
```R
boxplot(weight~group, data) # a little visualization
```


![png](output_7_0.png)


# Fitting the model


```R
fit = lm(weight~group, data) # setup ANOVA model
plot(fit)
```


![png](output_9_0.png)



![png](output_9_1.png)



![png](output_9_2.png)



![png](output_9_3.png)



```R
aov = anova(fit) 
aov # whou, works great!
```


<table>
<thead><tr><th></th><th scope=col>Df</th><th scope=col>Sum Sq</th><th scope=col>Mean Sq</th><th scope=col>F value</th><th scope=col>Pr(&gt;F)</th></tr></thead>
<tbody>
	<tr><th scope=row>group</th><td> 2        </td><td> 3.76634  </td><td>1.8831700 </td><td>4.846088  </td><td>0.01590996</td></tr>
	<tr><th scope=row>Residuals</th><td>27        </td><td>10.49209  </td><td>0.3885959 </td><td>      NA  </td><td>        NA</td></tr>
</tbody>
</table>

# Post-hoc analysis

## no adjustment


```R
pairwise.t.test(data$weight, data$group, p.adj = "none") # using pooled SD (different from individual t tests)
```

    	Pairwise comparisons using t tests with pooled SD 
    
    data:  data$weight and data$group 
    
         ctrl   trt1  
    trt1 0.1944 -     
    trt2 0.0877 0.0045
    
    P value adjustment method: none 

```R
pairwise.t.test(data$weight, data$group, p.adj='none', pool.sd = FALSE)
```

    	Pairwise comparisons using t tests with non-pooled SD 
    
    data:  data$weight and data$group 
    
         ctrl   trt1  
    trt1 0.2504 -     
    trt2 0.0479 0.0093
    
    P value adjustment method: none 


## Tukey


```R
library("lsmeans") # needed for ANOVA analysis and post-hoc tests
```

```R
lsmeans(fit, pairwise ~ group) # results of Tukey's test with P values
```


    $lsmeans
     group lsmean        SE df lower.CL upper.CL
     ctrl   5.032 0.1971284 27 4.627526 5.436474
     trt1   4.661 0.1971284 27 4.256526 5.065474
     trt2   5.526 0.1971284 27 5.121526 5.930474
    
    Confidence level used: 0.95 
    
    $contrasts
     contrast    estimate        SE df t.ratio p.value
     ctrl - trt1    0.371 0.2787816 27   1.331  0.3909
     ctrl - trt2   -0.494 0.2787816 27  -1.772  0.1980
     trt1 - trt2   -0.865 0.2787816 27  -3.103  0.0120
    
    P value adjustment: tukey method for comparing a family of 3 estimates 


## Bonferroni adjustment


```R
pairwise.t.test(data$weight, data$group, p.adj = "bonferroni")
```

    	Pairwise comparisons using t tests with pooled SD 
    
    data:  data$weight and data$group 
    
         ctrl  trt1 
    trt1 0.583 -    
    trt2 0.263 0.013
    
    P value adjustment method: bonferroni 


## Holm


```R
pairwise.t.test(data$weight, data$group, p.adj = "holm")
```

    	Pairwise comparisons using t tests with pooled SD 
    
    data:  data$weight and data$group 
    
         ctrl  trt1 
    trt1 0.194 -    
    trt2 0.175 0.013
    
    P value adjustment method: holm 


## Dunnett


```R
summary(glht(fit, linfct=mcp(group="Dunnett")))
```

    	 Simultaneous Tests for General Linear Hypotheses
    
    Multiple Comparisons of Means: Dunnett Contrasts

    Fit: lm(formula = weight ~ group, data = data)
    
    Linear Hypotheses:
                     Estimate Std. Error t value Pr(>|t|)
    trt1 - ctrl == 0  -0.3710     0.2788  -1.331    0.323
    trt2 - ctrl == 0   0.4940     0.2788   1.772    0.153
    (Adjusted p values reported -- single-step method)

