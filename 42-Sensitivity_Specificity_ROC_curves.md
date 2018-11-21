
# Sensitivity, Specificity and Receiver Operating Characteristic curves

## Statistical power

Type I (False Positive) and Type II (False Negative) error:  

&nbsp;| Reject H0 | Do no reject H0 | Total
---|---|---|---
H0 TRUE | A (Type I) | B | A+B
H0 FALSE | C | D (Type II) | C+D
TOTAL | A+C | B+D | A+B+C+D

The False Positive Report Probability (FPRP) or False Positive Rate FPR $= \frac{A}{A+C}$, and the significance level $\alpha = \frac{A}{A+B}$.

For example, with a prior probability that a drug X will work of 10%, i.e. the estimation that the effect will be observed, we can calculate the following:
- of the 1000 drugs screened we expect 100 (10%) will really work;
- of the 100 drugs that really work, we expect to obtain a statistically significant result in 80 (the experimental design, i.e. sample size, looking for large effect and little scatter, has 80% power;
- of the 900 drugs that are really ineffective, we expect to obtain statistically significant result in 5% ($\alpha=0.05$), therefore 45 false positive;
- of 1000 drugs tested, we expect to obtain a statistically significant difference in 80+45=125;
- the FPR equals 45/125=36%:

&nbsp;| effect  detected | effect not detected | Total
---|---|---|---
drug doesn't work | 45 | 855 | 900
drug works | 80 | 20 | 100
TOTAL | 125 | 875 | 1000

&nbsp;| effect  detected | effect not detected
---|---|---
true no effect | type I ($\alpha$)| B
true effect | $1-\beta$ | type II

## Accuracy of tests

False Negative (**FN**): result is _normal_ when patient _has disease_;  
False Positive (**FP**): result is _abnormal_ when patient really _has no disease_.

**Sensitivity** = fraction of all those _with disease_ who get a _positive_ test result=$\frac{Positive result}{All True Disease} = \frac{A}{A+C}$.  
**Specificity** = fraction of those _without disease_ who get a _negative_ test result=$\frac{Negative result}{All No Disease} = \frac{D}{B+D}$.

**Positive Predictive** value=$\frac{True Positives}{All Positive results} = \frac{A}{A+B}$.  
**Negative Predictive** value=$\frac{True Negatives}{All Negative results} = \frac{D}{C+D}$.

The last two values answer the question, `If the test is positive, what is the chance that the patient really has the disease?` (and vice-versa).

&nbsp;| True Disease | No Disease | Total
---|---|---|---
Abnormal (positive) test | A | B (false positive) | A+B
Normal (negative) test | C (false negative) | D | C+D
TOTAL | A+C | B+D | A+B+C+D

### Example 1 - Porphyria
With this test, 82% of patients with porphyria have an abnormal test result, and 3.7% of normal people have an abnormal test result.  
Sensitivity = 82%; Specificity = 100%-3.7% = 96.3%.  
This is a rare disease with prevalent of 1 for 10,000.

&nbsp;| Porphyria | No Porphyria | Total
---|---|---|---
Abnormal (positive) test | 82 | 36,996 | 37,078
Normal (negative) test | 18 | 962,904 | 962,922
TOTAL | 100 | 999,900 | 1,000,000

What is the probability that a patient with below threshold test results, i.e. abnormal, has porphyria?  
Positive Predictive value = 82/37078 = 0.22%, which corresponds to 1 in 500 of the people with positive test result have the disease. In other words, the other 499 of 500 positive tests are **false positives**.  
Of the 962,922 negative test results, only 18 are false negative, the predictive value of a negative test is 99.998%.

### Example 2 - HIV test
Its sensitivity = 99.9% and specificity = 99.6%.  
If the `prevalence of HIV is 10%` and we screen 1 million people:

&nbsp;| True HIV | No HIV | Total
---|---|---|---
Abnormal (positive) test | 99,900 | 3,600 | 103,500
Normal (negative) test | 100 | 896,400 | 896,500
TOTAL | 100,000 | 900,000 | 1,000,000

Of those 103,500 positive tests, 3.5% will be False Positive.

If the `prevalence of HIV is 0.1%` and we screen 1 million people:

&nbsp;| True HIV | No HIV | Total
---|---|---|---
Abnormal (positive) test | 999 | 3,996 | 4,995
Normal (negative) test | 1 | 995,004 | 995,005
TOTAL | 1,000 | 999,000 | 1,000,000

Of those 4,995 positive tests, 80% will be False Positive. **The fraction of the Positive tests that are False Positives depends on the prevalence of the disease in the population tested.**

### Bayes revisited
The _likelihood ratio_ is the probability of obtaining a positive test result in a patient with the disease divided by the probability of obtaining a positive test result in a patient without the disease: LLR $=\frac{sensitivity}{1 - specificity}$.  
In the porphyria example, LLR = 0.82 / (1 - 0.963) = 22.2. **A person with the condition is 22.2 times more likely to get a positive test result than a person without the condition.** In the HIV example, LLR = 249.75.  

The posttest odds are the odds that a patient has the disease, taking into account both the test result and prior knowledge about the patient:
$$
posttest odds = pretest odds \times \frac{sensitivity}{1 - specificity}
$$

| &nbsp;          | Pretest | &nbsp; | Posttest | &nbsp; |
| --------------- | ------- | ------ | -------- | ------ |
| Who was tested? | proba   | odds   | odds     | proba  |
| Random screen   | 0.0001  | 0.0001 | 0.0022   | 0.0022 |
| Sibling         | 0.50    | 1.000  | 22.2     | 0.957  |
|                 |         |        |          |        |
| 10% HIV         | 0.1     | 0.111  | 27.722   | 0.965  |
| 0.1% HIV        | 0.001   | 0.001  | 0.24975  | 0.20   |
with $odds=\frac{proba}{1-proba}$ and $proba=\frac{odds}{1+odds}$.

## ROC curves
It's often difficult to decide where to set the <u>threshold/cut-off</u> of a test that separates a clinical diagnosis of normal from one of abnormal.  
If `threshold is set high`, the **sensitivity will be low, but the specificity will be high**. Few of the positive tests will be False Positives, but many of the negative tests will be _False Negatives_.  
If `threshold is set low`, most individuals with the disease will be detected, but the test will mistakenly diagnose many normal individuals as abnormal. The **sensitivity will be high, but the specificity will be low**. Few of the positive tests will be False Positives, but many of the positive tests will be _False Positives_.  

Each point of the ROC curve shows the sensitivity and specificity for one possible threshold value for deciding when a test is abnormal. Although the consequence of FP and FN are not comparable, the **best threshold** may be the one that corresponds to the point on the ROC curve that is **the closest to the upper-left corner**.

### AUC and classifier

![ROC curve](C:\Python_DataScience\biostatistics\Intuitive_biostatics_4th\rest_of_statistics\custom_thresh_mod0-1.svg)
At the bottom-left extreme, the test never ever returns a positive test (sensitivity = 0%), even for controls (specificity = 100%).  At the upper-right, the test always returns a positive test (sensitivity = 100%), even for controls (specificity = 0%).

The confusion matrix used in fine-tuning regression models loos like:

&nbsp;| Spam predict. | Not spam predict.
---|---|---
Real spam | TP | FN
Wrong spam | FP | TN

accuracy = $\frac{TP+TN}{TP+TN+FP+FN}$  
precision = $\frac{TP}{TP+FP}$  
recall = $\frac{TP}{TP+FN}$  
F1 score = $2\times \frac{precision \times recall}{precision + recall}$

TPR = $\frac{TP}{TP+FN}$  
FPR = $\frac{FP}{TN+FP}$

The Area Under the Curve (AUC) is an indicator of the overall performance of the classifier, with:
- AUC = 1.0, perfect/ideal classifier;
- AUC > 0.5, good classifier;
- AUC = 0.5, random classifier;
- AUC < 0.5, classifier performing worse than random;
- AUC = 0.0, incorrect classifier.
