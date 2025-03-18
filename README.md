# CONTINA (Conformal Traffic Intervals with Adaptation): Confidence Interval for Traffic Demand Prediction with Coverage Guarantee 
## Abstract
Accurate short-term traffic demand prediction is critical for the operation of traffic system. Besides point estimation, confidence interval of the prediction is also important because many models about traffic operation, such as shared bike rebalancing and taxi dispatching, take the uncertainty of future demand into account and require confidence interval as the input. However, existing methods require strict assumptions such as unchanging traffic pattern and correct model specification ensure that enough coverage. Therefore, the confidence intervals provided could be invalid, especially in a changing traffic environment. To fill this gap, we propose a simple but efficient method, CONTINA (Conformal Traffic Intervals with Adaptation). The main idea of this method is collecting errors of interval during deployment, and the interval will be widened in the next step if the errors are larger, and shorted otherwise. Besides, we theoretically prove that the coverage of the confidence intervals provided by our method can converge to the target coverage level. Experiments across four real-world datasets and prediction models demonstrate that our method can provide valid confidence interval with shorter length. Our method is able to help traffic management personnel develop more reasonable operation plan in practice. 

## Main results

| Dataset |  time  | metric |   QR   | MC-dropout | boostrap |   MIS  | DESQRUQ | UATGCN | ProbGNN | QuanTraffic |   CP   |   ACI  |     QCP    |  DtACI |   CONTINA  |
|---------|:------:|:------:|:------:|:----------:|:--------:|:------:|:-------:|:------:|:-------:|:-----------:|:------:|:------:|:----------:|:------:|:----------:|
| NYCbike | 1month |   cov  |  89.6% |    54.8%   |   30.8%  |  88.7% |  91.7%  |  91.7% |  93.1%  |    91.3%    |  89.2% |  89.8% |    90.0%   |  89.6% |    89.6%   |
|         |        | length | 0.265  |   0.218    |  0.084   | 0.278  |  0.284  | 0.284  |  0.306  |    0.288    | 0.285  | 0.300  | **0.266** | 0.291  |   0.276    |
|         |        |  minRC |  81.1% |    36.4%   |   20.4%  |  80.1% |  86.6%  |  85.0% |  87.8%  |    86.4%    |  84.3% |  88.8% |    85.3%   |  87.3% |    88.7%   |
|         | 2month |   cov  |  90.0% |    55.8%   |   30.7%  |  89.0% |  91.9%  |  92.3% |  93.4%  |    92.0%    |  89.3% |  90.1% |    90.5%   |  90.0% |    89.8%   |
|         |        | length | 0.270  |   0.227    |  0.087   | 0.283  |  0.283  | 0.291  |  0.315  |    0.285    | 0.289  | 0.304  |   0.271    | 0.298  | **0.275** |
|         |        |  minRC |  80.9% |    37.6%   |   18.6%  |  79.6% |  88.0%  |  86.6% |  87.9%  |    86.4%    |  84.8% |  88.9% |    84.1%   |  85.9% |    89.3%   |
|         | 3month |   cov  |  89.3% |    56.2%   |   30.6%  |  88.6% |  91.2%  |  91.9% |  93.0%  |    91.4%    |  87.6% |  89.7% |    89.9%   |  89.2% |    89.8%   |
|         |        | length | 0.289  |   0.250    |  0.094   | 0.301  |  0.296  | 0.315  |  0.340  |    0.306    | 0.295  | 0.322  |   0.290    | 0.321  | **0.298** |
|         |        |  minRC |  80.5% |    38.1%   |   18.1%  |  80.9% |  84.5%  |  85.0% |  86.5%  |    84.8%    |  82.3% |  88.6% |    82.4%   |  86.8% |    89.2%   |
|         | 4month |   cov  |  87.7% |    57.1%   |   30.5%  |  87.6% |  89.8%  |  90.3% |  91.3%  |    89.8%    |  82.9% |  90.0% |    88.4%   |  89.3% |    89.7%   |
|         |        | length | 0.343  |   0.323    |  0.117   | 0.355  |  0.356  | 0.386  |  0.419  |    0.360    | 0.312  | 0.411  |   0.345    | 0.400  | **0.363** |
|         |        |  minRC |  73.1% |    33.4%   |   19.9%  |  72.8% |  78.8%  |  78.4% |  80.8%  |    76.8%    |  69.1% |  88.8% |    74.1%   |  88.4% |    89.1%   |
|         |   AVG  |   cov  |  89.2% |    56.0%   |   30.6%  |  88.5% |  91.2%  |  91.5% |  92.7%  |    91.1%    |  87.3% |  89.9% |    89.7%   |  89.5% |    89.7%   |
|         |        | length | 0.292  |   0.254    |  0.095   | 0.304  |  0.305  | 0.319  |  0.345  |    0.310    | 0.295  | 0.334  |   0.293    | 0.327  | **0.303** |
|         |        |  minRC |  78.9% |    36.4%   |   19.2%  |  78.3% |  84.5%  |  83.8% |  85.8%  |    83.6%    |  80.1% |  88.8% |    81.5%   |  87.1% |    89.1%   |
| NYCtaxi | 1month |   cov  |  87.9% |    64.5%   |   47.4%  |  88.9% |  92.4%  |  89.6% |  94.3%  |    82.6%    |  91.0% |  89.9% |    89.5%   |  90.2% |    89.7%   |
|         |        | length | 0.241  |   0.148    |  0.096   | 0.243  |  0.262  | 0.264  |  0.300  |    0.294    | 0.283  | 0.270  |   0.258    | 0.271  | **0.237** |
|         |        |  minRC |  77.3% |    31.8%   |   34.2%  |  77.9% |  82.9%  |  75.1% |  83.0%  |    59.8%    |  78.4% |  88.6% |    80.4%   |  87.1% |    89.0%   |
|         | 2month |   cov  |  87.3% |    64.9%   |   46.5%  |  88.7% |  92.0%  |  89.0% |  93.8%  |    82.3%    |  89.7% |  89.9% |    89.0%   |  89.7% |    89.9%   |
|         |        | length | 0.248  |   0.141    |  0.099   | 0.252  |  0.270  | 0.275  |  0.312  |    0.301    | 0.281  | 0.284  |   0.265    | 0.280  | **0.248** |
|         |        |  minRC |  75.4% |    31.9%   |   34.1%  |  76.5% |  63.4%  |  73.6% |  80.8%  |    59.7%    |  76.7% |  88.6% |    78.9%   |  87.4% |    89.3%   |
|         | 3month |   cov  |  86.6% |    65.2%   |   46.4%  |  88.7% |  91.8%  |  88.9% |  93.7%  |    81.8%    |  89.6% |  90.1% |    88.5%   |  89.9% |    89.9%   |
|         |        | length | 0.251  |   0.138    |  0.098   | 0.252  |  0.272  | 0.275  |  0.311  |    0.304    | 0.280  | 0.286  |   0.268    | 0.280  | **0.248** |
|         |        |  minRC |  72.6% |    26.3%   |   33.9%  |  77.0% |  77.1%  |  71.2% |  78.0%  |    56.9%    |  73.9% |  88.5% |    67.0%   |  85.9% |    89.2%   |
|         | 4month |   cov  |  83.1% |    65.6%   |   46.3%  |  88.9% |  92.0%  |  89.0% |  93.7%  |    80.9%    |  89.8% |  90.0% |    87.5%   |  90.3% |    89.9%   |
|         |        | length | 0.259  |   0.214    |  0.098   | 0.250  |  0.274  | 0.271  |  0.312  |    0.312    | 0.281  | 0.290  |   0.275    | 0.285  | **0.252** |
|         |        |  minRC |  74.2% |    32.0%   |   32.9%  |  79.5% |  80.3%  |  73.8% |  77.7%  |    55.8%    |  71.9% |  88.9% |    78.5%   |  86.7% |    89.4%   |
|         |   AVG  |   cov  |  86.2% |    65.1%   |   46.7%  |  88.8% |  92.1%  |  89.1% |  93.9%  |    81.9%    |  90.0% |  90.0% |    88.6%   |  90.0% |    89.8%   |
|         |        | length | 0.250  |   0.160    |  0.098   | 0.249  |  0.270  | 0.271  |  0.309  |    0.303    | 0.281  | 0.283  |   0.267    | 0.279  | **0.246** |
|         |        |  minRC |  74.9% |    30.5%   |   33.8%  |  77.7% |  75.9%  |  73.4% |  79.9%  |    58.1%    |  75.2% |  88.6% |    76.2%   |  86.8% |    89.2%   |
| CHIbike | 1month |   cov  |  89.5% |    29.8%   |   22.9%  |  89.8% |  93.5%  |  92.1% |  93.6%  |    87.0%    |  90.0% |  90.2% |    89.9%   |  90.2% |    89.7%   |
|         |        | length | 0.514  |   0.188    |  0.107   | 0.513  |  0.531  | 0.553  |  0.593  |    0.527    | 0.623  | 0.638  |   0.514    | 0.624  | **0.521** |
|         |        |  minRC |  82.4% |    19.0%   |   9.3%   |  83.3% |  90.4%  |  86.9% |  90.3%  |    74.9%    |  86.2% |  88.3% |    83.2%   |  86.4% |    89.0%   |
|         | 2month |   cov  |  89.1% |    31.4%   |   24.0%  |  89.4% |  93.1%  |  92.1% |  93.6%  |    86.8%    |  88.1% |  89.5% |    89.5%   |  89.1% |    89.8%   |
|         |        | length | 0.553  |   0.215    |  0.120   | 0.554  |  0.572  | 0.593  |  0.637  |    0.566    | 0.624  | 0.696  |   0.553    | 0.655  |   **0.563**    |
|         |        |  minRC |  81.4% |    20.1%   |   10.7%  |  84.0% |  89.8%  |  87.3% |  90.2%  |    75.9%    |  83.9% |  87.9% |    82.8%   |  86.5% |    89.2%   |
|         | 3month |   cov  |  88.7% |    32.3%   |   23.8%  |  89.1% |  92.3%  |  91.4% |  93.1%  |    86.8%    |  86.6% |  90.2% |    88.9%   |  90.0% |    89.8%   |
|         |        | length | 0.613  |   0.251    |  0.134   | 0.617  |  0.635  | 0.657  |  0.706  |    0.625    | 0.655  | 0.789  |   0.613    | 0.765  |   **0.629**    |
|         |        |  minRC |  81.7% |    21.0%   |   10.8%  |  83.8% |  89.0%  |  87.4% |  90.4%  |    77.4%    |  82.2% |  89.4% |    82.9%   |  87.8% |    89.4%   |
|         | 4month |   cov  |  87.5% |    36.0%   |   22.8%  |  88.1% |  90.9%  |  90.5% |  92.4%  |    86.4%    |  79.8% |  90.0% |    87.7%   |  89.4% |    89.9%   |
|         |        | length | 0.835  |   0.405    |  0.188   | 0.844  |  0.889  | 0.906  |  0.964  |    0.843    | 0.713  | 1.124  |   0.835    | 1.059  |   **0.883**    |
|         |        |  minRC |  83.0% |    24.2%   |   12.6%  |  79.5% |  87.4%  |  86.3% |  89.5%  |    78.6%    |  71.0% |  88.6% |    83.0%   |  86.7% |    89.1%   |
|         |   AVG  |   cov  |  88.7% |    32.4%   |   23.4%  |  89.1% |  92.5%  |  91.5% |  93.2%  |    86.8%    |  86.1% |  90.0% |    89.0%   |  89.7% |    89.8%   |
|         |        | length | 0.629  |   0.265    |  0.137   | 0.632  |  0.657  | 0.677  |  0.725  |    0.640    | 0.654  | 0.812  |   0.629    | 0.776  |   **0.649**    |
|         |        |  minRC |  82.1% |    21.1%   |   10.8%  |  82.6% |  89.2%  |  87.0% |  90.1%  |    76.7%    |  80.8% |  88.6% |    83.0%   |  86.9% |    89.2%   |
| CHItaxi | 1month |   cov  |  90.3% |    46.3%   |   44.4%  |  91.3% |  93.2%  |  92.8% |  94.4%  |    89.2%    |  92.0% |  90.0% |    90.9%   |  90.2% |    89.6%   |
|         |        | length | 0.208  |   0.269    |  0.081   | 0.220  |  0.222  | 0.279  |  0.307  |    0.227    | 0.297  | 0.273  |   0.210    | 0.265  | **0.219**  |
|         |        |  minRC |  83.7% |    15.0%   |   27.3%  |  85.6% |  87.2%  |  87.8% |  91.0%  |    86.9%    |  89.6% |  87.5% |    83.4%   |  87.5% |    89.0%   |
|         | 2month |   cov  |  90.6% |    47.9%   |   44.2%  |  91.6% |  93.4%  |  93.1% |  94.5%  |    89.5%    |  90.8% |  90.0% |    91.1%   |  90.0% |    89.8%   |
|         |        | length | 0.221  |   0.297    |  0.085   | 0.234  |  0.235  | 0.300  |  0.330  |    0.240    | 0.283  | 0.282  |   0.222    | 0.264  | **0.228** |
|         |        |  minRC |  84.7% |    15.5%   |   27.1%  |  86.2% |  88.4%  |  88.4% |  91.4%  |    86.6%    |  88.3% |  87.6% |    84.3%   |  87.5% |    89.4%   |
|         | 3month |   cov  |  89.8% |    49.0%   |   43.3%  |  90.8% |  92.6%  |  92.4% |  93.9%  |    89.6%    |  88.6% |  89.7% |    90.4%   |  89.7% |    89.8%   |
|         |        | length | 0.241  |   0.333    |  0.096   | 0.257  |  0.258  | 0.331  |  0.365  |    0.262    | 0.283  | 0.326  |   0.243    | 0.294  | **0.256** |
|         |        |  minRC |  83.3% |    16.9%   |   25.7%  |  84.7% |  86.9%  |  86.5% |  89.3%  |    86.5%    |  84.0% |  86.9% |    82.9%   |  86.5% |    89.4%   |
|         | 4month |   cov  |  90.1% |    48.7%   |   42.8%  |  91.1% |  92.8%  |  92.5% |  94.3%  |    89.3%    |  89.4% |  90.1% |    90.6%   |  90.2% |    89.9%   |
|         |        | length | 0.237  |   0.319    |  0.091   | 0.252  |  0.252  | 0.319  |  0.351  |    0.257    | 0.288  | 0.320  |   0.238    | 0.302  | **0.251** |
|         |        |  minRC |  82.1% |    20.4%   |   28.3%  |  84.7% |  84.7%  |  86.3% |  90.3%  |    84.2%    |  82.3% |  88.2% |    82.0%   |  87.0% |    89.4%   |
|         |   AVG  |   cov  |  90.2% |    48.0%   |   43.7%  |  91.2% |  93.0%  |  92.7% |  94.3%  |    89.4%    |  90.2% |  89.9% |    90.8%   |  90.0% |    89.8%   |
|         |        | length | 0.227  |   0.305    |  0.088   | 0.241  |  0.242  | 0.307  |  0.338  |    0.247    | 0.288  | 0.300  |   0.228    | 0.281  | **0.238** |
|         |        |  minRC |  83.4% |    17.0%   |   27.1%  |  85.3% |  86.8%  |  87.3% |  90.5%  |    86.0%    |  86.1% |  87.5% |    83.1%   |  87.1% |    89.3%   |
|   1st   |        |        |    0   |      0     |     0    |    0   |    0    |    0   |    0    |      0      |    0   |    0   |      1     |    0   |   **19**   |

## How to reproduce this algorithm?

Just install pytorch and run AQCI.ipynb is OK. Pretained model and dataset can be downloaded.
