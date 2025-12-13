# Comparison of Gap Filling Methods for OpenET



## Gap Filling Methods

| Name | Description |
| - | - |
| interpolate | Simple linear interpolation |
| climo_mean | Monthly climatology computed using the mean |
| climo_median | Monthly climatology computed using the median |
| conor | Conor's method |
| interp_clim_a | Simple mean of the interpolated and mean climatology |
| interp_clim_b | Weighted mean of the interpolated value and climatology, where the climatology is weighted based on the number of months used to compute the climatology |
| interp_clim_c | Simple mean of the interpolated and median climatology |
| whit_a_0p50 | Whittaker-Eilers smoothing with a lambda term of 0.5 |
| whit_a_0p20 | Whittaker-Eilers smoothing with a lambda term of 0.2 |
| whit_a_0p10 | Whittaker-Eilers smoothing with a lambda term of 0.1 |
| whit_a_0p05 | Whittaker-Eilers smoothing with a lambda term of 0.05 |
| whit_a_0p01 | Whittaker-Eilers smoothing with a lambda term of 0.01 |

## Summary Statistics

| Name | Description |
| - | - |
| rmse | Root mean squared error |
| mae | Mean absolute error |
| mbe | Mean bias error |
| m | Slope of the best fit regression line |
| b | Intercept of the best fit regression line |
| r2 | Coefficient of determination (R2) of the best fit regression line |
| n | Number of test points |

## Output Example

Example of summary statistics output for all sampling points:

```
Randomly drop one datapoint during the year
            method     rmse      mae      mbe        m        b       r2        n
       interpolate   0.1182   0.0881   0.0014   0.8644   0.0862   0.8619   136547
        climo_mean   0.1225   0.0906   0.0015   0.8515   0.0943   0.8516   136547
      climo_median   0.1266   0.0873   0.0007   0.8725   0.0804   0.8426   136547
             conor   0.1005   0.0753  -0.0002   0.9208   0.0493   0.9005   136547
     interp_clim_a   0.1055   0.0795   0.0014   0.8580   0.0903   0.8912   136547
     interp_clim_b   0.1063   0.0798   0.0018   0.8576   0.0908   0.8895   136547
     interp_clim_c   0.1063   0.0784   0.0010   0.8685   0.0833   0.8887   136547
       whit_a_0p50   0.1197   0.0892   0.0010   0.8765   0.0782   0.8586   136547
       whit_a_0p20   0.1187   0.0879   0.0007   0.8927   0.0678   0.8619   136547
       whit_a_0p10   0.1191   0.0882   0.0005   0.9005   0.0627   0.8614   136547
       whit_a_0p05   0.1201   0.0888   0.0004   0.9055   0.0595   0.8597   136547
       whit_a_0p01   0.1221   0.0903   0.0002   0.9104   0.0562   0.8561   136547
```
