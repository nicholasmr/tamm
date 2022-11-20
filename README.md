## Radio-wave transfer matrix model for glacier ice
Model of electromagnetic plane-wave propagation through a column of polycrystalline ice, composed of vertically-stacked, horizontally-homogeneous layers with unconstrained permittivity tensors (orientation fabrics).

<img src="https://raw.githubusercontent.com/nicholasmr/tamm/main/model.jpg" height="320px">

| Documentation | Reference |
| :--- | :--- |
| Model | [Rathmann et al. (2021)](https://doi.org/10.1029/2021GL096244) |
| Examples of use | TBA |
| Acknowledgements | [Passler and Paarmann (2017)](https://doi.org/10.1364/JOSAB.34.002128), [Passler et al. (2020)](https://doi.org/10.1103/PhysRevB.101.165425), [Jeannin (2019)](https://doi.org/10.5281/zenodo.3724504), [Yeh (1980)](https://doi.org/10.1016/0039-6028(80)90293-9), [Xu et al. (2000)](https://doi.org/10.1103/PhysRevB.61.1740) |



## Examples

In short, the model layer stack is initialized by specifying either the spectral coefficients of the fabric orientation distribution function (ODF) with depth, or by specifying the second-order structure tensor with depth. 
Thereafter, radar returns can easily be calculated and plotted by specifying the characteristics of the transmitted plane wave.

| Examples | Path |
| :--- | :--- |
| Jupyter demo | `experiments/demo/demo.ipynb` |
| Jupyter demo using [specfab](https://github.com/nicholasmr/specfab) | `experiments/demo/demo_specfab.ipynb` |
| [Rathmann et al. (2021)](https://doi.org/10.1029/2021GL096244) plots | `experiments/rathmann-etal-2021` |

