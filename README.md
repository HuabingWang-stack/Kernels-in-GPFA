# Kernels_in_GPFA
Development of a 9 new kernels functions in Elephant GPFA python package. Original version at [link](https://github.com/NeuralEnsemble/elephant).

## installation
To install package, go to 'elephant_modified' directory and execute `pip install -e.` in command line.

## new features added to Elephant GPFA

Added Exponential, Triangular, Rational Quadratic, Matern, Spectral Mixture kernels and 3 Production of kernels: Triangular times Rational Quadratic, Exponential times Rational Quadratic, Exponential times Triangular kernels.
Parameters of new kernels are currently optimized by 2 gradient-free method: `Scipy.optimize.minimize` Powell mehthod and a built-in Bayeisian Optimization script.

## execution
Kernel names are in short, specify kernel names in '{'rbf', 'tri', 'exp', 'rq', 'matern', 'sm','tri_times_rq', 'exp_times_rq', 'exp_times_tri'}' at parameter `covTpe`.
Specify optimization method at parameter `bo`. Default is False to use Powell method. Input a positive numbers will use Bayesian Optimization instead, the number will be iterations of Bayesian Optimisation.
Example to use Spectral Mixture kernel, optimize its kernel parameter by  Bayesian Optimization in 50 iterations. Time binned in 20ms, extract trajectory to 3 dimensional space.
```
gpfa_3dim_sm = Elephant.gpfa.GPFA(bin_size=20*pq.ms, x_dim=3,covType='sm',bo=3)
gpfa_3dim_sm.fit(spiketrains)
```
## kernel performance benchmark
The 9 kernels are benchmarked in neuroscience dataset collected from Salamander Retina Ganglion cells in `Spikes` directory
data available at [link](https://datadryad.org/stash/dataset/doi:10.5061/dryad.4ch10). The results of GPFA when 
adopting all types of kernels on this dataset is at a cloud storage at [link](https://www.dropbox.com/scl/fo/upo6z57eqlx0dilgymdsx/h?dl=0&rlkey=fe0or0kpz93km3oo96nldl3c6).
In terms of GPFA returned data log-likelihood, 
Spectral Mixture kernel with 2 spectral mixtures has a significant improvement than originally adopted rbf kernel. 

## math expressions of kernels
In order to unify the expressions of kernels, equations of each kernel used in 
GPFA are listed here.

$$
\text{rbf kernel: } k_{rbf}(t_{1},t_{2},\gamma) = exp\left (-\dfrac{(t_1-t_2)^2}{2\gamma}\right ) 
$$

$$
\text{rational quadratic kernel: } k_{rq}(t_{1},t_{2},\alpha,\ell) = \left(1+\frac{(t_1-t_2)^{2}}{2\alpha \ell^{2}}\right)^{-\alpha}
$$

## additional visualization methods
`GPFA_visualisation_addons` contains `plot_trajectories_vs_time`, which is developed in addition to `viziphant.gpfa.plot_trajectoires`.
This function emphasize temporal order between neural states by adopting gradient color in plotting single trail latent trajectories. 
The function invokes GPFA internally, parameter fields are nearly the same with `viziphant.gpfa.plot_trajectoires`. Example to use this function to extract latent trajectories to 3D.
```
plot_trajectories_vs_time(spikeTrains,GPFA_kargs = {'x_dim':3, 'bin_size': 20 *pq.ms, 'covType' : 'sm', 'bo': False}, dimensions=[0, 1, 2])
```
Example of plotted latent trajectory of the first experiment of natural images dataset.
![alt text](./LatentTrajectories/NaturalImages1/sm/0_3d.png?raw=true)