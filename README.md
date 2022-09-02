# Kernels_in_GPFA
Development of a 9 new kernels functions in Elephant GPFA python package. Original version at [link](https://github.com/NeuralEnsemble/elephant).

## installation
To install package, go to 'elephant_modified' directory and execute `pip install -e.` in command line.

## new features added to Elephant GPFA

Added Exponential, Triangular, Rational Quadratic, Matern, Spectral Mixture kernels and 3 Production of kernels: Triangular times Rational Quadratic, Exponential times Rational Quadratic, Exponential times Triangular kernels.
Parameters of new kernels are currently optimized by 2 gradient-free method: `Scipy.optimize.minimize` Powell mehthod and a built-in Bayeisian Optimization script.

## execution
Kernel names are in short, specify kernel names in '{'rbf', 'tri', 'exp', 'rq', 'matern', 'sm','tri_times_rq', 'exp_times_rq', 'exp_times_tri'}' at parameter `covTpe`.
Specify optimization mehthod at parameter `bo`. Default to use Powell method. Input a positive numbers will use Bayesian Optimization instead, the number will be iterations of Bayesian Optimisation.
Example to use Spectral Mixture kernel, optimize its kernel parameter by  Bayesian Optimization in 50 iterations. Time binned in 20ms, extract trajectory to 3 dimensional space.
```
gpfa_3dim_sm = Elephant.gpfa.GPFA(bin_size=20*pq.ms, x_dim=3,covType='sm',bo=3)
gpfa_3dim_sm.fit(spiketrains)
```
## additional visualization methods
`GPFA_visualisation_addons` contains `plot_trajectories_vs_time`, which is developed in addition to `viziphant.gpfa.plot_trajectoires`.
This functrion emphasize temporal order between neural states by adopting gradient color in plotting single trail latent trajectories. 
The function involkes GPFA internally, parameter fields are nearly the same with `viziphant.gpfa.plot_trajectoires`. Example to use this function to extract latent trajectories to 3D.
```
plot_trajectories_vs_time(spikeTrains,GPFA_kargs = {'x_dim':3, 'bin_size': 20 *pq.ms, 'covType' : 'sm', 'bo': False}, dimensions=[0, 1, 2])
```
Example of plotted trajectory using first experiment of natural iamges dataset.

## kernel performance benchmark
The 9 kernels are benchmarked in neuroscience dataset collected from Salamander Retina Ganglion cells at [link](https://datadryad.org/stash/dataset/doi:10.5061/dryad.4ch10) in terms of GPFA returned data log-likelihood, Spectral Mixture kernel with 2 spectral mixtures has a significant improvement than orginally adopted rbf kernel. 