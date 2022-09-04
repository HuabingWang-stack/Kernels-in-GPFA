# Kernels in GPFA
This repository contains a modified version of Elephant python package where 9 new kernel functions are developed onto, and 2 
tutorials to explain the properties of latent trajectories extracted by GPFA with new kernels. Original version of Elephant 
at [link](https://github.com/NeuralEnsemble/elephant).

## installation
To install package, go to 'elephant_modified' directory and execute `pip install -e.` in command line.

## new features added to Elephant GPFA

Added Exponential, Triangular, Rational Quadratic, Matern, Spectral Mixture kernels and 3 Production of kernels: Triangular times Rational Quadratic, Exponential times Rational Quadratic, Exponential times Triangular kernels.
Parameters of new kernels are currently optimized by 2 gradient-free method: `Scipy.optimize.minimize` Powell mehthod and a built-in Bayeisian Optimization script.

## execution
Kernel names are in short, choose kernel in '{'rbf', 'tri', 'exp', 'rq', 'matern', 'sm','tri_times_rq', 'exp_times_rq', 'exp_times_tri'}' at parameter field `covTpe`.
Specify optimization method at parameter `bo`. Default is False to use Powell method. Input a positive numbers will use Bayesian Optimization instead, the number will be iterations of Bayesian Optimisation.


The following example initialize GPFA with Spectral Mixture kernel, and optimize its kernel parameter by  Bayesian Optimization in 50 iterations. Time binned in 20ms, extract trajectory to 3 dimensional space.
```python
gpfa_3dim_sm = Elephant.gpfa.GPFA(bin_size=20*pq.ms, x_dim=3,covType='sm',bo=50)
gpfa_3dim_sm.fit(spiketrains)
```

## math expressions of kernels
In order to unify the expressions of kernels, equations of each kernel used in 
the Elephant GPFA python package are listed here:

$$
\text{rbf kernel: } k_{rbf}(t_{1},t_{2},\gamma) = exp\left (-\dfrac{(t_1-t_2)^2}{2\gamma}\right ) 
$$

$$
\text{rational quadratic kernel: } k_{rq}(t_{1},t_{2},\alpha,\ell) = \left(1+\frac{(t_1-t_2)^{2}}{2\alpha \ell^{2}}\right)^{-\alpha}
$$

$$
\text{matern kernel: } k_{matern}(t_{1},t_{2},\nu,\ell) = \frac{2^{1-\nu}}{\Gamma(\nu)} \left (\frac{\sqrt{2\nu}(t_{1}-t_{2})}{\ell}\right )^{\nu}K_{\nu}\left(\frac{\sqrt{2\nu}(t_1-t_2)}{\ell}\right)
$$

$$
\text{spectral mixture kernel: } k_{sm}(t_{1},t_{2},Q,w,\mu,\nu) = \sum_{q=1}^{Q} \exp \left(-2\pi^{2}(t_{1}-t_{2})^{2}\nu \right) \\cos \left(2\pi (t_{1}-t_{2}) \mu \right)
$$

$$
\text{exponential kernel: } k_{exp}(t_{1},t_{2},\sigma) = \exp\left(-\frac{|t_{1}-t_{2}|}{2\sigma^2}\right)
$$

$$
\text{triangular kernel: } k_{tri}(t_{1},t_{2},\sigma) = \begin{cases}\frac{1}{\sqrt{6}\sigma}\left(1-\frac{|t_{1}-t_{2}|}{\sqrt{6}\sigma}\right),  & |t_{1}-t_{2}|  < \sqrt{6}\sigma\\
    0, & |t_{1}-t_{2}|  \geq \sqrt{6}\sigma
    \end{cases}
$$

$$
\text{triangular times rational quadratic kernel: } k_{tri \textunderscore times \textunderscore rq}(t_{1},t_{2},\sigma,\alpha,\ell) = \begin{cases}\frac{1}{\sqrt{6}\sigma}\left(1-\frac{|t_{1}-t_{2}|}{\sqrt{6}\sigma}\right)\left(1+\frac{(t_1-t_2)^{2}}{2\alpha \ell^{2}}\right)^{-\alpha},  & |t_{1}-t_{2}|  < \sqrt{6}\sigma\\
    0, & |t_{1}-t_{2}|  \geq \sqrt{6}\sigma
    \end{cases}
$$

$$
\text{exponential times rational quadratic kernel: } k_{exp \textunderscore times \textunderscore rq}(t_{1},t_{2},\sigma,\alpha,\ell) = \exp\left(-\frac{|t_{1}-t_{2}|}{2\sigma^2}\right) \left(1+\frac{(t_1-t_2)^{2}}{2\alpha \ell^{2}}\right)^{-\alpha}
$$

$$
\text{exponential times triangular kernel: } k_{exp \textunderscore times \textunderscore tri}(t_{1},t_{2},\sigma_{e},\sigma_{t}) = \begin{cases} \exp\left(-\frac{|t_{1}-t_{2}|}{2\sigma_{e}^2}\right) \frac{1}{\sqrt{6}\sigma_{t}}\left(1-\frac{|t_{1}-t_{2}|}{\sqrt{6}\sigma_{t}}\right),  & |t_{1}-t_{2}|  < \sqrt{6}\sigma\\
    0, & |t_{1}-t_{2}|  \geq \sqrt{6}\sigma
    \end{cases}
$$

## kernel performance benchmark
The 9 kernels are benchmarked in neuroscience dataset collected from Salamander Retina Ganglion cells in `Spikes` directory
data available at [link](https://datadryad.org/stash/dataset/doi:10.5061/dryad.4ch10). The benchmark results are at cloud storage [link](https://www.dropbox.com/scl/fo/upo6z57eqlx0dilgymdsx/h?dl=0&rlkey=fe0or0kpz93km3oo96nldl3c6),
please refer to `README.md` file in directory Experiments Data for detail information of this benchmark result.
In terms of data log-likelihood returned by GPFA model, 
Spectral Mixture kernel with 2 spectral mixtures has a significant improvement over than originally adopted rbf kernel. 

## additional visualization methods
`GPFA_visualisation_addons` contains `plot_trajectories_vs_time`, which is developed in addition to `viziphant.gpfa.plot_trajectoires`.
This function emphasize temporal order between neural states by adopting gradient color in plotting single trail latent trajectories. 
The function invokes GPFA internally, parameter fields are nearly the same with `viziphant.gpfa.plot_trajectoires`. Example to use this function to extract latent trajectories to 3D.
```
plot_trajectories_vs_time(spikeTrains,GPFA_kargs = {'x_dim':3, 'bin_size': 20 *pq.ms, 'covType' : 'sm', 'bo': False}, dimensions=[0, 1, 2])
```
Example of plotted latent trajectory of the first experiment of `NaturalImages1.mat` dataset.
![alt text](./LatentTrajectories/NaturalImages1/sm/0_3d.png?raw=true)

## tutorials

`kernels_inGPFA_tutorial.ipynb` has a detailed practice to perform this modified GPFA on the first 
experiment of `NaturalImages1.mat` dataset. It compares the difference of extracted trajectories between rbf and spectral 
mixture kernel in part 1. Then it demonstrates the new plotting function in part 2, and shows the way to achieve multi-threading
when running on a large dataset in part 3.

`analyze_kernels_in_synthetic_data.ipynb` compares the smoothness property of the 9 kernels on synthetic data with 
pre-defined harmonic oscillator latent dynamics.