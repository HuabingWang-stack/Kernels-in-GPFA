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

$$
\text{matern kernel: } (k_{Matern}(t_{1},t_{2},\nu,\ell) = \frac{2^{1-\nu}}{\Gamma(\nu)} \left (\frac{\sqrt{2\nu}(t_{1}-t_{2})}{\ell}\right )^{\nu}K_{\nu}\left(\frac{\sqrt{2\nu}(t_1-t_2)}{\ell}\right)
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
\text{triangular times rational quadratic kernel: } k_{tri\_times\_rq}(t_{1},t_{2},\sigma,\alpha,\ell) = \begin{cases}\frac{1}{\sqrt{6}\sigma}\left(1-\frac{|t_{1}-t_{2}|}{\sqrt{6}\sigma}\right)\left(1+\frac{(t_1-t_2)^{2}}{2\alpha \ell^{2}}\right)^{-\alpha},  & |t_{1}-t_{2}|  < \sqrt{6}\sigma\\
    0, & |t_{1}-t_{2}|  \geq \sqrt{6}\sigma
    \end{cases}
$$

$$
\text{exponential times rational quadratic kernel: } k_{exp\_times\_rq}(t_{1},t_{2},\sigma,\alpha,\ell) = \exp\left(-\frac{|t_{1}-t_{2}|}{2\sigma^2}\right) \left(1+\frac{(t_1-t_2)^{2}}{2\alpha \ell^{2}}\right)^{-\alpha}
$$

$$
\text{exponential times triangular kernel: } k_{exp\_times\_tri}(t_{1},t_{2},\sigma_{e},\sigma_{t}) = \begin{cases} \exp\left(-\frac{|t_{1}-t_{2}|}{2\sigma_{e}^2}\right) \frac{1}{\sqrt{6}\sigma_{t}}\left(1-\frac{|t_{1}-t_{2}|}{\sqrt{6}\sigma_{t}}\right),  & |t_{1}-t_{2}|  < \sqrt{6}\sigma\\
    0, & |t_{1}-t_{2}|  \geq \sqrt{6}\sigma
    \end{cases}
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