"""
This file is a copy of visualisation functions in kernels_in_GPFA_tutorial, and used in analyze_kernels_in_synthetic_data

"""
import os
import numpy as np
import quantities as pq
from elephant.gpfa import GPFA
import viziphant
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

def plot_trajectories_vs_time(lock,
                        spikeTrain_matrix,
                         GPFA_kargs = {'x_dim':3, 'bin_size': 20 *pq.ms, 'covType' : 'rbf', 'bo': False},
                         dimensions=[0, 1, 2],
                         orthonormalized_dimensions=True,
                         n_trials_to_plot = 30,
                         plot_average=True,
                         gradient_color_bar = False,
                         plot_args_single={'linewidth': 0.5,
                                           'alpha': 1,
                                           'linestyle': '-'},
                         plot_args_average={'linewidth': 2,
                                            'alpha': 1,
                                            'linestyle': '-',
                                            'color' : 'C1'},
                         figure_kargs=dict(figsize=(7.5, 7))):
    '''
    Plot latent 2D or 3D trajectories of a spike train matrix with color gradient to emphasizing temporal sequence in every single trajectory.
    Line segments from start to end between each neural state are mapped with gradient color in colormap plt.cm.jet .
    Parameters
    ----------
    lock : Threading Lock
        prevent matplotlib instance rewritting in multithreading execution
    spikeTrain_matrix : list of list of neo.SpikeTrain
        Spike train data to be fit to latent variables.
    GPFA_kargs : dict
        Arguments dictionary passed to GPFA
    dimensions : list of int
        Dimensions to plot.
    orthonormalized_dimensions : bool
        Boolean which specifies whether to plot the orthonormalized latent
        state space dimension corresponding to the entry 'latent_variable_orth'
        in returned data (True) or the unconstrained dimension corresponding
        to the entry 'latent_variable' (False).
        Beware that the unconstrained state space dimensions 'latent_variable'
        are not ordered by their explained variance. These dimensions each
        represent one Gaussian process timescale $\tau$.
        On the contrary, the orthonormalized dimensions 'latent_variable_orth'
        are ordered by decreasing explained variance, allowing a similar
        intuitive interpretation to the dimensions obtained in a PCA. Due to
        the orthonormalization, these dimensions reflect mixtures of
        timescales.
    n_trials_to_plot : int, optional
        Number of single trial trajectories to plot.
    plot_average : bool
        If True, trajectories are averaged and plotted.
    gradient_color_bar : bool
        If true, plot the color bar in gradient color, used when too much time bins to show in color bar.
        If false, plot the color bar in discrete intervals, color of each interval matches a line segment between each neural state.
    plot_args_single : dict
        Arguments dictionary passed to ax.plot() of the single trajectories.
    plot_args_average : dict
        Arguments dictionary passed to ax.plot() of the average trajectories.
    figure_kwargs : dict
        Arguments dictionary passed to ``plt.figure()``.

    Returns
    -------
    loglikelihood : float
        Data log-likelihood using GPFA to extract latent trajectories to 3 dimensions
    f : matplotlib.figure.Figure
    ax1 : matplotlib.axes.Axes

    ax2 : matplotlib.axes.Axes
    '''

    gpfa_3dim = GPFA(**GPFA_kargs)
    gpfa_3dim.fit(spikeTrain_matrix)
    returned_data = gpfa_3dim.transform(spikeTrain_matrix, returned_data=['latent_variable_orth','latent_variable'])
    log_likelihood = gpfa_3dim.fit_info["log_likelihoods"][-1]

    trajectories = viziphant.gpfa._check_input_data(returned_data, orthonormalized_dimensions)
    projection, n_dimensions = viziphant.gpfa._check_dimensions(gpfa_3dim,dimensions)

    lock.acquire()
    f = plt.figure(**figure_kargs)
    # partition the geometry into equally spaced 13 columns
    gs = f.add_gridspec(1,13)
    # place latent trajectory plot at the first 12 columns space and the color bar at the last column
    ax1 = f.add_subplot(gs[0,:-2],projection=projection)
    ax2 = f.add_subplot(gs[0,-1])

    ax1.set_title(gpfa_3dim.covType+' with LL '+str(round(log_likelihood,2)),fontsize=30)

    # single trial trajectories
    n_trials = trajectories.shape[0]
    for trail_idx in range(min(n_trials, n_trials_to_plot)):
        single_trial_trajectory = trajectories[trail_idx][dimensions, :]
        colour_map = np.linspace(0,len(single_trial_trajectory[0]),num=len(single_trial_trajectory[0])-1)
        colour_map_min = colour_map.min()
        colour_map_max = colour_map.max()

        for i in range(len(single_trial_trajectory[0])-1):
            if n_dimensions == 2:
                ax1.plot(single_trial_trajectory[0][i:i+2],single_trial_trajectory[1][i:i+2],
                color= plt.cm.jet(int((np.array(colour_map[i])-colour_map_min)*255/(colour_map_max-colour_map_min))),**plot_args_single)
            if n_dimensions == 3:
                # map each line segment (the line between each neural state) of a single trail trajectory with a gradient color in matplotlib.cm.jet color bar
                ax1.plot(single_trial_trajectory[0][i:i+2],single_trial_trajectory[1][i:i+2],single_trial_trajectory[2][i:i+2],
                color= plt.cm.jet(int((np.array(colour_map[i])-colour_map_min)*255/(colour_map_max-colour_map_min))),**plot_args_single)

    average_trajectory = np.mean(trajectories, axis=0)
    if plot_average:
        if n_dimensions == 2:
            ax1.plot(average_trajectory[0], average_trajectory[1], label='Trial averaged trajectory',**plot_args_average)
        if n_dimensions == 3:
            ax1.plot(average_trajectory[0], average_trajectory[1], average_trajectory[2], label='Trial averaged trajectory',**plot_args_average)
        ax1.legend(fontsize=18)
    # ax1.view_init(azim=-5, elev=60)  # pre-set viewing angle for the trajectory
    viziphant.gpfa._set_axis_labels_trajectories(ax1,orthonormalized_dimensions,dimensions)
    # set up color bar
    cmap = mpl.colors.ListedColormap(plt.cm.jet((np.linspace(0,1,len(trajectories[0][0])-1)*255).astype(int)))
    if gradient_color_bar:
        cb = mpl.colorbar.ColorbarBase(ax2, cmap=plt.cm.jet, ticks = [0,1])
        bounds = [1,len(trajectories[0][0])]
        cb.set_ticklabels(bounds,fontsize=15)
    else:
        # set up discrete color bar
        bounds = list(range(1,len(trajectories[0][0])+1))
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                    norm=norm,
                                    boundaries= bounds,
                                    # extend='both',
                                    ticks=bounds,
                                    # spacing='proportional',
                                    orientation='vertical')
        cb.set_ticklabels(bounds,fontsize=15)
    cb.set_label('time bin '+str(int(gpfa_3dim.bin_size.magnitude))+'ms',fontsize=15)
    plt.tight_layout()
    lock.release()
    del spikeTrain_matrix,gpfa_3dim,trajectories,colour_map,colour_map_max,colour_map_min,average_trajectory
    return log_likelihood, f, ax1, ax2

def plot_data_summary(lock,
                         spikeTrain_matrix,
                         GPFA_kargs = {'x_dim':2, 'bin_size': 20 *pq.ms, 'covType' : 'rbf', 'bo': False},
                         x_dims = [1,2,3],
                         plot_args_single={'linewidth': 0.5,
                                           'alpha': 1,
                                           'linestyle': '-'},
                         plot_args_average={'linewidth': 2,
                                            'alpha': 1,
                                            'linestyle': '-',
                                            'color' : 'C1'},):
    """
    Plot the 2D latent trajectory (ax1), first and second latent space state dimensions of the first trail versus time (ax2),
    raster plot of spike trains of trail 0 (ax3) and data loglikelihood when extracting latent trajectory to different dimensions (ax4)
    Parameters
    ----------
     lock : Threading Lock
        prevent matplotlib instance rewritting in multithreading execution
    spikeTrain_matrix : list of list of neo.SpikeTrain
        Spike train data to be fit to latent variables.
    GPFA_kargs : dict
        Arguments dictionary passed to GPFA
    x_dims : list of int
        Dimensions to analyse GPFA's performance, measured by data log-likelihood
    plot_args_single : dict
        Arguments dictionary passed to ax.plot() of the single trajectories.
    plot_args_average : dict
        Arguments dictionary passed to ax.plot() of the average trajectories.

    Returns
    -------
    loglikelihood : float
        Data log-likelihood using GPFA to extract latent trajectories to 2 dimensions
    f : matplotlib.figure.Figure
    ax1 : matplotlib.axes.Axes
        2D latent trajectory
    ax2 : matplotlib.axes.Axes
        First trail's first and second latent space state dimensions versus time
    ax3 : matplotlib.axes.Axes
        First trail's raster plot of spike trains
    ax4 : matplotlib.axes.Axes
        Data loglikelihood when extracting latent trajectory to different dimensions
    """

    log_likelihoods = []
    log_likelihood = np.nan
    # perform GPFA on every dimension to get data-loglikelihood on different dimensions
    for x_dim in x_dims:
        GPFA_kargs['x_dim'] = x_dim
        gpfa = GPFA(**GPFA_kargs)
        gpfa.fit(spikeTrain_matrix)
        log_likelihood_x_dim = gpfa.fit_info["log_likelihoods"][-1]
        log_likelihoods.append(log_likelihood_x_dim)
        if x_dim == 2:
            log_likelihood = log_likelihood_x_dim
            trajectories = gpfa.transform(spikeTrain_matrix)
            bin_size = gpfa.bin_size

    if 2 not in x_dims:
        GPFA_kargs['x_dim'] = 2
        gpfa = GPFA(**GPFA_kargs)
        log_likelihood_x_dim = gpfa.fit_info["log_likelihoods"][-1]
        log_likelihoods.append(log_likelihood_x_dim)
        trajectories = gpfa.transform(spikeTrain_matrix)
        bin_size = gpfa.bin_size

    lock.acquire()
    f = plt.figure(figsize=(15, 10))
    ax1 = f.add_subplot(2, 2, 1)
    ax2 = f.add_subplot(2, 2, 2)
    ax3 = f.add_subplot(2, 2, 3)
    ax4 = f.add_subplot(2, 2, 4)

    ax1.set_title('Latent dynamics extracted by GPFA with LL '+str(round(log_likelihood,2)))
    ax1.set_xlabel('Dim 1')
    ax1.set_ylabel('Dim 2')
    ax1.set_aspect(1)
    n_trials = trajectories.shape[0]
    for trail_idx in range(n_trials):
        # single trial trajectories
        single_trial_trajectory = trajectories[trail_idx]
        colour_map = np.linspace(0,len(single_trial_trajectory[0]),num=len(single_trial_trajectory[0])-1)
        colour_map_min = colour_map.min()
        colour_map_max = colour_map.max()

        for i in range(len(single_trial_trajectory[0])-1):
                ax1.plot(single_trial_trajectory[0][i:i+2],single_trial_trajectory[1][i:i+2],
                color= plt.cm.jet(int((np.array(colour_map[i])-colour_map_min)*255/(colour_map_max-colour_map_min))),**plot_args_single)

    # trial averaged trajectory
    average_trajectory = np.mean(trajectories, axis=0)
    ax1.plot(average_trajectory[0], average_trajectory[1], **plot_args_average, label='Trial averaged trajectory')
    ax1.legend()

    trial_to_plot = 0
    ax2.set_title(f'Trajectory for trial {trial_to_plot}')
    ax2.set_xlabel('Time [s]')
    times_trajectory = np.arange(len(trajectories[trial_to_plot][0])) * bin_size.rescale('s')
    ax2.plot(times_trajectory, trajectories[0][0], c='C0', label="Dim 1, fitting with all of trials")
    ax2.plot(times_trajectory, trajectories[0][1], c='C1', label="Dim 2, fitting with all of trials")
    ax2.legend()

    trial_to_plot = 0
    ax3.set_title(f'Raster plot of trial {trial_to_plot}')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Neuron id')
    # loop through every neuron in the first trail of spikeTrain_matrix
    for i, spiketrain in enumerate(spikeTrain_matrix[trial_to_plot]):
        ax3.plot(spiketrain, np.ones(len(spiketrain)) * i, ls='', marker='|')

    ax4.set_xlabel('Dimensionality of latent variables')
    ax4.set_ylabel('Log-likelihood')
    ax4.plot(x_dims, log_likelihoods, '.-')
    ax4.plot(x_dims[np.argmax(log_likelihoods)], np.max(log_likelihoods), 'x', markersize=10, color='r')

    plt.tight_layout()
    lock.release()
    return log_likelihood,f,ax1,ax2,ax3,ax4