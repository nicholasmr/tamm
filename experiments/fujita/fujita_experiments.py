#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 dlilien <dlilien@hozideh>
#
# Distributed under terms of the MIT license.

"""

"""
import sys
import os
sys.path.append(os.path.abspath('../../lib'))
import numpy as np
import matplotlib.pyplot as plt
import fujita_model

π = np.pi

for test in ['ridge', 'shearmargin', 'icestream']:
    fname = '../example_data/synthprofile_{:s}__tamm_out.h5'.format(test)
    with open(fname, 'rb') as f:
        depths = np.load(f)  # depth in meters
        A_xy = np.load(f)
        obs_angles = np.load(f)  # angle of frame rotation
        # indices: frame no. x layer no. x 2x2
        # ... i.e. a 2x2 matrix per interface per rotation of the radar frame
        R = np.load(f)
        T_in = np.load(f)
        Trev_in = np.load(f)
        M_in = np.load(f)
        Mrev_in = np.load(f)

    # We have no need for positive depths here
    depths = depths[1:]

    freq = 179.0e6
    EH_transmit = np.array([1.0e3, 0])
    EV_transmit = np.array([0.0, 1.0e3])

    # Do this the silly way
    S_iso = np.zeros((depths.shape[0], 2, 2))
    S_iso[:, 0, 0] = 1.0e0
    S_iso[:, 1, 1] = 1.0e0
    S_iso = S_iso * 1.0e-4

    S_aniso = np.zeros((depths.shape[0], 2, 2))
    S_aniso[:, 0, 0] = 1.0e0
    S_aniso[:, 1, 1] = 1.0e0
    S_aniso = S_aniso * 1.0e-4
    S_aniso[:, 1, 1] = S_aniso[:, 1, 1] * 0.2

    # I don't know exactly how a is specified here--I think it is per-layer, and we ignore the first.
    φ, ε, E = fujita_model.φεE(A_xy[1:, :2, :2])

    # One more scattering type
    S_drews = fujita_model.paren_scatter(E[:, 0], E[:, 1])

    # Isotropic scattering
    power = fujita_model.fujita_model_εS(EH_transmit, depths, obs_angles,
                                         ε, S_iso, φ, freq)
    powerV = fujita_model.fujita_model_εS(EV_transmit, depths, obs_angles,
                                          ε, S_iso, φ, freq)

    # Very basic assumption aniso
    power_scat = fujita_model.fujita_model_εS(EH_transmit, depths, obs_angles,
                                              ε, S_aniso, φ, freq)
    powerV_scat = fujita_model.fujita_model_εS(EV_transmit, depths, obs_angles,
                                               ε, S_aniso, φ, freq)

    # Paren/Drews/Ershadi scattering
    power_drews = fujita_model.fujita_model_εS(EH_transmit, depths, obs_angles,
                                               ε, S_drews, φ, freq)
    powerV_drews = fujita_model.fujita_model_εS(EV_transmit, depths, obs_angles,
                                                ε, S_drews, φ, freq)

    # Now plot them all
    plot_depths = depths / 1000.0
    vmin = -10
    vmax = 10
    fig, ((ax1, ax2, ax3, caxdb), (ax4, ax5, ax6, caxdeg), (ax7, ax8, ax9, dumax)) = plt.subplots(3, 4, gridspec_kw={'width_ratios': (1, 1, 1, 0.1), 'wspace': 0.4}, figsize=(12, 9))
    dumax.axis('off')

    cm = ax1.imshow(fujita_model.power_anomaly(power[:, :, 0]),
                    extent=[np.min(obs_angles), np.max(obs_angles),
                    plot_depths[-1], 0], vmin=vmin, vmax=vmax, cmap='RdBu_r')
    cm = ax2.imshow(fujita_model.power_anomaly(power[:, :, 1]),
                    extent=[np.min(obs_angles), np.max(obs_angles),
                    plot_depths[-1], 0], vmin=vmin, vmax=vmax, cmap='RdBu_r')
    cm2 = ax3.imshow(fujita_model.coherence_phase(power[:, :, 0], powerV[:, :, 1]),
                     extent=[np.min(obs_angles), np.max(obs_angles),
                     plot_depths[-1], 0], cmap='twilight_shifted', vmin=-180, vmax=180)

    cm = ax4.imshow(fujita_model.power_anomaly(power_scat[:, :, 0]),
                    extent=[np.min(obs_angles), np.max(obs_angles),
                    plot_depths[-1], 0], vmin=vmin, vmax=vmax, cmap='RdBu_r')
    cm = ax5.imshow(fujita_model.power_anomaly(power_scat[:, :, 1]),
                    extent=[np.min(obs_angles), np.max(obs_angles),
                    plot_depths[-1], 0], vmin=vmin, vmax=vmax, cmap='RdBu_r')
    cm2 = ax6.imshow(fujita_model.coherence_phase(power_scat[:, :, 0], powerV_scat[:, :, 1]),
                     extent=[np.min(obs_angles), np.max(obs_angles),
                     plot_depths[-1], 0], cmap='twilight_shifted', vmin=-180, vmax=180)

    cm = ax7.imshow(fujita_model.power_anomaly(power_drews[:, :, 0]),
                    extent=[np.min(obs_angles), np.max(obs_angles),
                    plot_depths[-1], 0], vmin=vmin, vmax=vmax, cmap='RdBu_r')
    cm = ax8.imshow(fujita_model.power_anomaly(power_drews[:, :, 1]),
                    extent=[np.min(obs_angles), np.max(obs_angles),
                    plot_depths[-1], 0], vmin=vmin, vmax=vmax, cmap='RdBu_r')
    cm2 = ax9.imshow(fujita_model.coherence_phase(power_drews[:, :, 0], powerV_drews[:, :, 1]),
                     extent=[np.min(obs_angles), np.max(obs_angles),
                     plot_depths[-1], 0], cmap='twilight_shifted', vmin=-180, vmax=180)

    plt.colorbar(cm, cax=caxdb, label='Power anomaly (dB)')
    plt.colorbar(cm2, cax=caxdeg, label=r'Coherence phase ($^\circ$)', extend='both')
    ax1.set_title('HH')
    ax2.set_title('HV')
    ax3.set_title('Coherence phase')
    ax1.set_ylabel('Isotropic scattering\n\nDepth (km)')
    ax4.set_ylabel('10 dB anisotropic scattering\n\nDepth (km)')
    ax7.set_ylabel('Paren (1981) scattering\n\nDepth (km)')

    for ax in (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9):
        ax.set_aspect(aspect="auto")
        ax.set_xlim(0, np.pi)
        ax.set_ylim(plot_depths[-1], 0)
    fig.savefig('simple_{:s}.png'.format(test), dpi=300)
