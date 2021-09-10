#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © 2020 David Lilien <david.lilien@nbi.ku.dk>
#
# Distributed under terms of the GNU GPL3.0 license.


"""
Model anisotropic radar returns following Fujita et al., 2006.

Created on Thu Jul  9 12:01:00 2020
"""

import numpy as np
import matplotlib.pyplot as plt

ε_0 = 8.8541878128e-12
μ_0 = 1.25663706212e-6
π = np.pi
C = 3.0e8
ε_prime = 3.17
ε_perp = 3.17


def dB(amp):
    """Convert the (complex) wave amplitude to decibels.

    Parameters
    ----------
    amp: array_like
        Wave amplitude to convert to dB.

    Returns
    -------
    power: array_like
        Power, with same shape as input.
    """
    return 10.0 * np.log10(np.abs(amp) ** 2.0)


def dielectric_anisotropy(T):
    """Dielectric anisotropy as function of temperature.

    (in kelvin).

    Fujita 2006 Equation 3.

    Parameters
    ----------
    T: array_like
        Temperature

    Returns
    -------
    Δε: The change in ε we expect at that temperature.
    """
    return 0.0256 + 3.57e-5 * T


def prop_constant(ε, σ, f):
    """Get propagation constant from eq. 7 in Fujita.

    Works the same for x or y.
    """
    ω = 2.0 * π * f
    return np.sqrt(ε_0 * μ_0 * ε * ω ** 2.0 + 1.0j * μ_0 * σ * ω)


def transmissivity(ε, σ, f, depths):
    """Get transmissivity at all depths.

    ε, σ can be scalar or nd arrays.
    """
    Δd = np.diff(depths)
    k = prop_constant(ε, σ, f)
    return np.exp(-1.0j * k0(f) * Δd + 1.0j * k * Δd)


def incident_to_scatterer(E,
                          depths,
                          T,
                          θs,
                          k_0=1):
    """Get || and |- power at each depth and angle.

    Direct implementation of Fujita et al., 2006 Eq. 9.

    We do this for all depths, but only one angle.

    The θs given are the orthotropic reference frame
    angles minus the measurement angle.
    """
    prop = np.exp(1.0j * k_0 * depths[1:]) / (4.0 * π * depths[1:])
    rot = np.stack((np.vstack((np.cos(θs), -np.sin(θs))).T,
                    np.vstack((np.sin(θs), np.cos(θs))).T))
    rot = np.swapaxes(rot, 1, 0)
    RTR = np.matmul(np.matmul(rot, T), np.swapaxes(rot, 1, 2))
    prod = RTR.copy()
    for i in range(1, prod.shape[0]):
        prod[i, :, :] = np.matmul(prod[i - 1, :, :], RTR[i, :, :])
    return (prop * np.matmul(prod, E).T).T


def scattered(E, S, θs):
    """Scattering at each layer for given θ and S."""
    rot = np.stack((np.vstack((np.cos(θs), -np.sin(θs))).T,
                    np.vstack((np.sin(θs), np.cos(θs))).T))
    rot = np.swapaxes(rot, 1, 0)
    return np.matmul(np.matmul(np.matmul(rot, S), np.swapaxes(rot, 1, 2)),
                     np.atleast_3d(E))[:, :, 0]


def observed_from_scaterrer(E,
                            depths,
                            T,
                            θs,
                            k_0=1):
    """Get || and |- power at each depth and angle.

    Direct implementation of Fujita et al., 2006 Eq. 12.

    This is actually identical to the opposite direction, since all terms
    cancel. Not really sure what is up with this...

    We do this for all depths, but only one angle.

    The θs given are the orthotropic reference frame
    angles minus the measurement angle.
    """
    prop = np.exp(1.0j * k_0 * depths[1:]) / (4.0 * π * depths[1:])
    rot = np.stack((np.vstack((np.cos(θs), -np.sin(θs))).T,
                    np.vstack((np.sin(θs), np.cos(θs))).T))
    rot = np.swapaxes(rot, 1, 0)
    RTR = np.matmul(np.matmul(rot, T), np.swapaxes(rot, 1, 2))
    prod = RTR.copy()
    for i in range(1, prod.shape[0]):
        prod[i, :, :] = np.matmul(RTR[i, :, :], prod[i - 1, :, :])
    return (prop * np.matmul(prod, np.atleast_3d(E))[:, :, 0].T).T


def fujita_model_TS(E_transmit, depths, obs_angles, T, S, iso_angles, k_0, Trev=None):
    """Run the Fujuta model assuming we already have T_i and S_i.

    The shapes of T and S differ by 1 in the depth dimension, since S is per-interface (including 0) and
    T is per layer.
    """
    if Trev is None:
        Trev = T
    return_matrix = np.zeros((depths.shape[0], obs_angles.shape[0], 2), dtype=np.complex128)
    for i, angle in enumerate(obs_angles):
        θs = iso_angles - angle

        # Fencepost because the 0 return has not passed through anything
        return_matrix[0, i, :] = scattered(E_transmit, S[0, :, :], 0.0)

        # Subsequent layers have passet through stuff
        return_matrix[1:, i, :] = observed_from_scaterrer(
            scattered(
                incident_to_scatterer(E_transmit, depths, T, θs, k_0=k_0),
                S[1:, :, :], θs),
            depths, Trev, θs, k_0=k_0)
    return return_matrix


def fujita_model_εS(E_transmit, depths, obs_angles,
                    ε, S, iso_angles, f, σ=1.0e-5):
    """Run the model assuming that we know the scattering and the permitivity.

    We expect ε to be a nx2 matrix where n is the number of depths. However,
    we can also accept matrix of length 2

    σ is constant, taken to be 1e-5 from casual mention in Fujita 2006. See
    Fujita 2000 for more. In principle, this function would accept a matrix
    here.
    """
    assert depths.shape[0] == S.shape[0]
    # T exists per-layer, so shape is reduced by 1.
    T = np.zeros((depths.shape[0] - 1, 2, 2), dtype=np.complex128)
    if len(ε.shape) > 1:
        T[:, 0, 0] = transmissivity(np.hstack(([0.0], (ε[:-1, 0] + ε[1:, 0]) / 2.0)), σ, f, depths)
        T[:, 1, 1] = transmissivity(np.hstack(([0.0], (ε[:-1, 1] + ε[1:, 1]) / 2.0)), σ, f, depths)
    else:
        T[:, 0, 0] = transmissivity(ε[0], σ, f, depths)
        T[:, 1, 1] = transmissivity(ε[1], σ, f, depths)
    return fujita_model_TS(E_transmit, depths, obs_angles,
                           T, S, iso_angles, k0(f))


def k0(f):
    """Wavenumber for a given frequency

    Parameters
    ----------
    f: float
        The frequency (in Hz)
    """
    return 2.0 * π / (C / f)


def ϕ(ε_x, ε_y, f, depths):
    """Phase offset as a function of permitivitty and depth.

    Fujita 2006 Equation 13.
    """
    if hasattr(ε_x, 'shape') and len(ε_x.shape) > 0:
        return np.hstack(([0], 4.0 * π * f / C * np.cumsum(np.diff(depths) * (np.sqrt((ε_x[1:] + ε_x[:-1]) / 2.) - np.sqrt((ε_y[1:] + ε_y[:-1]) / 2.)))))
    else:
        return 4.0 * π * f / C * np.cumsum(np.hstack(([0], np.diff(depths))) * (np.sqrt(ε_x) - np.sqrt(ε_y)))


def ε_aniso(E_1, E_2, T=263.15):
    """Get the matrix for transmissivity in the orthotropic frame."""
    Δε = dielectric_anisotropy(T)
    if hasattr(E_1, 'shape') and len(E_1.shape) > 0:
        return np.vstack([ε_perp + E_1 * Δε, ε_perp + E_2 * Δε]).T
    else:
        return np.array([ε_perp + E_1 * Δε, ε_perp + E_2 * Δε])


def φεE(A_xy, T=263.15):
    """Get angles and real permitivity from the fabric tensor.

    Parameters
    ----------
    A_xy:
        mx2x2 matrix of the x and y components of the fabric.
        We assume that one eigenvector is vertical. A must be Hermitian
    T:
        The temperature
    """
    E, rot_mat = np.linalg.eig(A_xy)

    if len(A_xy.shape) == 2:
        # We now need to normalize the eigenvalues
        E = E * (A_xy[0, 0] + A_xy[1, 1]) / (E[0] + E[1])
        φ = np.arccos(rot_mat[0, 0])
        if rot_mat[0, 1] < 0.0:
            φ = -φ
        return φ, ε_aniso(E[0], E[1], T=T), E
    else:
        norm = (A_xy[:, 0, 0] + A_xy[:, 1, 1]) / (E[:, 0] + E[:, 1])
        E[:, 0] = E[:, 0] * norm
        E[:, 1] = E[:, 1] * norm
        φ = np.arccos(rot_mat[:, 0, 0])
        φ[rot_mat[:, 0, 1] < 0.0] = -φ[rot_mat[:, 0, 1] < 0.0]
        return φ, ε_aniso(E[:, 0], E[:, 1], T=T), E


def paren_scatter(E1, E2, T=263.15):
    S = np.zeros((E1.shape[0] + 1, 2, 2), dtype=np.complex128)
    Δε = dielectric_anisotropy(T)
    # Dummy values at the top and bottom
    S[0, 0, 0] = Δε / (4.0 * ε_prime)
    S[0, 1, 1] = Δε / (4.0 * ε_prime)
    S[-1, 0, 0] = Δε / (4.0 * ε_prime)
    S[-1, 1, 1] = Δε / (4.0 * ε_prime)

    # Drews et al., 2012 Eq. (3)
    S[1:-1, 0, 0] = (np.diff(E1) * Δε / (4.0 * ε_prime)) ** 2.0
    S[1:-1, 1, 1] = (np.diff(E2) * Δε / (4.0 * ε_prime)) ** 2.0
    return S


def abs_angle(power):
    """Find co-cross polarized return difference.

    Not calculated in Brisbourne.
    """
    ang = np.angle(power, deg=True)[:, :, 0] - np.angle(power, deg=True)[:, :, 1]
    return to_180(ang)


def power_anomaly(power):
    """Compute power anomaly as in Brisbourne."""
    return dB(power) - np.atleast_2d(np.mean(dB(power), axis=1)).T


def to_180(ang):
    """Clip bounds using periodicity--needed because we run np.angle twice.

    Parameters
    ----------
    ang: np.array
        angles in degrees
    """
    ang[ang < -180] = ang[ang < -180] + 360.
    ang[ang > 180] = ang[ang > 180] - 360.
    return ang


def angle(power, obs_angles=π / 12.0):
    """Difference in phase between neighboring returns. Matches Brisbourne.

    obs_angles is used for somewhat arbitrary rescaling to match the 15 deg
    spacing used in Brisbourne
    """
    if hasattr(obs_angles, 'shape'):
        obs_angles = np.ones((power.shape[1],)) * obs_angles
    return to_180(np.diff(np.angle(power[:, :, 0], deg=True), axis=1)
                  ) / np.diff(obs_angles) * π / 12.0


def coherence_phase(p1, p2):
    ang = np.angle(p1 * p2.conjugate() / (np.abs(p1) * np.abs(p2)), deg=True)
    return ang


def fujita_2006_plots(debug=False):
    """Make Figure 5 from Fujita 2006."""
    freq = 179.0e6
    d = 1700.
    depths = np.linspace(0, d, 600)
    obs_angles = np.linspace(0, π, 180)
    ε = np.array([1.0, 1.0]) * ε_prime
    ε[1] = ε[1] - 0.1 * dielectric_anisotropy(263.15)
    E_transmit = np.array([1.0e3, 0])
    iso_angles = np.zeros_like(depths)

    # Isotropic scattering, should include the top interface (i.e. at 0).
    S = np.zeros((depths.shape[0], 2, 2))
    S[:, 0, 0] = 1.0
    S[:, 1, 1] = 1.0
    S = S * 1.0e-4  # -40 dB returns

    # Fujita 2006 Figure 1a-b
    power = fujita_model_εS(E_transmit, depths, obs_angles,
                            ε, S, iso_angles, freq)

    # Anisotropic scattering
    S_aniso = np.zeros((depths.shape[0], 2, 2))
    S_aniso[:, 0, 0] = 1.0
    S_aniso[:, 1, 1] = 0.1
    S_aniso = S_aniso * 1.0e-4  # -40 dB returns

    power_anisoscatter = fujita_model_εS(E_transmit, depths, obs_angles,
                                         np.ones((2,)) * ε_prime, S_aniso,
                                         iso_angles, freq)
    power_bothaniso = fujita_model_εS(E_transmit, depths, obs_angles,
                                      ε, S_aniso, iso_angles, freq)
    iso_power = fujita_model_εS(E_transmit, depths, obs_angles,
                                np.ones((2,)) * ε_prime, S, iso_angles, freq)

    fig, ((axpb, axcb),
          (axps, axcs),
          (axpa, axca)) = plt.subplots(3, 2,
                                       sharex=True, sharey=True,
                                       figsize=(5, 9))

    # Birefringence only
    axpb.imshow(dB(power[:, :, 0]) - dB(iso_power[:, :, 0]),
                extent=[0, π, d, 0], vmin=-10, vmax=10)
    axpb.set_aspect(aspect="auto")
    CS = axpb.contour(np.flipud(dB(power[:, :, 0]) - dB(iso_power[:, :, 0])),
                      levels=[-10, -5, -3], colors='k',
                      extent=[0, π, d, 0])
    axpb.clabel(CS, CS.levels, inline=True, fmt='%4.0f', fontsize=10)
    axpb.set_title('Co-polarized')
    axcb.imshow(dB(power[:, :, 1]) - dB(iso_power[:, :, 0]),
                extent=[0, π, d, 0], vmin=-10, vmax=10)
    axcb.set_aspect(aspect="auto")
    CS = axcb.contour(np.flipud(dB(power[:, :, 1]) - dB(iso_power[:, :, 0])),
                      levels=[-10, -5, -3], colors='k',
                      extent=[0, π, d, 0])
    axcb.clabel(CS, CS.levels, inline=True, fmt='%4.0f', fontsize=10)
    axcb.set_title('Cross-polarized')

    # Scattering only
    axps.imshow(dB(power_anisoscatter[:, :, 0]) - dB(iso_power[:, :, 0]),
                extent=[0, π, d, 0], vmin=-10, vmax=10)
    axps.set_aspect(aspect="auto")
    CS = axps.contour(
        np.flipud(dB(power_anisoscatter[:, :, 0]) - dB(iso_power[:, :, 0])),
        levels=[-10, -5, -3], colors='k', extent=[0, π, d, 0])
    axps.clabel(CS, CS.levels, inline=True, fmt='%4.0f', fontsize=10)
    axps.set_ylabel('Depth (m)')
    axcs.imshow(dB(power_anisoscatter[:, :, 1]) - dB(iso_power[:, :, 0]),
                extent=[0, π, d, 0], vmin=-10, vmax=10)
    axcs.set_aspect(aspect="auto")
    CS = axcs.contour(
        np.flipud(dB(power_anisoscatter[:, :, 1]) - dB(iso_power[:, :, 0])),
        levels=[-10, -5, -3], colors='k', extent=[0, π, d, 0])
    axcs.clabel(CS, CS.levels, inline=True, fmt='%4.0f', fontsize=10)

    # Anisotropic birefringence and scattering
    axpa.imshow(dB(power_bothaniso[:, :, 0]) - dB(iso_power[:, :, 0]),
                extent=[0, π, d, 0], vmin=-10, vmax=10)
    axpa.set_aspect(aspect="auto")
    CS = axpa.contour(
        np.flipud(dB(power_bothaniso[:, :, 0]) - dB(iso_power[:, :, 0])),
        levels=[-30, -20, -10, -5, -3], colors='k', extent=[0, π, d, 0])
    axpa.clabel(CS, CS.levels, inline=True, fmt='%4.0f', fontsize=10)
    axpa.set_xticks(np.linspace(0, 1, 5) * π)
    axpa.set_xticklabels(['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$',
                          r'$\frac{3\pi}{4}$', r'$\pi$'])
    axpa.set_xlabel('Antenna orientation θ\n(radians)')
    axca.imshow(dB(power_bothaniso[:, :, 1]) - dB(iso_power[:, :, 0]),
                extent=[0, π, d, 0], vmin=-10, vmax=10)
    axca.set_aspect(aspect="auto")
    CS = axca.contour(
        np.flipud(dB(power_bothaniso[:, :, 1]) - dB(iso_power[:, :, 0])),
        levels=[-10, -5, -3], colors='k', extent=[0, π, d, 0])
    axca.clabel(CS, CS.levels, inline=True, fmt='%4.0f', fontsize=10)
    axca.set_xlabel('Antenna orientation θ\n(radians)')

    # These axes are annoying if we have to check values
    if not debug:
        for axc in [axcb, axcs, axca]:
            ax1 = axc.twinx()
            phase_off = ϕ(ε[0], ε[1], freq, depths)
            ax1.set_ylim(phase_off[-1], 0)
            phase_iters = np.floor(phase_off[-1] / π)
            ax1.set_yticks(np.arange(phase_iters + 1) * π)
            ax1.set_yticklabels(['0', 'π'] + [
                '{:d}π'.format(i) for i in range(2, int(phase_iters) + 1)])
            if axc is axcs:
                ax1.set_ylabel(r'Phase offset $\phi$ (Radians)')

    # We may need additional plots if this is broken
    if debug:
        fig, (axp, axc) = plt.subplots(1, 2, sharex=True, sharey=True)
        axp.imshow(dB(power[:, :, 0]) - dB(iso_power[:, :, 0]),
                   extent=[0, π, d, 0], vmin=-10, vmax=10)
        axp.set_aspect(aspect="auto")
        CS = axp.contour(np.flipud(
            dB(power[:, :, 0]) - dB(iso_power[:, :, 0])),
            levels=[-10, -5, -3], colors='k', extent=[0, π, d, 0])
        axp.clabel(CS, CS.levels, inline=True, fmt='%4.0f', fontsize=10)
        axp.set_xticks(np.linspace(0, 1, 5) * π)
        axp.set_xticklabels(['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$',
                             r'$\frac{3\pi}{4}$', r'$\pi$'])
        axp.set_title('Co-polarized')
        axp.set_xlabel(r'Antenna orientation $\θ$ (radians)')
        axp.set_ylabel('Depth (m)')

        axc.imshow(dB(power[:, :, 1]) - dB(iso_power[:, :, 0]),
                   extent=[0, π, d, 0], vmin=-10, vmax=10)
        axc.set_aspect(aspect="auto")
        CS = axc.contour(np.flipud(
            dB(power[:, :, 1]) - dB(iso_power[:, :, 0])),
            levels=[-10, -5, -3], colors='k', extent=[0, π, d, 0])
        axc.clabel(CS, CS.levels, inline=True, fmt='%4.0f', fontsize=10)
        axc.set_title('Cross-polarized')
        axc.set_xlabel('Antenna orientation θ\n(radians)')

        fig, (axp, axc) = plt.subplots(1, 2, sharex=True, sharey=True)
        axp.imshow(dB(power[:, :, 1]), extent=[0, π, d, 0])
        axp.set_aspect(aspect="auto")
        axp.set_xticks(np.linspace(0, 1, 5) * π)
        axp.set_xticklabels(['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$',
                             r'$\frac{3\pi}{4}$', r'$\pi$'])
        axp.set_title('Cross-polarized, anisotropic')
        axp.set_xlabel(r'Antenna orientation $\θ$ (radians)')
        axp.set_ylabel('Depth (m)')
        axc.imshow(dB(iso_power[:, :, 1]), extent=[0, π, d, 0])
        axc.set_aspect(aspect="auto")
        ax1 = axc.twinx()
        phase_off = ϕ(ε[0], ε[1], freq, depths)
        ax1.set_ylim(phase_off[-1], 0)
        phase_iters = np.floor(phase_off[-1] / π)
        ax1.set_yticks(np.arange(phase_iters + 1) * π)
        ax1.set_yticklabels(
            ['0', 'π'] + [
                '{:d}π'.format(i) for i in range(2, int(phase_iters) + 1)])
        axc.set_title('Cross-polarized, isotropic')
        axc.set_xlabel(r'Antenna orientation $\θ$ (radians)')
        ax1.set_ylabel(r'Phase offset $\phi$ (Radians)')

        fig, (axp, axc) = plt.subplots(1, 2, sharex=True, sharey=True)
        axp.imshow(dB(power[:, :, 0]), extent=[0, π, d, 0])
        axp.set_aspect(aspect="auto")
        axp.set_xticks(np.linspace(0, 1, 5) * π)
        axp.set_xticklabels(['0', r'$\frac{\pi}{4}$',
                             r'$\frac{\pi}{2}$',
                             r'$\frac{3\pi}{4}$', r'$\pi$'])
        axp.set_title('Co-polarized, anisotropic')
        axp.set_xlabel(r'Antenna orientation $\θ$ (radians)')
        axp.set_ylabel('Depth (m)')
        axc.imshow(dB(iso_power[:, :, 0]), extent=[0, π, d, 0])
        axc.set_aspect(aspect="auto")
        ax1 = axc.twinx()
        phase_off = ϕ(ε[0], ε[1], freq, depths)
        ax1.set_ylim(phase_off[-1], 0)
        phase_iters = np.floor(phase_off[-1] / π)
        ax1.set_yticks(np.arange(phase_iters + 1) * π)
        ax1.set_yticklabels(
            ['0', 'π'] + [
                '{:d}π'.format(i)for i in range(2, int(phase_iters) + 1)])
        axc.set_title('Co-polarized, isotropic')
        axc.set_xlabel(r'Antenna orientation $\θ$ (radians)')
        ax1.set_ylabel(r'Phase offset $\phi$ (Radians)')


def brisbourne_2019_plots(debug=False):
    """Left panels match the modeling figure from Brisbourne."""
    npts = 2000
    A_xy = np.zeros((npts, 2, 2))
    A_xy[:, 0, 0] = 0.0
    A_xy[:, 1, 1] = 0.3333
    φ_fab, ε, E = φεE(A_xy)

    freq = 300.0e6
    d = 600.
    depths = np.linspace(0, d, npts)
    obs_angles = np.linspace(0, π, 181)
    E_transmit = np.array([1.0e3, 0])

    extent_angle = [0 + obs_angles[1] / 2.0,
                    obs_angles[-2] + (obs_angles[-1] - obs_angles[-2]) / 2.0,
                    d,
                    0]

    # Isotropic scattering
    S = np.zeros((depths.shape[0], 2, 2))
    S[:, 0, 0] = 1.0
    S[:, 1, 1] = 1.0
    S = S * 1.0e-4  # -40 dB returns

    # Isotropic scattering
    S_aniso = np.zeros((depths.shape[0], 2, 2))
    S_aniso[:, 0, 0] = 1.0e0
    S_aniso[:, 1, 1] = 1.0e-1
    S_aniso = S_aniso * 1.0e-4  # -40 dB returns

    power = fujita_model_εS(E_transmit, depths, obs_angles,
                            ε, S, φ_fab, freq)
    power_anisoscatter = fujita_model_εS(E_transmit, depths, obs_angles,
                                         np.ones((2,)) * ε_prime, S_aniso,
                                         φ_fab, freq)
    power_bothaniso = fujita_model_εS(E_transmit, depths, obs_angles,
                                      ε, S_aniso, φ_fab, freq)

    fig, ((axpb, axcb),
          (axps, axcs),
          (axpa, axca)) = plt.subplots(3, 2,
                                       sharex=True, sharey=True,
                                       figsize=(6, 9))

    # Birefringence only
    clim = 5.0
    alim = 20.0
    axpb.imshow(power_anomaly(power[:, :, 0]),
                extent=[0, π, d, 0], vmin=-clim, vmax=clim)
    axpb.set_aspect(aspect="auto")
    axpb.set_title('Power')
    axcb.imshow(angle(power, obs_angles),
                extent=extent_angle,
                vmin=-alim, vmax=alim)
    axcb.set_aspect(aspect="auto")
    axcb.set_title('Angle')

    # Scattering only
    axps.imshow(power_anomaly(power_anisoscatter[:, :, 0]),
                extent=[0, π, d, 0], vmin=-clim, vmax=clim)
    axps.set_aspect(aspect="auto")
    axps.set_ylabel('Depth (m)')
    axcs.imshow(angle(power_anisoscatter, obs_angles),
                extent=extent_angle,
                vmin=-alim, vmax=alim)
    axcs.set_aspect(aspect="auto")

    # Anisotropic birefringence and scattering
    axpa.imshow(power_anomaly(power_bothaniso[:, :, 0]),
                extent=[0, π, d, 0], vmin=-clim, vmax=clim)
    axpa.set_aspect(aspect="auto")
    axpa.set_xticks(np.linspace(0, 1, 5) * π)
    axpa.set_xticklabels(['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$',
                          r'$\frac{3\pi}{4}$', r'$\pi$'])
    axpa.set_xlabel('Antenna orientation θ\n(radians)')
    axca.imshow(angle(power_bothaniso, obs_angles),
                extent=extent_angle,
                vmin=-alim, vmax=alim)
    # plt.colorbar(cm, cax=axcca)
    axca.set_aspect(aspect="auto")
    axca.set_xlabel('Antenna orientation θ\n(radians)')

    # These axes are annoying if we have to check values
    if not debug:
        for axc in [axcb, axcs, axca]:
            ax1 = axc.twinx()
            phase_off = ϕ(ε[:, 0], ε[:, 1], freq, depths)
            ax1.set_ylim(abs(phase_off[-1]), 0)
            phase_iters = np.floor(abs(phase_off[-1] / π))
            ax1.set_yticks(np.arange(phase_iters + 1) * π)
            ax1.set_yticklabels(['0', 'π'] + [
                '{:d}π'.format(i) for i in range(2, int(phase_iters) + 1)])
            if axc is axcs:
                ax1.set_ylabel(r'Phase offset $\phi$ (Radians)')


def crossovers(debug=False):
    """Left panels match the modeling figure from Brisbourne."""
    # Number of observations with depth
    npts = 2000

    # Define the fabric
    A_xy = np.zeros((npts, 2, 2))
    A_xy[:, 0, 0] = 1.0
    A_xy[:, 1, 1] = 1.0

    # Convert the fabric to dielectric properties
    φ_fab, ε, _ = φεE(A_xy)

    # Radar frequency
    freq = 180.0e6

    # Thickness of ice column
    d = 2000.
    depths = np.linspace(0, d, npts)

    # Change 181 for finer or coarser angular resolution
    obs_angles = np.linspace(0, π, 181)
    # Handle fenceposts
    extent_angle = [0 + obs_angles[1] / 2.0,
                    obs_angles[-2] + (obs_angles[-1] - obs_angles[-2]) / 2.0,
                    d,
                    0]

    # This gets normalized, but need a value. This is co- and cross- transmit power
    E_transmit = np.array([1.0e3, 0])

    # Isotropic scattering
    S = np.zeros((depths.shape[0], 2, 2))
    S[:, 0, 0] = 1.0
    S[:, 1, 1] = 1.0
    S = S * 1.0e-4  # -40 dB returns

    # Isotropic scattering
    S_aniso = np.zeros((depths.shape[0], 2, 2))
    S_aniso[:, 0, 0] = 1.0e0
    S_aniso[:, 1, 1] = 1.0e0
    S_aniso[depths > 800., 1, 1] = 1.0e-1
    S_aniso = S_aniso * 1.0e-4  # -40 dB returns

    power = fujita_model_εS(E_transmit, depths, obs_angles,
                            ε, S, φ_fab, freq)

    power_anisoscatter = fujita_model_εS(E_transmit, depths, obs_angles,
                                         np.ones((2,)) * ε_prime, S_aniso,
                                         φ_fab, freq)
    power_bothaniso = fujita_model_εS(E_transmit, depths, obs_angles,
                                      ε, S_aniso, φ_fab, freq)

    fig, ((axpb, axcb),
          (axps, axcs),
          (axpa, axca)) = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(6, 9))

    # Birefringence only
    clim = 5.0
    alim = 20.0
    axpb.imshow(power_anomaly(power[:, :, 0]),
                extent=[0, π, d, 0], vmin=-clim, vmax=clim)
    axpb.set_aspect(aspect="auto")
    axpb.set_title('Power')
    axcb.imshow(angle(power, obs_angles),
                extent=extent_angle,
                vmin=-alim, vmax=alim)
    axcb.set_aspect(aspect="auto")
    axcb.set_title('Angle')

    # Scattering only
    axps.imshow(power_anomaly(power_anisoscatter[:, :, 0]),
                extent=[0, π, d, 0], vmin=-clim, vmax=clim)
    axps.set_aspect(aspect="auto")
    axps.set_ylabel('Depth (m)')
    axcs.imshow(angle(power_anisoscatter, obs_angles),
                extent=extent_angle,
                vmin=-alim, vmax=alim)
    axcs.set_aspect(aspect="auto")

    # Anisotropic birefringence and scattering
    axpa.imshow(power_anomaly(power_bothaniso[:, :, 0]),
                extent=[0, π, d, 0], vmin=-clim, vmax=clim)
    axpa.set_aspect(aspect="auto")
    axpa.set_xticks(np.linspace(0, 1, 5) * π)
    axpa.set_xticklabels(['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$',
                          r'$\frac{3\pi}{4}$', r'$\pi$'])
    axpa.set_xlabel('Antenna orientation θ\n(radians)')
    axca.imshow(angle(power_bothaniso, obs_angles),
                extent=extent_angle,
                vmin=-alim, vmax=alim)
    axca.set_aspect(aspect="auto")
    axca.set_xlabel('Antenna orientation θ\n(radians)')

    # These axes are annoying if we have to check values
    if not debug:
        for axc in [axcb, axcs, axca]:
            ax1 = axc.twinx()
            phase_off = ϕ(ε[:, 0], ε[:, 1], freq, depths)
            ax1.set_ylim(abs(phase_off[-1]), 0)
            phase_iters = np.floor(abs(phase_off[-1] / π))
            phase_iters = max(phase_iters, 1)
            ax1.set_yticks(np.arange(phase_iters + 1) * π)
            ax1.set_yticklabels(['0', 'π'] + [
                '{:d}π'.format(i) for i in range(2, int(phase_iters) + 1)])
            if axc is axcs:
                ax1.set_ylabel(r'Phase offset $\phi$ (Radians)')


if __name__ == '__main__':
    crossovers()
    fujita_2006_plots(debug=False)
    brisbourne_2019_plots(debug=False)
    plt.show()
