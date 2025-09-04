import numpy as np
from scipy.fft import fft2, ifft2


def littlewood_paley_sum(filter, level=0):
    phi, psi = filter['phi'], filter['psi']
    littlewood_sum = np.abs(phi['levels'][level].squeeze())**2
    for n1 in range(len(psi)):
        littlewood_sum += np.abs(psi[n1]['levels'][level].squeeze())**2

    return littlewood_sum

def renormalize_filters_to_tight(filters, eps=1e-8):
    phi,psi = filters['phi'], filters['psi']

    for level in range(len(phi['levels'])):
        littlewood_sum = littlewood_paley_sum(filters, level)
        phi['levels'][level] = phi['levels'][level]*(1/(littlewood_sum**0.5))
        for n1 in range(len(psi)):
            psi[n1]['levels'][level] = psi[n1]['levels'][level]*(1/(littlewood_sum**0.5))

def periodize_morlet_filter_fft(x, res):
    """
        Parameters
        ----------
        x : numpy array
            signal to periodize in Fourier
        res :
            resolution to which the signal is cropped.

        Returns
        -------
        crop : numpy array
            It returns a crop version of the filter, assuming that
             the convolutions will be done via compactly supported signals.
    """
    M = x.shape[0]
    N = x.shape[1]

    crop = np.zeros((M // 2 ** res, N // 2 ** res), x.dtype)

    mask = np.ones(x.shape, np.float32)
    len_x = int(M * (1 - 2 ** (-res)))
    start_x = int(M * 2 ** (-res - 1))
    len_y = int(N * (1 - 2 ** (-res)))
    start_y = int(N * 2 ** (-res - 1))
    mask[start_x:start_x + len_x,:] = 0
    mask[:, start_y:start_y + len_y] = 0
    x = np.multiply(x,mask)

    for k in range(int(M / 2 ** res)):
        for l in range(int(N / 2 ** res)):
            for i in range(int(2 ** res)):
                for j in range(int(2 ** res)):
                    crop[k, l] += x[k + i * int(M / 2 ** res), l + j * int(N / 2 ** res)]

    return crop

def periodize_mayer_filter_fft(x, res):
    M, N = x.shape
    m, n = int(M/2**res), int(N/2**res)
    out = np.zeros((m, n), dtype=x.dtype)
    for i in range(2**res):
        for j in range(2**res):
            out += x[i*m:(i+1)*m, j*n:(j+1)*n]
    return out


def morlet_2d(M, N, sigma, theta, xi, slant=0.5, offset=0):
    """
        Computes a 2D Morlet filter.
        A Morlet filter is the sum of a Gabor filter and a low-pass filter
        to ensure that the sum has exactly zero mean in the temporal domain.
        It is defined by the following formula in space:
        psi(u) = g_{sigma}(u) (e^(i xi^T u) - beta)
        where g_{sigma} is a Gaussian envelope, xi is a frequency and beta is
        the cancelling parameter.

        Parameters
        ----------
        M, N : int
            spatial sizes
        sigma : float
            bandwidth parameter
        xi : float
            central frequency (in [0, 1])
        theta : float
            angle in [0, pi]
        slant : float, optional
            parameter which guides the elipsoidal shape of the morlet
        offset : int, optional
            offset by which the signal starts

        Returns
        -------
        morlet_fft : ndarray
            numpy array of size (M, N)
    """
    wv = gabor_2d(M, N, sigma, theta, xi, slant, offset)
    wv_modulus = gabor_2d(M, N, sigma, theta, 0, slant, offset)
    K = np.sum(wv) / np.sum(wv_modulus)

    mor = wv - K * wv_modulus
    return mor


def gabor_2d(M, N, sigma, theta, xi, slant=1.0, offset=0):
    """
        Computes a 2D Gabor filter.
        A Gabor filter is defined by the following formula in space:
        psi(u) = g_{sigma}(u) e^(i xi^T u)
        where g_{sigma} is a Gaussian envelope and xi is a frequency.

        Parameters
        ----------
        M, N : int
            spatial sizes
        sigma : float
            bandwidth parameter
        xi : float
            central frequency (in [0, 1])
        theta : float
            angle in [0, pi]
        slant : float, optional
            parameter which guides the elipsoidal shape of the morlet
        offset : int, optional
            offset by which the signal starts

        Returns
        -------
        morlet_fft : ndarray
            numpy array of size (M, N)
    """
    gab = np.zeros((M, N), np.complex64)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32)
    R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float32)
    D = np.array([[1, 0], [0, slant * slant]])
    curv = np.dot(R, np.dot(D, R_inv)) / ( 2 * sigma * sigma)

    for ex in [-2, -1, 0, 1, 2]:
        for ey in [-2, -1, 0, 1, 2]:
            [xx, yy] = np.mgrid[offset + ex * M:offset + M + ex * M, offset + ey * N:offset + N + ey * N]
            arg = -(curv[0, 0] * np.multiply(xx, xx) + (curv[0, 1] + curv[1, 0]) * np.multiply(xx, yy) + curv[
                1, 1] * np.multiply(yy, yy)) + 1.j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))
            gab += np.exp(arg)

    norm_factor = (2 * 3.1415 * sigma * sigma / slant)
    gab /= norm_factor

    return gab

def build_morlet_filters(M, N, J, L=8, downsample=True, dilation_optimization=True,
                sigma0=None, theta0=None, xi0=None, slant0=None):
    """
        Builds in Fourier the Morlet filters used for the scattering transform.
        Each single filter is provided as a dictionary with the following keys:
        * 'j' : scale
        * 'theta' : angle used
        Parameters
        ----------
        M, N : int
            spatial support of the input
        J : int
            logscale of the scattering
        L : int, optional
            number of angles used for the wavelet transform
        downsample : bool, optional
            If True, the filters are cropped to the size of the
            downsampled signal at each resolution.
        dilation_optimization : bool, optional
            If True, the filters are designed to be used with dilation
            optimization. If False, the filters are designed to be used
            without dilation optimization.
        Returns
        -------
        filters : list
            A two list of dictionary containing respectively the low-pass and
             wavelet filters.
        Notes
        -----
        The design of the filters is optimized for the value L = 8.
    """
    filters = {}
    filters['psi'] = []

    

    for j in range(J):
        for theta in range(L):
            psi = {'levels': [], 'j': j, 'theta': theta}

            sigma = 0.8 * 2**j if sigma0 is None else sigma0
            theta = (int(L-L/2-1)-theta) * 2 * np.pi / L if theta0 is None else theta0
            xi = 3.0 / 4.0 * np.pi / 2**j if xi0 is None else xi0
            slant = 4.0 / L if slant0 is None else slant0


            psi_signal = morlet_2d(M=M, N=N, sigma=sigma,
                theta=theta,
                xi=xi, slant=slant)
            psi_signal_fourier = np.real(fft2(psi_signal))
            # drop the imaginary part, it is zero anyway
            psi_levels = []
            for res in range(min(j + 1, max(J - 1, 1))) if dilation_optimization else range(J):
                if downsample:
                    psi_levels.append(periodize_morlet_filter_fft(psi_signal_fourier, res))
                else:
                    psi_levels.append(psi_signal_fourier)
            psi['levels'] = psi_levels
            filters['psi'].append(psi)

    phi_signal = gabor_2d(M=M, N=N, sigma=0.8 * 2**(J-1), theta=0, xi=0)
    phi_signal_fourier = np.real(fft2(phi_signal))
    # drop the imaginary part, it is zero anyway
    filters['phi'] = {'levels': [], 'j': J}
    for res in range(J):
        if downsample:
            filters['phi']['levels'].append(
                periodize_morlet_filter_fft(phi_signal_fourier, res))
        else:
            filters['phi']['levels'].append(
                periodize_morlet_filter_fft(phi_signal_fourier, 0))


    return filters



# ------------------------------third version of meyer filters---------------------------------

SQRT2 = np.sqrt(2.0)
_TAU = np.float32(32 * np.finfo(np.float32).eps)  # ~3.8e-6, scale once

def _radial_lowpass_L(u):
    """
    Lowpass prototype on normalized radius u (after square-normalization).
    L(u) = 1 for u<=0.5; cosine rolloff to 0 by u=1.  We nudge the top
    boundary so L(1) is a hair > 0 instead of exactly 0.
    """
    L = np.zeros_like(u, dtype=np.float32)
    m1 = (u <= np.float32(0.5))
    L[m1] = 1.0

    m2 = (u > np.float32(0.5)) & (u < np.float32(1.0) + _TAU)
    # compute t = log2(2u) but clamp 2u to [1, 2 - tau] (nudge off the boundary)
    x = 2.0 * np.maximum(u[m2], np.float32(1e-12))
    x = np.clip(x, np.float32(1.0), np.float32(2.0 - _TAU))
    t = np.log2(x)  # in [0, 1 - tiny]
    L[m2] = np.cos(0.5 * np.pi * t).astype(np.float32)
    return L

def _radial_bandpass_B(u):
    """
    Bandpass prototype on normalized radius u; support (0.5, 2).
    We clamp away from 0.5 and 2 by ±tau so boundary pixels are assigned
    to the nearest band instead of hitting cos==0 exactly.
    """
    B = np.zeros_like(u, dtype=np.float32)
    m = (u > np.float32(0.5) - _TAU) & (u < np.float32(2.0) + _TAU)
    z = np.maximum(u[m], np.float32(1e-12))
    z = np.clip(z, np.float32(0.5 + _TAU), np.float32(2.0 - _TAU))
    t = np.log2(z)  # in [-1 + tiny, 1 - tiny]
    B[m] = np.cos(0.5 * np.pi * t).astype(np.float32)
    return B


def L_with_pad(v, pad):
    v = np.asarray(v, dtype=np.float32)
    L = np.zeros_like(v, dtype=np.float32)
    # flat-1 up to 0.5 + pad
    L[v <= (0.5 + pad)] = 1.0
    # cosine transition on (0.5 - pad, 1 + pad)
    m = (v > (0.5 - pad)) & (v < (1.0 + pad))
    x = np.clip(2.0 * np.maximum(v[m], np.float32(1e-12)),
                np.float32(1.0), np.float32(2.0 - 1e-6))
    t = np.log2(x)
    L[m] = np.cos(0.5*np.pi * t).astype(np.float32)
    return L

def B_with_pad(w, pad):
    w = np.asarray(w, dtype=np.float32)
    B = np.zeros_like(w, dtype=np.float32)
    # support (0.5 - pad, 2 + pad)
    m = (w > (0.5 - pad)) & (w < (2.0 + pad))
    z = np.clip(np.maximum(w[m], np.float32(1e-12)),
                np.float32(0.5 + 1e-6), np.float32(2.0 - 1e-6))
    t = np.log2(z)
    B[m] = np.cos(0.5*np.pi * t).astype(np.float32)
    return B



# 1) Keep full angle; don't fold to [0, π)
def _fftfreq2_rad(M, N):
    wx = 2*np.pi*np.fft.fftfreq(M)
    wy = 2*np.pi*np.fft.fftfreq(N)
    WX, WY = np.meshgrid(wy, wx)
    R = np.sqrt(WX**2 + WY**2)
    Theta = np.arctan2(WY, WX)              # (-π, π]
    return R, Theta

# 2) π-periodic wrap for angular difference
def _wrap_period(x, period):
    return (x + 0.5*period) % period - 0.5*period  # in (-period/2, period/2]

def _angular_windows_L2norm(Theta, L, eps=1e-12, pad=1e-6):
    centers = [(ell*np.pi)/L for ell in range(L)]

    def base(u):
        out = np.zeros_like(u, dtype=np.float32)
        m = (np.abs(u) <= 1.0 + pad)
        uc = np.clip(u[m], -1.0, 1.0)       # keep cosine arg in [-1,1]
        out[m] = np.cos(0.5*np.pi * uc)
        return out

    # π-periodic difference, then scale so neighbors meet at |u|=1
    G = []
    for c in centers:
        d = _wrap_period(Theta - c, np.pi)  # now in (-π/2, π/2]
        u = (L/np.pi) * d                   # support target ~|u|<=1
        G.append(base(u))
    G = np.stack(G, axis=0).astype(np.float32)   # [L, M, N]

    denom = np.sqrt(np.sum(G**2, axis=0) + eps)
    zeros = denom < np.sqrt(eps)*2
    if np.any(zeros):
        kmax = np.argmax(G[:, zeros], axis=0)
        G[:, zeros] = 0.0
        G[kmax, zeros] = 1.0
        denom[zeros] = 1.0

    return [(G[k]/denom).astype(np.float32) for k in range(L)]


def build_meyer_filters_fourier(M, N, J, L, downsample=True, dilation_optimization=True):
    R, Theta = _fftfreq2_rad(M, N)



    # s = (R / np.pi).astype(np.float32)               # radius / π
    # u = ((R/np.pi)/np.sqrt(2.0)).astype(np.float32)

    # we want c s.t for all 0<=R<0.5*sqrt(M^2 + N^2) : u = c*R <= 1
    # choosing 0<= c <= 1/0.5*sqrt(M^2 + N^2) we have : c*R <= 1




    # for all 0<=R<0.5*sqrt(M^2 + N^2): c < 1/R 
    # c < 1/min(R)
    # 2^(-j-1) <= c*R <=  2^(-j+1)
    # 2^(-j-1)/R <= c <= 2^(-j+1)/R for all j=0,...,J

    # choosing c >= 1/2(0.5*sqrt(M^2 + N^2)) then:
    # c*R >= R/2
    #c = 1/(2**J)*(0.5*np.sqrt(M**2 + N**2))
    R_max = np.pi * np.sqrt(2.0)  # max R is pi*sqrt(2) (corner of freq square)
    R_min = min(2*np.pi/M, 2*np.pi/N)  # min nonzero R is 2π/min(M,N) (side of freq square)
    lower_bound_for_c = 1/((2**(J+1))*R_max)
    upper_bound_for_c = min(1/(R_max), 1/((2**(J))*R_min))
    t = 0.1
    c = t*lower_bound_for_c + (1-t)*upper_bound_for_c
    u = c*R

    R_step = 2*np.pi / float(min(M, N))
    u_step = c * R_step
    kappa  = 0.75  # ~half-to-one pixel; tweak 0.5–1.0 if needed

    # Quick probe of L on your u-grid
    vals = L_with_pad((2.0**J) * u, kappa * u_step * (2.0**J))
    # vals = _radial_lowpass_L((2.0**J) * u)   # or your exact L function call
    print("phi pre-periodize: min/max =", float(vals.min()), float(vals.max()),
        "   phi at DC =", float(vals[0,0]))


    # pad_u = 1.5 * np.sqrt(2.0) / float(min(M, N))
    filters = {'psi': [], 'phi': {'levels': [], 'j': J}}

    # Father: amplitude = L( 2^J * u )
    amp_phi = _radial_lowpass_L((2.0**J) * u).astype(np.complex64)
    for res in range(J):
        phi_res = periodize_mayer_filter_fft(amp_phi, res) if downsample else amp_phi.copy()
        filters['phi']['levels'].append(phi_res.real)  # drop imag part, should be 0

    # Angular: L²-normalized so sum_ℓ Γ_ℓ(θ)^2 == 1
    gammas = _angular_windows_L2norm(Theta, L)

    # Mothers: amplitude = B( 2^j * u ) * Γ_ℓ(θ)
    #B(2^j *u) != 0 only for u in (2^(-j-1), 2^(-j+1))
    for j in range(J):
        amp_beta = B_with_pad((2.0**j) * u, kappa * u_step * (2.0**j)).astype(np.complex64)
        # amp_beta = _radial_bandpass_B((2.0**j) * u).astype(np.float32)
        for ell in range(L):
            band = (amp_beta * gammas[ell]).astype(np.complex64)
            psi = {'levels': [], 'j': j, 'theta': ell}
            res_range = range(min(j+1, max(J-1, 1))) if dilation_optimization else range(J)
            for res in res_range:
                psi_res = periodize_mayer_filter_fft(band, res) if downsample else band.copy()
                psi['levels'].append(psi_res.real)  # drop imag part, should be 0
            filters['psi'].append(psi)

    
    return filters



def filter_bank(M, N, J, L=8, downsample=True, dilation_optimization=True, tighten=False, filter_type='morlet',
                sigma0=None, theta0=None, xi0=None, slant0=None):
    # Build filters
    filters = None
    if filter_type == 'morlet':
        filters = build_morlet_filters(M, N, J, L, downsample, dilation_optimization, sigma0, theta0, xi0, slant0)
    elif filter_type == 'meyer':
        filters = build_meyer_filters_fourier(M, N, J, L, downsample, dilation_optimization)
    else:
        raise ValueError(f"Unknown filter_type '{filter_type}'")
    
    if tighten:
        renormalize_filters_to_tight(filters)

    # ANG check
    wx = 2*np.pi*np.fft.fftfreq(M); wy = 2*np.pi*np.fft.fftfreq(N)
    WX, WY = np.meshgrid(wy, wx)
    Theta = (np.arctan2(WY, WX) + np.pi) % np.pi
    gammas = _angular_windows_L2norm(Theta, L)
    S_ang = sum(G**2 for G in gammas)
    print("ANG min/max:", float(S_ang.min()), float(S_ang.max()))

    # RAD check (square-aware)
    R = np.sqrt(WX**2 + WY**2)
    u = (R/np.pi)/np.sqrt(2.0)
    pad_u = 1.5 * np.sqrt(2.0) / float(min(M, N))


    S_rad = _radial_lowpass_L((2.0**J) * u)**2
    for j in range(J+1):
        S_rad += _radial_bandpass_B((2.0**j) * u)**2
    print("RAD min/max:", float(S_rad.min()), float(S_rad.max()))
    idx = np.unravel_index(np.argmin(S_rad), S_rad.shape)
    u0 = float(u[idx])
    vals = [float(_radial_bandpass_B(np.array([(2.0**j)*u0]))[0]**2) for j in range(J+1)]
    print("min S_rad at", idx, "  u=", u0,
        "  L^2=", float(_radial_lowpass_L(np.array([(2.0**J)*u0]))[0]**2),
        "  B^2 per j:", vals, "  sum=", sum(vals)+float(_radial_lowpass_L(np.array([(2.0**J)*u0]))[0]**2))


    # Full LP check
    phi0 = filters['phi']['levels'][0]
    LP = np.abs(phi0)**2
    for w in filters['psi']:
        if len(w['levels'])>0:
            LP += np.abs(w['levels'][0])**2
    print("LP min/max/mean abs dev from 1:",
        float(LP.min()), float(LP.max()), float(np.mean(np.abs(LP-1))))


    R_max = np.pi * np.sqrt(2.0)  # max R is pi*sqrt(2) (corner of freq square)
    R_min = min(2*np.pi/M, 2*np.pi/N)  # min nonzero R is 2π/min(M,N) (side of freq square)
    lower_bound_for_c = 1/((2**(J+1))*R_max)
    upper_bound_for_c = min(1/(R_max), 1/((2**(J))*R_min))
    t = 0.1
    c = t*lower_bound_for_c + (1-t)*upper_bound_for_c
    Rt = 0.75 * (2.0**(-J)) / c
    Rgrid = np.sqrt(WX**2 + WY**2)
    idx = np.unravel_index(np.argmin(np.abs(Rgrid - Rt)), Rgrid.shape)
    print("transition pixel u, L(2^J u) =", float((c*Rgrid[idx])), 
        float(_radial_lowpass_L((2.0**J)*c*Rgrid[idx])))
    
    return filters

__all__ = ['filter_bank']




# -------------first version of meyer filters, kept for reference-----------------

# def _smoothstep01(t):
#     t = np.clip(t, 0.0, 1.0)
#     # C^∞-like smoothstep; any smooth monotone [0,1]->[0,1] is fine
#     return np.exp(-1.0/(t+1e-12)) / (np.exp(-1.0/(t+1e-12)) + np.exp(-1.0/(1.0-t+1e-12)))

# def _fftfreq2_rad(M, N):
#     wx = 2*np.pi*np.fft.fftfreq(M)  # [-pi, pi)
#     wy = 2*np.pi*np.fft.fftfreq(N)
#     WX, WY = np.meshgrid(wy, wx)    # note: rows vs cols — same convention as your current code
#     R = np.sqrt(WX**2 + WY**2)
#     Theta = np.arctan2(WY, WX)      # [-pi, pi)
#     Theta = (Theta + np.pi) % np.pi  # fold to [0, pi): steerable symmetry
#     return R, Theta

# def _meyer_lowpass(R_scaled, r0=np.pi/2, r1=np.pi):
#     t = (R_scaled - r0) / (r1 - r0)
#     w = _smoothstep01(t)
#     chi = np.cos(0.5*np.pi * w)
#     chi[R_scaled <= r0] = 1.0
#     chi[R_scaled >= r1] = 0.0
#     return chi

# def _meyer_bandpass(R_scaled, r0=np.pi/2, r1=np.pi):
#     t = (R_scaled - r0) / (r1 - r0)
#     w = _smoothstep01(t)
#     beta = np.sin(0.5*np.pi * w)
#     beta[(R_scaled <= r0) | (R_scaled >= r1)] = 0.0
#     return beta

# def _wrap_to_pi(x):
#     return (x + np.pi) % (2*np.pi) - np.pi

# def _angular_windows(Theta, L):
#     gammas = []
#     for ell in range(L):
#         c = (ell*np.pi)/L
#         u = _wrap_to_pi(Theta - c) * (L/np.pi)  # support intended in [-1,1]
#         g = np.zeros_like(Theta, dtype=np.float32)
#         m = (np.abs(u) <= 1.0)
#         if ell % 2 == 0:
#             g[m] = np.cos(0.5*np.pi * u[m])
#         else:
#             g[m] = np.sin(0.5*np.pi * u[m])
#         gammas.append(g.astype(np.float32))
#     return gammas

# def build_meyer_filters_fourier(M, N, J, L, downsample=True, dilation_optimization=True):
#     R, Theta = _fftfreq2_rad(M, N)
#     filters = {'psi': [], 'phi': {'levels': [], 'j': J}}

#     # φ̂_J(ω) = χ(2^{-J}|ω|), provide all res levels
#     chiJ_full = _meyer_lowpass(R*(2**J)).astype(np.complex64)
#     for res in range(J):
#         if downsample:
#             phi_res = periodize_filter_fft(chiJ_full, res)   # make sure periodize preserves complex dtype
#         else:
#             phi_res = chiJ_full.copy()
#             # printing the max real value and max imag value of each filter
#             print(f"j={J}, res={res}, max real value = {np.max(phi_res.real)}, max imag value = {np.max(phi_res.imag)}")
#         filters['phi']['levels'].append((phi_res))

#     # Angular windows shared across scales
#     gammas = _angular_windows(Theta, L)

#     for j in range(J):
#         beta_full = _meyer_bandpass(R*(2**j)).astype(np.float32)
#         for ell in range(L):
#             band = (beta_full * gammas[ell]).astype(np.complex64)
#             psi = {'levels': [], 'j': j, 'theta': ell}
#             res_range = range(min(j+1, max(J-1, 1))) if dilation_optimization else range(J)
#             for res in res_range:
#                 if downsample:
#                     psi_res = periodize_filter_fft(band, res)
#                 else:
#                     psi_res = band.copy()
#                 psi['levels'].append((psi_res))
#                 # printing the max real value and max imag value of each filter
#                 print(f"j={j}, ell={ell}, res={res}, max real value = {np.max(psi_res.real)}, max imag value = {np.max(psi_res.imag)}")
#             filters['psi'].append(psi)

#     return filters


# def filter_bank(M, N, J, L=8, downsample=True, dilation_optimization=True, tighten=False, 
#                 sigma0=None, theta0=None, xi0=None, slant0=None):
#     return build_meyer_filters_fourier(M, N, J, L, downsample, dilation_optimization)


# ------------------------------second version of meyer filters---------------------------------

# # ---------- utilities ----------
# def _smoothstep01(t):
#     t = np.clip(t, 0.0, 1.0)
#     # C^∞-like smoothstep
#     return np.exp(-1.0/(t+1e-12)) / (np.exp(-1.0/(t+1e-12)) + np.exp(-1.0/(1.0-t+1e-12)))

# def _fftfreq2_rad(M, N):
#     wx = 2*np.pi*np.fft.fftfreq(M)  # [-pi, pi)
#     wy = 2*np.pi*np.fft.fftfreq(N)
#     WX, WY = np.meshgrid(wy, wx)    # (row, col)
#     R = np.sqrt(WX**2 + WY**2)      # radius
#     Theta = np.arctan2(WY, WX)      # [-pi, pi)
#     Theta = (Theta + np.pi) % np.pi # fold to [0, pi)
#     return R, Theta

# def _wrap_to_pi(x):
#     return (x + np.pi) % (2*np.pi) - np.pi

# # ---------- radial prototypes: lowpass L and bandpass B = L(ρ/2) - L(ρ) ----------
# def _meyer_lowpass_proto(X, r0=np.pi/2, r1=np.pi):
#     """ L(X): 1 for X<=r0, smooth to 0 by r1. """
#     t = (X - r0) / (r1 - r0)
#     w = _smoothstep01(t)
#     L = np.cos(0.5*np.pi * w)
#     L[X <= r0] = 1.0
#     L[X >= r1] = 0.0
#     return L

# def _meyer_banddiff_proto(X, r0=np.pi/2, r1=np.pi):
#     """ B(X) = L(X/2) - L(X), guaranteed in [0,1] with telescoping sum. """
#     Lx   = _meyer_lowpass_proto(X, r0, r1)
#     Lx2  = _meyer_lowpass_proto(X/2.0, r0, r1)
#     B    = Lx2 - Lx
#     B[B < 0] = 0.0  # numerical guard
#     return B

# # ---------- angular windows with local L2 normalization ----------
# def _angular_windows_normalized(Theta, L, eps=1e-12):
#     centers = [(ell*np.pi)/L for ell in range(L)]
#     # base symmetric window (support |u|<=1)
#     def base(u):
#         out = np.zeros_like(u, dtype=np.float32)
#         m = (np.abs(u) <= 1.0)
#         out[m] = np.cos(0.5*np.pi * u[m])
#         return out
#     # stack all raw lobes
#     G = []
#     for c in centers:
#         u = _wrap_to_pi(Theta - c) * (L/np.pi)  # [-1,1] ≈ support
#         G.append(base(u))
#     G = np.stack(G, axis=0)  # [L, M, N]
#     denom = np.sqrt(np.sum(G**2, axis=0) + eps)  # [M, N]
#     Gammas = [ (G[k] / denom).astype(np.float32) for k in range(L) ]  # sum of squares = 1
#     return Gammas

# # ---------- complex-safe periodization (no masks) ----------
# def periodize_filter_fft(x, res):
#     M, N = x.shape
#     m, n = int(M/2**res), int(N/2**res)
#     out = np.zeros((m, n), dtype=x.dtype)
#     for i in range(2**res):
#         for j in range(2**res):
#             out += x[i*m:(i+1)*m, j*n:(j+1)*n]
#     return out

# # ---------- main builder ----------
# def build_meyer_filters_fourier(M, N, J, L, downsample=True, dilation_optimization=True):
#     """
#     Kymatio-compatible: returns dict with Fourier-domain filters.
#     Radial: telescoping Meyer (exact LP=1 radially).
#     Angular: locally normalized (exact sum_{ℓ} Γ_ℓ^2 = 1 at each θ).
#     """
#     R, Theta = _fftfreq2_rad(M, N)
#     filters = {'psi': [], 'phi': {'levels': [], 'j': J}}

#     # amplitudes (not squared): take sqrt of the proto windows
#     amp_phi = np.sqrt(_meyer_lowpass_proto(R / (2**J))).astype(np.complex64)

#     # lowpass levels
#     for res in range(J):
#         phi_res = periodize_filter_fft(amp_phi, res) if downsample else amp_phi.copy()
#         filters['phi']['levels'].append(phi_res)

#     # angular partition with Γ_ℓ normalized so sum Γ_ℓ^2 = 1
#     gammas = _angular_windows_normalized(Theta, L)  # list of [M,N] float32

#     # band-passes
#     for j in range(J):
#         amp_beta = np.sqrt(_meyer_banddiff_proto(R / (2**j))).astype(np.float32)  # amplitude
#         for ell in range(L):
#             band = (amp_beta * gammas[ell]).astype(np.complex64)
#             psi = {'levels': [], 'j': j, 'theta': ell}
#             res_range = range(min(j+1, max(J-1, 1))) if dilation_optimization else range(J)
#             for res in res_range:
#                 psi_res = periodize_filter_fft(band, res) if downsample else band.copy()
#                 psi['levels'].append(psi_res)
#             filters['psi'].append(psi)

#     return filters

# def filter_bank(M, N, J, L=8, downsample=True, dilation_optimization=True, tighten=False, 
#                 sigma0=None, theta0=None, xi0=None, slant0=None):
#     return build_meyer_filters_fourier(M, N, J, L, downsample, dilation_optimization)

