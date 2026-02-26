"""
Esophageal Peristalsis Model
============================
Python translation of EsophagusModel.nb (Mathematica 13.3)

Model description
-----------------
Simulates esophageal peristalsis using three coupled variables per spatial
lattice point:

  u : ENS (Enteric Nervous System) neural activity
  v : ENS recovery variable
  w : Smooth-muscle pressure (mmHg)

Governing equations (forward Euler):

  du/dt = f_u(u, v) + interaction_uu(u) + S_CNS(t)
  dv/dt = f_v(u, v)
  dw/dt = f_w(u, w) + les_mask · interaction_uw(u) + PEP(t)

Reaction terms:

  f_u(u, v) = 10 · (−u (u − T_ENS A_ENS)(u − 2 A_ENS) − 0.5 v)
  f_v(u, v) = (u − D_ENS v)  if u > 0,  else (−D_ENS v)
  f_w(u, w) = E_SM u − D_SM w + B_SM

Interaction (spatial convolution with kernel K):

  interaction_uu(u)[j] = Σ_i relu(u[i]) · K_IN[N−1+j−i]
  interaction_uw(u)[j] = Σ_i relu(u[i]) · K_MN[N−1+j−i]

Equivalent to: np.convolve(relu_u, K)[N-1 : 2N-1]

Software design note (from original notebook)
---------------------------------------------
Reaction-term and interaction-term parameters are held in global (module-level)
numpy arrays and passed explicitly to functions.  Spatial distributions are
constructed once at start-up and reused across simulation runs.
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Physical parameter settings
# ─────────────────────────────────────────────────────────────────────────────

# System dimensions
SYSTEM_LENGTH = 200.0       # mm  (20 cm total oesophagus)
SIMULATION_LENGTH = 10.0    # s
SIMULATION_LENGTH_LONG_FACTOR = 2

# Spatial ratio parameters
LES_RATIO = 0.2             # LES occupies distal 20 %
CDP_RATIO = 0.5             # cricopharyngeal / distal boundary at 50 %

# ENS reaction parameters (D: decay, T: threshold, A: amplitude)
DENS_BODY = 0.6
DENS_LES = 100.0
TENS_BODY = 0.4
TENS_LES = 0.4
AENS_BODY = 0.6
AENS_LES = 0.6

# Smooth-muscle (SM) reaction parameters
ESM_LOW = 3500.0            # Pa/mmHg, low excitation coupling
ESM_HIGH = 12000.0          # Pa/mmHg, high excitation coupling (at CDP peak)
DSM_BODY = 100.0
DSM_LES = 100.0
BSM_BODY = 0.0
BSM_LES = 0.0

# Law-of-intestine kernel parameters
S_IN = 0.15                 # excitatory kernel strength
D_IN = 4.0                  # kernel delay  (mm)
R_IN = 10.0                 # kernel range  (mm)
I_MN = -50.0                # motor-neuron kernel strength (inhibitory)
R_MN = 100.0                # motor-neuron kernel range   (mm)

# CNS stimulus parameters
S_CNS = 2.0                 # swallow excitatory amplitude
S_CNS_WIDTH = 5.0           # mm
S_CNS_LOCATION = 0.0        # mm (proximal end of oesophagus)
S_CNS_DURATION = 0.5        # s
I_CNS = -4.0                # inhibitory (descending) CNS amplitude
I_CNS_WIDTH = SYSTEM_LENGTH  # mm (full oesophagus)
I_CNS_LOCATION = 0.0        # mm
I_CNS_DURATION = 2.0        # s

# PEP (Peristaltic Esophageal Pressure) amplitude
PEP_HG = 10000.0

# ─────────────────────────────────────────────────────────────────────────────
# Discretisation parameters
# ─────────────────────────────────────────────────────────────────────────────

DX = 0.5    # mm spatial step
DT = 0.01   # s  time step

N_LN = round(SYSTEM_LENGTH / DX)                        # 400
LES_LENGTH_LN = round(N_LN * LES_RATIO)                 # 80
BODY_LENGTH_LN = N_LN - LES_LENGTH_LN                   # 320
CDP_LN = round(N_LN * CDP_RATIO)                        # 200
SIMULATION_LN = round(SIMULATION_LENGTH / DT)           # 1000
SIMULATION_LN_LONG = SIMULATION_LN * SIMULATION_LENGTH_LONG_FACTOR  # 2000


# ─────────────────────────────────────────────────────────────────────────────
# Kernel construction
# ─────────────────────────────────────────────────────────────────────────────

def build_kin(r_in: float, s_in: float, d_in: float,
              n: int, dx: float) -> np.ndarray:
    """
    Excitatory ENS-to-ENS kernel K_IN of length 2·n.

    Non-zero region (Mathematica 1-indexed):
      KIN[n - rINLN + dINLN .. n + rINLN + dINLN] = s_in · dx

    Converted to Python 0-indexed:
      KIN[n - rINLN + dINLN - 1 : n + rINLN + dINLN] = s_in · dx
    """
    r_inln = round(r_in / dx)
    d_inln = round(d_in / dx)
    kin = np.zeros(2 * n)
    lo = n - r_inln + d_inln - 1        # inclusive lower bound (0-indexed)
    hi = n + r_inln + d_inln            # exclusive upper bound (Python slice)
    kin[lo:hi] = s_in * dx
    return kin


def build_kmn(i_mn: float, r_mn: float, n: int, dx: float) -> np.ndarray:
    """
    Inhibitory motor-neuron kernel K_MN of length 2·n.

    Non-zero region (Mathematica 1-indexed):
      KMN[n .. n + rMNLN] = i_mn · dx

    Converted to Python 0-indexed:
      KMN[n - 1 : n + rMNLN] = i_mn · dx
    """
    r_mnln = round(r_mn / dx)
    kmn = np.zeros(2 * n)
    kmn[n - 1 : n + r_mnln] = i_mn * dx
    return kmn


KIN_NORMAL = build_kin(R_IN, S_IN, D_IN, N_LN, DX)
KMN_NORMAL = build_kmn(I_MN, R_MN, N_LN, DX)


# ─────────────────────────────────────────────────────────────────────────────
# Spatial distributions of reaction parameters
# ─────────────────────────────────────────────────────────────────────────────

# DENS: body region then LES region
DENS_LIST: np.ndarray = np.concatenate([
    np.full(BODY_LENGTH_LN, DENS_BODY),
    np.full(LES_LENGTH_LN,  DENS_LES),
])

# TENS: proximal CDP region then distal region
TENS_LIST: np.ndarray = np.concatenate([
    np.full(CDP_LN,          TENS_BODY),
    np.full(N_LN - CDP_LN,   TENS_LES),
])

# AENS: same structure as DENS
AENS_LIST: np.ndarray = np.concatenate([
    np.full(BODY_LENGTH_LN, AENS_BODY),
    np.full(LES_LENGTH_LN,  AENS_LES),
])

# ESM: linearly increases from ESM_LOW to ESM_HIGH over the first CDP_LN
#      positions, then decreases back to ~ESM_LOW (triangular profile).
_i1 = np.arange(1, CDP_LN + 1, dtype=float)
_i2 = np.arange(1, N_LN - CDP_LN + 1, dtype=float)
ESM_LIST: np.ndarray = np.concatenate([
    ESM_LOW + (ESM_HIGH - ESM_LOW) * (_i1 / CDP_LN),
    ESM_HIGH - (ESM_HIGH - ESM_LOW) * _i2 / (N_LN - CDP_LN),
])

# DSM and BSM: body region then LES region
DSM_LIST: np.ndarray = np.concatenate([
    np.full(BODY_LENGTH_LN, DSM_BODY),
    np.full(LES_LENGTH_LN,  DSM_LES),
])
BSM_LIST: np.ndarray = np.concatenate([
    np.full(BODY_LENGTH_LN, BSM_BODY),
    np.full(LES_LENGTH_LN,  BSM_LES),
])

# LES mask: 1 for body, 0 for LES
LES_MASK: np.ndarray = np.concatenate([
    np.ones(BODY_LENGTH_LN),
    np.zeros(LES_LENGTH_LN),
])

PARAMETER_SET_NORMAL = (DENS_LIST, TENS_LIST, AENS_LIST, ESM_LIST, DSM_LIST, BSM_LIST)


# ─────────────────────────────────────────────────────────────────────────────
# Stimulus construction
# ─────────────────────────────────────────────────────────────────────────────

def make_stimulus(
    s_cns_width: float,
    s_cns_location: float,
    s_cns_duration: float,
    s_cns: float,
    i_cns_width: float,
    i_cns_location: float,
    i_cns_duration: float,
    i_cns: float,
    simulation_ln: int,
    n: int,
    dx: float = DX,
    dt: float = DT,
) -> np.ndarray:
    """
    Build a CNS stimulus array of shape (simulation_ln, n).

    Phase 1 – inhibitory  (descending): rows 0 .. iCNSDurLN-1 (Python),
                                        entire oesophagus.
    Phase 2 – excitatory  (swallow):    rows iCNSDurLN-1 .. iCNSDurLN+sCNSDurLN-1,
                                        narrow proximal zone.

    Note: the two phases overlap at one row (the boundary), with the
    excitatory value overwriting the inhibitory one, matching the
    original Mathematica implementation.
    """
    s_width_ln = round(s_cns_width / dx)
    s_loc_ln   = round(s_cns_location / dx)       # 0-indexed
    s_dur_ln   = round(s_cns_duration / dt)

    i_width_ln = round(i_cns_width / dx) - 1
    i_loc_ln   = round(i_cns_location / dx)       # 0-indexed
    i_dur_ln   = round(i_cns_duration / dt)

    stim = np.zeros((simulation_ln, n))

    # Inhibitory phase
    stim[:i_dur_ln,
         i_loc_ln : i_loc_ln + i_width_ln + 1] = i_cns

    # Excitatory swallow (overwrites boundary row)
    stim[i_dur_ln - 1 : i_dur_ln - 1 + s_dur_ln + 1,
         s_loc_ln : s_loc_ln + s_width_ln + 1] = s_cns

    return stim


def make_pep_positive(
    simulation_ln: int,
    n: int,
    body_length_ln: int,
    s_cns_dur_ln: int,
    pep_hg: float = PEP_HG,
) -> np.ndarray:
    """
    Build a PEP (Peristaltic Esophageal Pressure) array of shape
    (simulation_ln, n).

    Applied only to the body region (columns 0 .. body_length_ln - 2).
    Time profile: Gaussian bell centred at time step s_cns_dur_ln.
    """
    t_idx = np.arange(1, simulation_ln + 1, dtype=float)   # 1-indexed
    lo = round(s_cns_dur_ln / 4)
    hi = 7.0 / 4.0 * s_cns_dur_ln
    pep_time = np.where(
        (t_idx > lo) & (t_idx < hi),
        pep_hg * np.exp(-0.001 * (t_idx - s_cns_dur_ln) ** 2),
        0.0,
    )
    pep = np.zeros((simulation_ln, n))
    # Mathematica: For[y=1, y < bodyLengthLN, y++, ...]  → columns 0..bodyLN-2
    pep[:, : body_length_ln - 1] = pep_time[:, np.newaxis]
    return pep


# Compute sCNS duration in lattice steps (needed for PEP)
_S_CNS_DUR_LN = round(S_CNS_DURATION / DT)     # 50

SCNS_NORMAL: np.ndarray = make_stimulus(
    S_CNS_WIDTH, S_CNS_LOCATION, S_CNS_DURATION, S_CNS,
    I_CNS_WIDTH, I_CNS_LOCATION, I_CNS_DURATION, I_CNS,
    SIMULATION_LN, N_LN,
)

SCNS_NORMAL_LONG: np.ndarray = make_stimulus(
    S_CNS_WIDTH, S_CNS_LOCATION, S_CNS_DURATION, S_CNS,
    I_CNS_WIDTH, I_CNS_LOCATION, I_CNS_DURATION, I_CNS,
    SIMULATION_LN_LONG, N_LN,
)

_I_CNS_DUR_LN = round(I_CNS_DURATION / DT)     # 200

# Repeated swallow: 5 swallows each separated by iCNSDurLN time steps
SCNS_REPEATED_SWALLOW: np.ndarray = sum(
    np.roll(SCNS_NORMAL, k * _I_CNS_DUR_LN, axis=0)
    for k in range(5)
)
SCNS_REPEATED_SWALLOW_LONG: np.ndarray = sum(
    np.roll(SCNS_NORMAL_LONG, k * _I_CNS_DUR_LN, axis=0)
    for k in range(5)
)

PEP_NULL: np.ndarray = np.zeros((SIMULATION_LN, N_LN))
PEP_POSITIVE: np.ndarray = make_pep_positive(
    SIMULATION_LN, N_LN, BODY_LENGTH_LN, _S_CNS_DUR_LN
)


# ─────────────────────────────────────────────────────────────────────────────
# Reaction terms
# ─────────────────────────────────────────────────────────────────────────────

def f_u(u: np.ndarray, v: np.ndarray,
        tens: np.ndarray, aens: np.ndarray) -> np.ndarray:
    """ENS excitable dynamics (cubic nullcline)."""
    return 10.0 * (-u * (u - tens * aens) * (u - 2.0 * aens) - 0.5 * v)


def f_v(u: np.ndarray, v: np.ndarray, dens: np.ndarray) -> np.ndarray:
    """ENS recovery variable — piecewise linear."""
    return np.where(u > 0.0, u - dens * v, -dens * v)


def f_w(u: np.ndarray, w: np.ndarray,
        esm: np.ndarray, dsm: np.ndarray, bsm: np.ndarray) -> np.ndarray:
    """Smooth-muscle pressure dynamics."""
    return esm * u - dsm * w + bsm


# ─────────────────────────────────────────────────────────────────────────────
# Interaction terms (discrete spatial convolution)
# ─────────────────────────────────────────────────────────────────────────────

def interaction_uu(u: np.ndarray, kin: np.ndarray, n: int) -> np.ndarray:
    """
    Excitatory ENS→ENS interaction.

    interaction_uu[j] = Σ_i relu(u[i]) · KIN[N−1+j−i]
                      = np.convolve(relu_u, KIN)[N−1 : 2N−1]
    """
    relu_u = np.maximum(u, 0.0)
    return np.convolve(relu_u, kin)[n - 1 : 2 * n - 1]


def interaction_uw(u: np.ndarray, kmn: np.ndarray, n: int) -> np.ndarray:
    """
    Inhibitory motor-neuron (ENS→SM) interaction.

    interaction_uw[j] = Σ_i relu(u[i]) · KMN[N−1+j−i]
                      = np.convolve(relu_u, KMN)[N−1 : 2N−1]
    """
    relu_u = np.maximum(u, 0.0)
    return np.convolve(relu_u, kmn)[n - 1 : 2 * n - 1]


# ─────────────────────────────────────────────────────────────────────────────
# Initial conditions
# ─────────────────────────────────────────────────────────────────────────────

def make_initial_conditions(
    n: int = N_LN,
    body_ln: int = BODY_LENGTH_LN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return zero-initialised (u0, v0, w0) with the LES region set to its
    resting steady state.

    Mathematica: u0[[bodyLengthLN ;; systemLengthLN]] = 0.8
    In Python 0-indexed that is u0[body_ln - 1 : n] = 0.8
    (includes the last body lattice point and all LES lattice points).
    """
    u0 = np.zeros(n)
    v0 = np.zeros(n)
    w0 = np.zeros(n)
    u0[body_ln - 1 : n] = 0.8
    v0[body_ln - 1 : n] = 0.008
    w0[body_ln - 1 : n] = 80.0
    return u0, v0, w0


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation (forward Euler)
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    u0: np.ndarray,
    v0: np.ndarray,
    w0: np.ndarray,
    scns: np.ndarray,
    pep: np.ndarray,
    kin: np.ndarray,
    kmn: np.ndarray,
    dens: np.ndarray,
    tens: np.ndarray,
    aens: np.ndarray,
    esm: np.ndarray,
    dsm: np.ndarray,
    bsm: np.ndarray,
    les_mask: np.ndarray,
    n: int = N_LN,
    dt: float = DT,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate the esophageal peristalsis model by forward Euler.

    Parameters
    ----------
    u0, v0, w0 : np.ndarray, shape (n,)
        Initial conditions.
    scns : np.ndarray, shape (simulation_ln, n)
        CNS stimulus time series.
    pep : np.ndarray, shape (simulation_ln, n)
        PEP (Peristaltic Esophageal Pressure) time series.
    kin, kmn : np.ndarray, shape (2n,)
        Spatial interaction kernels.
    dens, tens, aens, esm, dsm, bsm : np.ndarray, shape (n,)
        Spatially distributed reaction-term parameters.
    les_mask : np.ndarray, shape (n,)
        Mask that restricts the KMN interaction to body region.

    Returns
    -------
    states_u, states_v, states_w : np.ndarray, shape (simulation_ln + 1, n)
        Full time series including the initial state at index 0.
    """
    simulation_ln = scns.shape[0]

    states_u = np.empty((simulation_ln + 1, n))
    states_v = np.empty((simulation_ln + 1, n))
    states_w = np.empty((simulation_ln + 1, n))

    u = u0.copy()
    v = v0.copy()
    w = w0.copy()

    states_u[0] = u
    states_v[0] = v
    states_w[0] = w

    for t in range(simulation_ln):
        iuu = interaction_uu(u, kin, n)
        iuw = interaction_uw(u, kmn, n)

        u_new = u + dt * (f_u(u, v, tens, aens) + iuu + scns[t])
        v_new = v + dt * f_v(u, v, dens)
        w_new = w + dt * (f_w(u, w, esm, dsm, bsm) + les_mask * iuw + pep[t])

        u, v, w = u_new, v_new, w_new

        states_u[t + 1] = u
        states_v[t + 1] = v
        states_w[t + 1] = w

        if verbose and (t + 1) % 200 == 0:
            print(f"  Step {t + 1}/{simulation_ln}")

    return states_u, states_v, states_w


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def hrm_plot(
    states_w: np.ndarray,
    label: str = "",
    simulation_length: float = SIMULATION_LENGTH,
    system_length: float = SYSTEM_LENGTH,
    cdp_ln: int = CDP_LN,
    vmin: float = -20.0,
    vmax: float = 150.0,
    figsize: tuple = (5, 6),
) -> plt.Figure:
    """
    High-Resolution Manometry (HRM) colour map.

    Replicates the Mathematica HRMPlot: clips negative pressures, returns the
    proximal half at full spatial resolution and the distal half at ×½
    (every other row), matching the original visualisation convention.

    Parameters
    ----------
    states_w : np.ndarray, shape (T, n)
        Pressure time series (including t = 0 initial state).
    """
    w_pos = np.maximum(states_w, 0.0)           # clip negatives (HeavisideTheta)
    w_orig = w_pos.T                             # shape (n, T)

    w1 = w_orig[:cdp_ln, :]                     # proximal full resolution
    w2 = w_orig[cdp_ln::2, :]                   # distal every-other row
    w_plot = np.vstack([w1, w2])                 # shape (~n*0.75, T)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        w_plot,
        aspect="auto",
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        cmap="rainbow",
        extent=[0, simulation_length, system_length / 10, 0],
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Length (cm)")
    if label:
        ax.set_title(label)
    fig.colorbar(im, ax=ax, label="mmHg", shrink=0.8)
    fig.tight_layout()
    return fig


def uvw_plot(
    states_u: np.ndarray,
    states_v: np.ndarray,
    states_w: np.ndarray,
    simulation_length: float = SIMULATION_LENGTH,
    system_length: float = SYSTEM_LENGTH,
) -> plt.Figure:
    """
    Spatiotemporal colour maps of all three state variables U, V, W.
    """
    panels = [
        (states_u.T, "U (ENS activity)",         (-0.5, 2.0),   "gray"),
        (states_v.T, "V (ENS recovery)",          (-0.5, 2.0),   "gray"),
        (states_w.T, "W (pressure, mmHg)",        (-20.0, 150.0),"rainbow"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (data, title, (vmin, vmax), cmap) in zip(axes, panels):
        im = ax.imshow(
            data,
            aspect="auto",
            origin="upper",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            extent=[0, simulation_length, system_length / 10, 0],
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Length (cm)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_parameter_distributions(
    dens: np.ndarray = DENS_LIST,
    tens: np.ndarray = TENS_LIST,
    aens: np.ndarray = AENS_LIST,
    esm: np.ndarray  = ESM_LIST,
    dsm: np.ndarray  = DSM_LIST,
    bsm: np.ndarray  = BSM_LIST,
    dx: float = DX,
    system_length: float = SYSTEM_LENGTH,
) -> plt.Figure:
    """
    Spatial profiles of all six reaction parameters.
    Y-axis shows position along the oesophagus (cm, proximal at top).
    """
    x_mm = np.arange(len(dens)) * dx
    x_cm = x_mm / 10.0

    params = [
        (dens, r"$D_{ENS}$"),
        (tens, r"$T_{ENS}$"),
        (aens, r"$A_{ENS}$"),
        (esm,  r"$E_{SM}$"),
        (dsm,  r"$D_{SM}$"),
        (bsm,  r"$B_{SM}$"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (param, name) in zip(axes.flat, params):
        ax.plot(param, x_cm)
        ax.invert_yaxis()
        ax.set_xlabel(name)
        ax.set_ylabel("Position (cm)")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_kernels(
    kin: np.ndarray = KIN_NORMAL,
    kmn: np.ndarray = KMN_NORMAL,
    n: int = N_LN,
    dx: float = DX,
    plot_half_width_mm: float = 25.0,
) -> plt.Figure:
    """Plot the centred K_IN and K_MN kernel profiles."""
    half = round(plot_half_width_mm / dx)
    x_cm = np.arange(-half, half + 1) * dx / 10.0

    kin_centre = kin[n - 1 - half : n + half] / dx
    kmn_centre = kmn[n - 1 - half : n + half] / dx

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    ax1.plot(kin_centre, x_cm, "r")
    ax1.invert_yaxis()
    ax1.set_xlabel(r"$K_{IN}$")
    ax1.set_ylabel("Position (cm)")
    ax1.set_title(r"$K_{IN}$ kernel")

    ax2.plot(kmn_centre, x_cm, "b")
    ax2.invert_yaxis()
    ax2.set_xlabel(r"$K_{MN}$")
    ax2.set_ylabel("Position (cm)")
    ax2.set_title(r"$K_{MN}$ kernel")

    fig.tight_layout()
    return fig


def plot_stimulus(
    scns: np.ndarray,
    simulation_length: float,
    system_length: float = SYSTEM_LENGTH,
    title: str = r"$S_{CNS}$",
) -> plt.Figure:
    """Spatiotemporal map of the CNS stimulus."""
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        scns.T,
        aspect="auto",
        origin="upper",
        cmap="gray_r",
        extent=[0, simulation_length, system_length / 10, 0],
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Length (cm)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — Control simulation (no PEP)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUT = "Results/Normal"
    os.makedirs(OUT, exist_ok=True)

    # ── Initial conditions ────────────────────────────────────────────────────
    u0, v0, w0 = make_initial_conditions()

    # ── Run control (normal) simulation ──────────────────────────────────────
    print("Running normal simulation (Control, PEP = 0)...")
    dens, tens, aens, esm, dsm, bsm = PARAMETER_SET_NORMAL

    states_u, states_v, states_w = run_simulation(
        u0, v0, w0,
        scns=SCNS_NORMAL,
        pep=PEP_NULL,
        kin=KIN_NORMAL,
        kmn=KMN_NORMAL,
        dens=dens, tens=tens, aens=aens,
        esm=esm,  dsm=dsm,  bsm=bsm,
        les_mask=LES_MASK,
    )
    print("Simulation complete.")

    # ── Save numerical results ────────────────────────────────────────────────
    np.save(f"{OUT}/states_u_normal.npy", states_u)
    np.save(f"{OUT}/states_v_normal.npy", states_v)
    np.save(f"{OUT}/states_w_normal.npy", states_w)
    print(f"Saved states to {OUT}/.")

    # ── HRM plot ──────────────────────────────────────────────────────────────
    fig = hrm_plot(states_w, label="Control")
    fig.savefig(f"{OUT}/HRMPlot_control.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/HRMPlot_control.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}/HRMPlot_control.*")

    # ── UVW spatiotemporal plot ───────────────────────────────────────────────
    fig = uvw_plot(states_u, states_v, states_w)
    fig.savefig(f"{OUT}/UVWPlot_control.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/UVWPlot_control.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}/UVWPlot_control.*")

    # ── Parameter distribution plots ─────────────────────────────────────────
    fig = plot_parameter_distributions()
    fig.savefig(f"{OUT}/ParameterDistributions.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/ParameterDistributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Kernel plots ─────────────────────────────────────────────────────────
    fig = plot_kernels()
    fig.savefig(f"{OUT}/KernelPlots.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/KernelPlots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Stimulus plots ────────────────────────────────────────────────────────
    fig = plot_stimulus(SCNS_NORMAL, SIMULATION_LENGTH)
    fig.savefig(f"{OUT}/StimulusPlot_normal.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/StimulusPlot_normal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"All figures saved to {OUT}/.")
