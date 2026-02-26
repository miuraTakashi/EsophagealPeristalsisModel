"""
Esophageal Disease Simulations
================================
Python translation of the disease-condition sections of EsophagusModel.nb.

Diseases / disorders modelled
------------------------------
EGJ outflow disorders
  1.  Normal peristalsis (control)
  2.  Type I  Achalasia   — NoIRP × NoPulse
  3.  Type II Achalasia   — NoIRP × NoPulse + PEP (panesophageal pressurisation)
  4.  Type III Achalasia  — Premature Contraction × NoPulse
  5.  EGJOO               — NoIRP conditions only

Peristalsis disorders
  6.  Distal Esophageal Spasm (DES)       — Premature Contraction
  7.  Absent Contractility                — NoPulse
  8.  Hypercontractile / Jackhammer       — Increased Amplitude
  9.  Ineffective Esophageal Motility (IEM)
        a. Control (random TENS ×0.9–1.1)
        b. High TENS (body TENS ×1.95, random ×0.9–1.1)
        c. Low sIN  (random TENS ×0.9–1.1, KIN ×0.62)

Parameter-modification categories (Mathematica originals)
----------------------------------------------------------
NoIRP    — conditions that abolish IRP (integrated relaxation pressure)
  1: Positive-only CNS stimulus     (SCNS ← max(SCNS, 0))
  2: Low LES  T_ENS                 (LES TENS = 0.02 × TENS_LES)
  3: High LES B_SM                  (LES BSM  = BSM_LES  + 5000)

NoPulse  — conditions that abolish the peristaltic wave
  1: Negative-only CNS stimulus     (SCNS ← min(SCNS, 0))
  2: High body T_ENS                (body TENS = 2 × TENS_BODY)
  3: Low  s_IN                      (KIN ×0.25)
  4: Zero E_SM                      (ESM = 0)

PrematureContraction (PC) — conditions causing premature contraction
  1: Low  body T_ENS                (body TENS = 0.25 × TENS_BODY)
  2: High s_IN                      (KIN ×2)
  3: High r_IN                      (rebuild KIN with r_IN×2)
  4: High d_IN                      (rebuild KIN with d_IN×2)

IncreasedAmplitude (IA)
  1: High body A_ENS                (body AENS ×1.75)
  2: High E_SM                      (ESM ×2)
  3: Low  D_SM                      (DSM ×0.5)
"""

import copy
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import EsophagusModel as m

# ─────────────────────────────────────────────────────────────────────────────
# Parameter-set dictionary helpers
# ─────────────────────────────────────────────────────────────────────────────

def base_params(label: str = "Normal peristalsis") -> dict:
    """Return the default (normal) parameter set as a plain dict."""
    return {
        "DENS":  m.DENS_LIST.copy(),
        "TENS":  m.TENS_LIST.copy(),
        "AENS":  m.AENS_LIST.copy(),
        "ESM":   m.ESM_LIST.copy(),
        "DSM":   m.DSM_LIST.copy(),
        "BSM":   m.BSM_LIST.copy(),
        "KIN":   m.KIN_NORMAL.copy(),
        "KMN":   m.KMN_NORMAL.copy(),
        "SCNS":  m.SCNS_NORMAL.copy(),
        "PEP":   m.PEP_NULL.copy(),
        "label": label,
    }


def _copy(params: dict) -> dict:
    """Deep-copy a parameter dict (arrays are copied, not referenced)."""
    return {k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in params.items()}


# ─────────────────────────────────────────────────────────────────────────────
# NoIRP modifications (abolish LES relaxation / IRP)
# ─────────────────────────────────────────────────────────────────────────────
NO_IRP_LABELS = [
    r"Positive $S_{CNS}$ only",
    r"Low LES $T_{ENS}$",
    r"High LES $B_{SM}$",
]


def set_ps_no_irp(params: dict, variant: int) -> dict:
    """
    Apply one of the three NoIRP modifications.

    Parameters
    ----------
    variant : int
        1 – remove inhibitory CNS signal (keep positive SCNS only)
        2 – LES TENS reduced to 2 % of normal
        3 – LES BSM raised by +5000
    """
    ps = _copy(params)
    if variant == 1:
        ps["SCNS"] = np.maximum(ps["SCNS"], 0.0)
        ps["label"] = r"Positive $S_{CNS}$ only"
    elif variant == 2:
        tens = ps["TENS"].copy()
        tens[m.BODY_LENGTH_LN :] = 0.02 * m.TENS_LES
        ps["TENS"] = tens
        ps["label"] = r"Low LES $T_{ENS}$"
    elif variant == 3:
        bsm = ps["BSM"].copy()
        bsm[m.BODY_LENGTH_LN :] = m.BSM_LES + 5000.0
        ps["BSM"] = bsm
        ps["label"] = r"High LES $B_{SM}$"
    else:
        raise ValueError(f"NoIRP variant must be 1–3, got {variant}")
    return ps


# ─────────────────────────────────────────────────────────────────────────────
# NoPulse modifications (abolish peristaltic wave)
# ─────────────────────────────────────────────────────────────────────────────
NO_PULSE_LABELS = [
    r"Negative $S_{CNS}$ only",
    r"High body $T_{ENS}$",
    r"Low $s_{IN}$",
    r"Zero $E_{SM}$",
]


def set_ps_no_pulse(params: dict, variant: int) -> dict:
    """
    Apply one of the four NoPulse modifications.

    Parameters
    ----------
    variant : int
        1 – keep only negative SCNS (descending inhibition only)
        2 – body TENS doubled
        3 – KIN scaled by 0.25
        4 – ESM zeroed
    """
    ps = _copy(params)
    if variant == 1:
        ps["SCNS"] = np.minimum(ps["SCNS"], 0.0)
        ps["label"] = r"Negative $S_{CNS}$ only"
    elif variant == 2:
        tens = ps["TENS"].copy()
        tens[: m.BODY_LENGTH_LN] = 2.0 * m.TENS_BODY
        ps["TENS"] = tens
        ps["label"] = r"High body $T_{ENS}$"
    elif variant == 3:
        ps["KIN"] = 0.25 * ps["KIN"]
        ps["label"] = r"Low $s_{IN}$"
    elif variant == 4:
        ps["ESM"] = np.zeros_like(ps["ESM"])
        ps["label"] = r"Zero $E_{SM}$"
    else:
        raise ValueError(f"NoPulse variant must be 1–4, got {variant}")
    return ps


# ─────────────────────────────────────────────────────────────────────────────
# PrematureContraction modifications
# ─────────────────────────────────────────────────────────────────────────────
PC_LABELS = [
    r"Low body $T_{ENS}$",
    r"High $s_{IN}$",
    r"High $r_{IN}$",
    r"High $d_{IN}$",
]


def set_ps_pc(params: dict, variant: int) -> dict:
    """
    Apply one of the four PrematureContraction modifications.

    Parameters
    ----------
    variant : int
        1 – body TENS = 0.25 × normal
        2 – KIN doubled
        3 – rebuild KIN with r_IN × 2
        4 – rebuild KIN with d_IN × 2
    """
    ps = _copy(params)
    if variant == 1:
        tens = ps["TENS"].copy()
        tens[: m.BODY_LENGTH_LN] = 0.25 * m.TENS_BODY
        ps["TENS"] = tens
        ps["label"] = r"Low body $T_{ENS}$"
    elif variant == 2:
        ps["KIN"] = 2.0 * ps["KIN"]
        ps["label"] = r"High $s_{IN}$"
    elif variant == 3:
        ps["KIN"] = m.build_kin(2.0 * m.R_IN, m.S_IN, m.D_IN, m.N_LN, m.DX)
        ps["label"] = r"High $r_{IN}$"
    elif variant == 4:
        ps["KIN"] = m.build_kin(m.R_IN, m.S_IN, 2.0 * m.D_IN, m.N_LN, m.DX)
        ps["label"] = r"High $d_{IN}$"
    else:
        raise ValueError(f"PC variant must be 1–4, got {variant}")
    return ps


# ─────────────────────────────────────────────────────────────────────────────
# IncreasedAmplitude modifications
# ─────────────────────────────────────────────────────────────────────────────
IA_LABELS = [
    r"High body $A_{ENS}$",
    r"High $E_{SM}$",
    r"Low $D_{SM}$",
]


def set_ps_ia(params: dict, variant: int) -> dict:
    """
    Apply one of the three IncreasedAmplitude modifications.

    Parameters
    ----------
    variant : int
        1 – body AENS × 1.75
        2 – ESM  × 2
        3 – DSM  × 0.5
    """
    ps = _copy(params)
    if variant == 1:
        aens = ps["AENS"].copy()
        aens[: m.BODY_LENGTH_LN] = 1.75 * m.AENS_BODY
        ps["AENS"] = aens
        ps["label"] = r"High body $A_{ENS}$"
    elif variant == 2:
        ps["ESM"] = 2.0 * ps["ESM"]
        ps["label"] = r"High $E_{SM}$"
    elif variant == 3:
        ps["DSM"] = 0.5 * ps["DSM"]
        ps["label"] = r"Low $D_{SM}$"
    else:
        raise ValueError(f"IA variant must be 1–3, got {variant}")
    return ps


# ─────────────────────────────────────────────────────────────────────────────
# Run simulation from parameter set
# ─────────────────────────────────────────────────────────────────────────────

def simulate(ps: dict, verbose: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the forward-Euler simulation defined by parameter set *ps*."""
    u0, v0, w0 = m.make_initial_conditions()
    return m.run_simulation(
        u0, v0, w0,
        scns=ps["SCNS"],
        pep=ps["PEP"],
        kin=ps["KIN"],
        kmn=ps["KMN"],
        dens=ps["DENS"],
        tens=ps["TENS"],
        aens=ps["AENS"],
        esm=ps["ESM"],
        dsm=ps["DSM"],
        bsm=ps["BSM"],
        les_mask=m.LES_MASK,
        verbose=verbose,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Disease condition constructors
# ─────────────────────────────────────────────────────────────────────────────

def conditions_control() -> list[dict]:
    return [base_params("Normal peristalsis")]


def conditions_type1_achalasia() -> list[dict]:
    """Type I: NoIRP × NoPulse (variant 1 = positive S_CNS only)."""
    return [
        set_ps_no_irp(set_ps_no_pulse(base_params(), 1), ni)
        for ni in range(1, 4)
    ]


def conditions_type2_achalasia() -> list[dict]:
    """Type II: same as Type I but with PEP (panesophageal pressurisation)."""
    conds = conditions_type1_achalasia()
    for ps in conds:
        ps["PEP"] = m.PEP_POSITIVE.copy()
        ps["label"] = ps["label"] + " + PEP"
    return conds


def conditions_type3_achalasia() -> list[dict]:
    """Type III: PrematureContraction variants combined with NoPulse variant 1."""
    return [
        set_ps_pc(set_ps_no_pulse(base_params(), 1), pc)
        for pc in range(1, 5)
    ]


def conditions_egjoo() -> list[dict]:
    """EGJOO: NoIRP conditions only (no NoPulse)."""
    return [set_ps_no_irp(base_params(), ni) for ni in range(1, 4)]


def conditions_des() -> list[dict]:
    """Distal Esophageal Spasm: PrematureContraction variants."""
    return [set_ps_pc(base_params(), pc) for pc in range(1, 5)]


def conditions_absent_contractility() -> list[dict]:
    """Absent Contractility: NoPulse variants."""
    return [set_ps_no_pulse(base_params(), np_) for np_ in range(1, 5)]


def conditions_jackhammer() -> list[dict]:
    """Hypercontractile (Jackhammer) Esophagus: IncreasedAmplitude variants."""
    return [set_ps_ia(base_params(), ia) for ia in range(1, 4)]


def conditions_iem(n_runs: int = 5, seed_offset: int = 0) -> list[dict]:
    """
    Ineffective Esophageal Motility — three stochastic variants.

    Each variant is repeated *n_runs* times with different random TENS
    multipliers drawn from Uniform(0.9, 1.1).

    Parameters
    ----------
    n_runs : int
        Number of random realisations per IEM variant.
    seed_offset : int
        NumPy random seed used for the Control variant; High-TENS uses
        seed_offset+1, Low-sIN uses seed_offset+2.
    """
    conds = []

    # ── IEM Control: random TENS ×(0.9–1.1)
    rng = np.random.default_rng(seed_offset)
    for i in range(n_runs):
        ps = base_params()
        r = rng.uniform(0.9, 1.1)
        ps["TENS"] = r * ps["TENS"]
        ps["label"] = rf"IEM Control, $T_{{ENS}}\times{r:.2f}$"
        conds.append(ps)

    # ── IEM High TENS: body TENS ×1.95, random ×(0.9–1.1)
    rng = np.random.default_rng(seed_offset + 1)
    for i in range(n_runs):
        ps = base_params()
        r = rng.uniform(0.9, 1.1)
        tens = ps["TENS"].copy()
        tens[: m.BODY_LENGTH_LN] = tens[0] * 1.95
        ps["TENS"] = r * tens
        ps["label"] = rf"IEM High $T_{{ENS}}$, $\times{r*1.95:.2f}$"
        conds.append(ps)

    # ── IEM Low s_IN: random TENS + KIN ×0.62
    rng = np.random.default_rng(seed_offset + 2)
    for i in range(n_runs):
        ps = base_params()
        r = rng.uniform(0.9, 1.1)
        ps["TENS"] = r * ps["TENS"]
        ps["KIN"]  = 0.62 * ps["KIN"]
        ps["label"] = rf"IEM Low $s_{{IN}}$, $T_{{ENS}}\times{r:.2f}$"
        conds.append(ps)

    return conds


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hrm_array(states_w: np.ndarray) -> np.ndarray:
    """Prepare the HRM display matrix (clip, downsample distal half)."""
    w_pos  = np.maximum(states_w, 0.0)
    w_orig = w_pos.T
    w1     = w_orig[: m.CDP_LN, :]
    w2     = w_orig[m.CDP_LN :: 2, :]
    return np.vstack([w1, w2])


def make_panel_figure(
    condition_list: list[dict],
    disease_name: str,
    n_cols: int = 4,
    panel_w: float = 2.8,
    panel_h: float = 3.5,
    vmin: float = -20.0,
    vmax: float = 150.0,
) -> plt.Figure:
    """
    Simulate all conditions in *condition_list* and produce a tiled HRM panel
    figure.  Prints progress.

    Returns
    -------
    matplotlib Figure
    """
    n = len(condition_list)
    n_rows = (n + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(panel_w * n_cols, panel_h * n_rows))
    fig.suptitle(disease_name, fontsize=13, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.6, wspace=0.3)

    for idx, ps in enumerate(condition_list):
        print(f"  [{idx + 1}/{n}] {ps['label']} …", end=" ", flush=True)
        _, _, states_w = simulate(ps, verbose=False)
        w_plot = _hrm_array(states_w)
        print("done")

        row, col = divmod(idx, n_cols)
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(
            w_plot,
            aspect="auto",
            origin="upper",
            vmin=vmin,
            vmax=vmax,
            cmap="rainbow",
            extent=[0, m.SIMULATION_LENGTH, m.SYSTEM_LENGTH / 10, 0],
        )
        ax.set_title(ps["label"], fontsize=7, pad=3)
        ax.set_xlabel("Time (s)", fontsize=6)
        ax.set_ylabel("Length (cm)", fontsize=6)
        ax.tick_params(labelsize=6)

    # Shared colour bar
    fig.colorbar(im, ax=fig.axes, label="mmHg", shrink=0.6, pad=0.01)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Disease registry
# ─────────────────────────────────────────────────────────────────────────────

DISEASE_REGISTRY = [
    ("Control",                    conditions_control,               2),
    ("Type_I_Achalasia",           conditions_type1_achalasia,       3),
    ("Type_II_Achalasia",          conditions_type2_achalasia,       3),
    ("Type_III_Achalasia",         conditions_type3_achalasia,       4),
    ("EGJOO",                      conditions_egjoo,                 3),
    ("Distal_Esophageal_Spasm",    conditions_des,                   4),
    ("Absent_Contractility",       conditions_absent_contractility,  4),
    ("Jackhammer_Esophagus",       conditions_jackhammer,            3),
    ("IEM",                        lambda: conditions_iem(n_runs=3), 3),
]

DISEASE_PRETTY_NAMES = {
    "Control":                   "Normal peristalsis (control)",
    "Type_I_Achalasia":          "Type I Achalasia",
    "Type_II_Achalasia":         "Type II Achalasia (+ PEP)",
    "Type_III_Achalasia":        "Type III Achalasia",
    "EGJOO":                     "EGJOO (EGJ Outflow Obstruction)",
    "Distal_Esophageal_Spasm":   "Distal Esophageal Spasm",
    "Absent_Contractility":      "Absent Contractility",
    "Jackhammer_Esophagus":      "Hypercontractile (Jackhammer) Esophagus",
    "IEM":                       "Ineffective Esophageal Motility (IEM)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUT = "Results/Diseases"
    os.makedirs(OUT, exist_ok=True)

    for key, cond_fn, n_cols in DISEASE_REGISTRY:
        pretty = DISEASE_PRETTY_NAMES[key]
        print(f"\n{'='*60}")
        print(f"Disease: {pretty}")
        print(f"{'='*60}")

        conds = cond_fn()
        fig = make_panel_figure(conds, pretty, n_cols=n_cols)
        out_pdf = f"{OUT}/{key}.pdf"
        out_png = f"{OUT}/{key}.png"
        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → Saved {out_pdf}  {out_png}")

    print("\nAll disease simulations complete.")
