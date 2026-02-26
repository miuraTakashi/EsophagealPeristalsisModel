# EsophagealPeristalsisModel

Mathematical model of human esophageal motility

## Overview

This repository contains a Python implementation of a spatiotemporal mathematical model of esophageal peristalsis, ported from the original Mathematica notebook (`EsophagusModel.nb`).

The model simulates the coordinated contraction of esophageal smooth muscle driven by the Enteric Nervous System (ENS), using three coupled variables at each spatial lattice point:

| Variable | Description |
|---|---|
| `u` | ENS neural activity |
| `v` | ENS recovery variable |
| `w` | Smooth-muscle pressure (mmHg) |

The governing equations are integrated by the forward Euler method:

```
du/dt = f_u(u, v) + interaction_uu(u) + S_CNS(t)
dv/dt = f_v(u, v)
dw/dt = f_w(u, w) + les_mask · interaction_uw(u) + PEP(t)
```

Spatial coupling between lattice points is implemented via discrete convolution with the Law-of-Intestine kernels `K_IN` (ENS→ENS excitatory) and `K_MN` (ENS→SM inhibitory).

## Files

| File | Description |
|---|---|
| `EsophagusModel.nb` | Original Mathematica notebook |
| `EsophagusModel.py` | Python port — model parameters, simulation engine, and visualisation |
| `EsophagusDisease.py` | Disease condition simulations (see below) |
| `requirements.txt` | Python package dependencies |

## Requirements

```
numpy >= 1.24
scipy >= 1.10
matplotlib >= 3.7
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage

**Normal peristalsis and MRS:**

```bash
python EsophagusModel.py
```

**Disease simulations:**

```bash
python EsophagusDisease.py
```

## Output

Results are saved under `Results/` in two subdirectories:

### `Results/Normal/`

Outputs from `EsophagusModel.py` (normal esophageal dynamics):

| File | Description |
|---|---|
| `HRMPlot_control.*` | High-Resolution Manometry (HRM) plot — single swallow |
| `UVWPlot_control.*` | Spatiotemporal maps of u, v, w — single swallow |
| `StimulusPlot_normal.*` | CNS stimulus pattern — single swallow |
| `HRMPlot_mrs.*` | HRM plot — Multiple Rapid Swallows (MRS, 5 swallows / 20 s) |
| `UVWPlot_mrs.*` | Spatiotemporal maps of u, v, w — MRS |
| `StimulusPlot_mrs.*` | CNS stimulus pattern — MRS |
| `ParameterDistributions.*` | Spatial profiles of all six reaction parameters |
| `KernelPlots.*` | K_IN and K_MN kernel profiles |
| `states_*_normal.npy` | Raw simulation arrays — single swallow |
| `states_*_mrs.npy` | Raw simulation arrays — MRS |

### `Results/Diseases/`

Outputs from `EsophagusDisease.py`.  Each file is a panel figure showing HRM plots for all parameter-modification variants of that condition:

| File | Disease / Disorder |
|---|---|
| `Control.*` | Normal peristalsis (reference) |
| `Type_I_Achalasia.*` | Type I Achalasia |
| `Type_II_Achalasia.*` | Type II Achalasia (with panesophageal pressurisation, PEP) |
| `Type_III_Achalasia.*` | Type III Achalasia |
| `EGJOO.*` | EGJ Outflow Obstruction (EGJOO) |
| `Distal_Esophageal_Spasm.*` | Distal Esophageal Spasm (DES) |
| `Absent_Contractility.*` | Absent Contractility |
| `Jackhammer_Esophagus.*` | Hypercontractile (Jackhammer) Esophagus |
| `IEM.*` | Ineffective Esophageal Motility (IEM) — stochastic variants |

## Disease Modelling

Each disease is simulated by modifying one or more parameters of the default (normal) parameter set.  The modification categories are:

**NoIRP** — abolishes integrated relaxation pressure (LES relaxation failure):

| Variant | Modification |
|---|---|
| 1 | Remove inhibitory CNS signal: `S_CNS ← max(S_CNS, 0)` |
| 2 | Reduce LES `T_ENS` to 2 % of normal |
| 3 | Increase LES `B_SM` by +5000 |

**NoPulse** — abolishes the peristaltic wave:

| Variant | Modification |
|---|---|
| 1 | Remove excitatory CNS signal: `S_CNS ← min(S_CNS, 0)` |
| 2 | Double body `T_ENS` |
| 3 | Scale `K_IN` by ×0.25 |
| 4 | Zero out `E_SM` |

**PrematureContraction** — causes premature distal contraction:

| Variant | Modification |
|---|---|
| 1 | Body `T_ENS` = 0.25 × normal |
| 2 | Scale `K_IN` by ×2 |
| 3 | Rebuild `K_IN` with `r_IN` doubled |
| 4 | Rebuild `K_IN` with `d_IN` doubled |

**IncreasedAmplitude** — raises contraction amplitude:

| Variant | Modification |
|---|---|
| 1 | Body `A_ENS` × 1.75 |
| 2 | `E_SM` × 2 |
| 3 | `D_SM` × 0.5 |

The combinations applied to each disease are:

| Disease | Conditions |
|---|---|
| Type I Achalasia | NoIRP (×3) with NoPulse-1 |
| Type II Achalasia | Same as Type I + PEP |
| Type III Achalasia | PrematureContraction (×4) with NoPulse-1 |
| EGJOO | NoIRP (×3) |
| Distal Esophageal Spasm | PrematureContraction (×4) |
| Absent Contractility | NoPulse (×4) |
| Jackhammer Esophagus | IncreasedAmplitude (×3) |
| IEM | Stochastic `T_ENS` — Control / High-TENS / Low-`s_IN` (3 runs each) |
