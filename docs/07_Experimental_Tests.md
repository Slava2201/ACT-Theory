# Experimental Tests of ACT

*From Quantum Gravity to Collider Physics*

---

## Table of Contents

1. [Introduction: The Testability Problem](#1-introduction-the-testability-problem)
2. [Particle Physics Tests](#2-particle-physics-tests)
3. [Gravitational Wave Tests](#3-gravitational-wave-tests)
4. [Cosmological Tests](#4-cosmological-tests)
5. [Laboratory Tests](#5-laboratory-tests)
6. [Astrophysical Tests](#6-astrophysical-tests)
7. [Data Analysis Pipeline](#7-data-analysis-pipeline)
8. [Future Experiments](#8-future-experiments)
9. [Falsifiability Criteria](#9-falsifiability-criteria)

---

## 1. Introduction: The Testability Problem

### 1.1 The Challenge of Testing Quantum Gravity

Quantum gravity theories often face criticism for lack of experimental testability. **ACT addresses this** with specific, quantitative predictions across multiple energy scales.

**Key predictions at different scales**:

| Energy Scale | Prediction | Test |
|--------------|------------|------|
| **Planck scale** ($10^{19}$ GeV) | Quantum foam, non-commutativity | Indirect through RG running |
| **GUT scale** ($10^{16}$ GeV) | Proton decay, neutrino masses | Next-generation detectors |
| **LHC scale** (14 TeV) | New resonances, contact interactions | ATLAS, CMS, LHCb |
| **Astrophysical** (MeV-GeV) | Dark matter signals, GW echoes | Fermi, LIGO, IceCube |
| **Laboratory** (eV-keV) | Lorentz violation, fifth force | Atom interferometers, Eötvös |

### 1.2 The ACT Prediction Database

All predictions are stored in machine-readable format:

```python
{
    "prediction_id": "ACT-2024-001",
    "observable": "Z' → μ⁺μ⁻ cross section",
    "value": "σ = 1.2 ± 0.3 fb at √s = 14 TeV",
    "experiment": "ATLAS/CMS",
    "significance": "5σ with 300/fb",
    "test_status": "ongoing"
}
```

2. Particle Physics Tests
2.1 LHC Predictions
2.1.1 New Resonances
$Z'$ boson:

Mass: $M_{Z'} = 3.5 \pm 0.2$ TeV

Width: $\Gamma_{Z'} = 0.35 \pm 0.05$ TeV

Production: $σ(pp → Z') = 1.0 \pm 0.3$ fb

Decays: $BR(Z' → ℓ⁺ℓ⁻) = 8%$, $BR(Z' → q\bar{q}) = 70%$

Signature: High-mass dilepton resonance with forward-backward asymmetry.

Gravitational Kaluza-Klein modes:

Mass: $M_{G^*} = 5.0 \pm 0.5$ TeV

Signature: $G^* → γγ$ with $σ × BR = 0.3$ fb

2.1.2 Contact Interactions
Four-fermion operators suppressed by scale $\Lambda$:

$$
\mathcal{L}_{CI} = \frac{1}{\Lambda^2} (\bar{q}\gamma^{\mu}q)(\bar{\ell}\gamma_{\mu}\ell)
$$

ACT prediction: $\Lambda = 25 \pm 3$ TeV

2.1.3 Dark Matter Production
Mono-X signatures:

Mono-jet: $σ(pp → χχ + j) = 0.8$ fb

Mono-photon: $σ(pp → χχ + γ) = 0.1$ fb

Missing $E_T$ distribution has characteristic shape

2.2 Flavor Physics
2.2.1 Rare Decays
$B_s → μ⁺μ⁻$:

$$
\mathrm{BR}(B_s \to \mu^+ \mu^-)_{\mathrm{ACT}} = (3.45 \pm 0.15) \times 10^{-9}
$$

vs. experimental: $(3.09 \pm 0.20) \times 10^{-9}$

$B → K^{(*)}μ⁺μ⁻$ anomalies:
ACT predicts specific $q^2$ dependence in $R_K$ and $R_{K^*}$:

$$
R_K^{\mathrm{ACT}} = 1.000 \pm 0.005 \quad \text{for} \quad 1.1 < q^2 < 6.0 \,\text{GeV}^2
$$

2.2.2 Lepton Flavor Violation
$μ → eγ$:

$$
\mathrm{BR}(\mu \to e \gamma)_{\mathrm{ACT}} = (4.2 \pm 0.5) \times 10^{-14}
$$

Current limit: $< 4.2 \times 10^{-13}$ (MEG)
Future sensitivity (MEG II): $6 \times 10^{-14}$

2.3 Precision Electroweak
2.3.1 W Boson Mass
ACT correction to $M_W$:

$$
\Delta M_W = 4 \ \mathrm{MeV} \quad \Rightarrow \quad M_W^{\mathrm{ACT}} = 80.381 \ \mathrm{GeV}
$$

Current experimental: $80.377 \pm 0.012$ GeV

2.3.2 Higgs Couplings
Modifications to Higgs couplings:

| Coupling | $κ_i^{\text{ACT}}$ | SM | Difference |
| :--- | :--- | :--- | :--- |
| $κ_g$ | $0.98 \pm 0.02$ | 1.00 | $-0.02$ |
| $κ_γ$ | $1.02 \pm 0.02$ | 1.00 | $+0.02$ |
| $κ_μ$ | $1.20 \pm 0.10$ | 1.00 | $+0.20$ |
| $κ_τ$ | $1.05 \pm 0.05$ | 1.00 | $+0.05$ |

3. Gravitational Wave Tests
   
3.1 Quantum Gravity Effects in GWs

3.1.1 Modified Dispersion

Gravitational waves propagate with energy-dependent speed:

$$
v_g(E) = c \left[ 1 - \xi \left( \frac{E}{E_{\text{LV}}} \right)^n \right]
$$

ACT predictions:

$ξ = 1.0 \pm 0.2$

$n = 2$ (quadratic)

$E_{\text{LV}} = (1.0 \pm 0.1) \times 10^{19}$ GeV

Test: Compare arrival times of GW170817 and GRB 170817A:

$$
\Delta t = \frac{D}{c} \, \xi \left( \frac{E}{E_{\text{LV}}} \right)^2 \approx 0.1 \, \text{s} \quad \text{at} \quad E = 100 \, \text{MeV}
$$

3.1.2 Birefringence

Different polarizations propagate differently:

$$
v_{+} - v_{\times} = c \cdot \zeta \left( \frac{f}{f_{\text{LV}}} \right)^2
$$

ACT prediction: $ζ = (2.0 \pm 0.5) \times 10^{-17}$ at $f = 100$ Hz


3.2 Black Hole Echoes

3.2.1 Echo Signal

Post-merger echoes with characteristic pattern:

Time delay:

$$\Delta t_{\text{echo}} = \frac{GM}{c^3} \ln\left(\frac{M^2}{m_P^2}\right) \approx 0.3 \ \text{ms}$ for $M = 30 M_{\odot}$$

Amplitude damping:

$$A_n = A_0 e^{-n\gamma}$ with $\gamma = 0.8 \pm 0.1$$

Frequency content: Higher frequencies enhanced in echoes.

3.2.2 Template for LIGO/Virgo

The echo waveform:

$$h_{\text{echo}}(t) = \sum_{n=1}^{N} \alpha_n h_{\text{ringdown}}(t - n\Delta t) e^{-n\gamma} \cos(\varphi_n)$$

Parameters predicted by ACT for GW150914:

$\Delta t = 0.31$ ms

$\alpha = 0.25$

$N = 5-10$ detectable echoes

3.3 Stochastic Background

3.3.1 Primordial Gravitational Waves

From inflation:

$$\Omega_{\text{GW}}(f) = \frac{r A_s}{24} \left( \frac{f}{f_*} \right)^{n_t}$$

ACT predictions:

$r = 0.004 \pm 0.002$

$n_t = -0.008 \pm 0.004$

$\Omega_{\text{GW}}(100 \ \text{Hz}) = (1.2 \pm 0.3) \times 10^{-9}$

3.3.2 Astrophysical Background
From compact binary mergers:

$$\Omega_{\text{GW}}^{\text{astro}}(f) = \frac{8\pi^{5/3}}{9} f^{2/3} \int dz \frac{R(z)}{(1+z)^{4/3} H(z)}$$

ACT modifies merger rate $R(z)$ at high $z$.

4. Cosmological Tests
4.1 CMB Anomalies
ACT predicts specific patterns in CMB.

4.1.1 Low- $\ell$ Power Suppression
Angular power spectrum at $\ell = 2-5$:


| $\ell$ | $C_\ell^{\text{ACT}}$ (μK²) | $C_\ell^{\Lambda\text{CDM}}$ (μK²) | Ratio |
| :--- | :--- | :--- | :--- |
| 2 | $950 \pm 150$ | $1150 \pm 200$ | $0.83 \pm 0.15$ |
| 3	| $550 \pm 100$ | $650 \pm 120$ | $0.85 \pm 0.15$ |
| 4	| $350 \pm 70$ | $400 \pm 80$ |	$0.88 \pm 0.15$ |


4.1.2 Hemispherical Asymmetry

Dipole modulation:

$$\frac{\Delta T}{T}(\hat{n}) = \left[ 1 + A (\hat{n} \cdot \hat{p}) \right] \frac{\Delta T}{T}_{\text{iso}}$$

ACT predictions:

$A = 0.07 \pm 0.02$

$\hat{p} = (l,b) = (227^\circ \pm 10^\circ, -27^\circ \pm 10^\circ)$


4.1.3 Cold Spot

Non-Gaussian cold spot:

Location: $(l,b) = (209^\circ, -57^\circ)$

Size: $5^\circ$ radius

Significance: $3.5\sigma$

4.2 Large-Scale Structure

4.2.1 Matter Power Spectrum

Suppression at small scales ($k > 10 \ h/$Mpc):

$$P_{\text{ACT}}(k) = P_{\Lambda\text{CDM}}(k) \times \left[ 1 - 0.03 \left( \frac{k}{10 \ h/\text{Mpc}} \right)^2 \right]$$

4.2.2 Halo Mass Function

Fewer low-mass halos:

$$\frac{n_{\text{ACT}}(M)}{n_{\Lambda\text{CDM}}(M)} = \exp\left[ -0.1 \left( \frac{10^{10} M_{\odot}}{M} \right)^{0.5} \right]$$

4.3 BAO Measurements

Modified sound horizon:

$$r_d^{\text{ACT}} = r_d^{\Lambda\text{CDM}} \times \left[ 1 - 0.0012 (z/3)^2 \right]$$

At $z = 0.5$: $\Delta r_d / r_d = -0.0002$.
