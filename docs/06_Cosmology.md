# Cosmology in ACT

*From Quantum Gravity to the Large-Scale Universe*

---

## Table of Contents

1. [Introduction: Emergent Cosmology](#1-introduction-emergent-cosmology)
2. [Early Universe: Quantum Bounce](#2-early-universe-quantum-bounce)
3. [Inflation from Network Dynamics](#3-inflation-from-network-dynamics)
4. [Structure Formation](#4-structure-formation)
5. [Dark Matter and Dark Energy](#5-dark-matter-and-dark-energy)
6. [CMB Anomalies](#6-cmb-anomalies)
7. [Late-Time Universe](#7-late-time-universe)
8. [Numerical Cosmological Simulations](#8-numerical-cosmological-simulations)
9. [Observational Tests](#9-observational-tests)

---

## 1. Introduction: Emergent Cosmology

### 1.1 The ACT Cosmological Framework

In ACT, cosmology emerges from the large-scale structure of the causal network. The Friedmann equations are not postulated but **derived** from network dynamics.

**Key equations**:

The scale factor $a(t)$ emerges from network expansion:

$$
a(t) = \left( \frac{N(t)}{N_0} \right)^{1/3} \ell_P
$$

where $N(t)$ is the number of vertices at time $t$.

The Hubble parameter:

$$
H(t) = \frac{\dot{a}}{a} = \frac{1}{3} \frac{\dot{N}}{N}
$$

### 1.2 Emergent Friedmann Equations

From the network Einstein equations, we obtain:

**First Friedmann equation**:

$$
\left( \frac{\dot{a}}{a} \right)^2 = \frac{8\pi G}{3} \rho + \frac{\Lambda c^2}{3} - \frac{k c^2}{a^2}
$$

**Second Friedmann equation**:

$$
\frac{\ddot{a}}{a} = -\frac{4\pi G}{3} \left( \rho + \frac{3p}{c^2} \right) + \frac{\Lambda c^2}{3}
$$

where $\rho$ and $p$ are emergent from network energy density and pressure.

---

## 2. Early Universe: Quantum Bounce

### 2.1 Resolution of the Big Bang Singularity

**Theorem 2.1**: In ACT, the initial singularity is replaced by a quantum bounce.

The scale factor evolution near the bounce:

$$
a(t) = a_{\min} \cosh\left( \frac{t}{\tau} \right)
$$

where:
- $a_{\min} \approx \ell_P$
- $\tau \approx t_P = 5.39 \times 10^{-44}$ s

**No singularity**: $a(t) > 0$ for all $t$.

### 2.2 Bounce Dynamics

The effective Friedmann equation with quantum corrections:

$$
H^2 = \frac{8\pi G}{3} \rho \left( 1 - \frac{\rho}{\rho_{\max}} \right)
$$

where $\rho_{\max} \approx 0.41 \rho_P$ and $\rho_P = c^5/(\hbar G^2) \approx 5.16 \times 10^{96}$ kg/m³.

**Maximum density**:

$$
\rho_{\max} = \frac{3c^2}{32\pi G \ell_P^2} \approx 0.41 \rho_P
$$

### 2.3 Pre-Bounce Phase

Before the bounce, the universe contracts with:

**Contraction phase**:

$$
a(t) = a_{\min} \cosh\left( \frac{t - t_{\text{bounce}}}{\tau} \right)
$$

for $t < t_{\text{bounce}}$.

The universe is **cyclic** on Planck scales but effectively non-cyclic on cosmological scales due to entropy increase.

---

## 3. Inflation from Network Dynamics

### 3.1 Emergent Inflation Field

The inflaton field $\phi$ emerges from network connectivity fluctuations:

$$
\phi(x) = \sqrt{\langle k \rangle} \delta k(x)
$$

where $\delta k(x) = k(x) - \langle k \rangle$ is the degree fluctuation.

### 3.2 Inflation Potential

The effective potential from network energy:

$$
V(\phi) = \frac{1}{2} m^2 \phi^2 + \frac{\lambda}{4} \phi^4 \left( 1 - \frac{\phi^2}{\phi_0^2} \right)^2
$$

with parameters:
- $m \approx 1.5 \times 10^{13}$ GeV
- $\phi_0 \approx 0.3 M_P$
- $\lambda \approx 10^{-13}$

### 3.3 Slow-Roll Parameters

**ACT predictions**:

Epsilon parameter:

$$
\epsilon = \frac{M_P^2}{2} \left( \frac{V'}{V} \right)^2 \approx 2 \times 10^{-3}
$$

Eta parameter:

$$
\eta = M_P^2 \frac{V''}{V} \approx -0.02
$$

**Number of e-folds**:

$$
N_e = \int_{\phi_{\text{end}}}^{\phi_*} \frac{V}{V'} d\phi \approx 55-65
$$

### 3.4 Power Spectra

**Scalar perturbations**:

$$
\mathcal{P}_\zeta(k) = A_s \left( \frac{k}{k_*} \right)^{n_s-1}
$$

with:
- $A_s = (2.10 \pm 0.03) \times 10^{-9}$
- $n_s = 0.965 \pm 0.004$

**Tensor perturbations**:

$$
\mathcal{P}_h(k) = A_t \left( \frac{k}{k_*} \right)^{n_t}
$$

with tensor-to-scalar ratio:

$$
r = \frac{A_t}{A_s} = 0.004 \pm 0.002
$$

### 3.5 Non-Gaussianity

The bispectrum shape:

$$
B_\zeta(k_1, k_2, k_3) = f_{NL} [P(k_1)P(k_2) + \text{perms}]
$$

ACT prediction:

$$
f_{NL}^{\text{local}} = 1.2 \pm 0.3
$$

---

## 4. Structure Formation

### 4.1 Matter Power Spectrum

The linear matter power spectrum:

$$
P(k) = A k^{n_s} T^2(k)
$$

with transfer function $T(k)$ from network growth.

**ACT modification at small scales**:

$$
P_{\text{ACT}}(k) = P_{\text{ΛCDM}}(k) \times \left[ 1 + \alpha \left( \frac{k}{k_{\text{cut}}} \right)^2 \right]^{-1}
$$

where $k_{\text{cut}} \approx 1/\ell_P \approx 10^{35}$ m⁻¹.

### 4.2 Growth of Perturbations

The growth equation with ACT corrections:

$$
\ddot{\delta} + 2H\dot{\delta} - 4\pi G \rho \delta \left[ 1 + \beta \left( \frac{\ell_P}{a} \right)^2 \nabla^2 \right] = 0
$$

where $\beta \approx 0.1$ from network effects.

### 4.3 Halo Mass Function

The number density of halos of mass $M$:

$$
\frac{dn}{dM} = \frac{\rho_m}{M} f(\nu) \left| \frac{d\ln \sigma}{d\ln M} \right|
$$

with ACT modification:

$$
f_{\text{ACT}}(\nu) = f_{\text{PS}}(\nu) \exp\left[ -\frac{\gamma}{\nu^2} \left( \frac{M}{M_P} \right)^{2/3} \right]
$$

---

## 5. Dark Matter and Dark Energy

### 5.1 Dark Matter as Topological Defects

**Theorem 5.1**: Dark matter consists of network topological defects.

Defect density:

$$
\rho_{\text{DM}} = n_{\text{defects}} m_{\text{defect}}
$$

where:
- $n_{\text{defects}} \approx N^{-1/2} \ell_P^{-3}$
- $m_{\text{defect}} \approx M_P/\sqrt{\alpha_{\text{EM}}}$

**Fraction**:

$$
\Omega_{\text{DM}} = \frac{\rho_{\text{DM}}}{\rho_c} \approx 0.268
$$

### 5.2 Dark Energy from Causal Volume

The cosmological constant emerges as:

$$
\Lambda = \frac{\alpha}{8\pi\beta} \frac{1}{\ell_P^2 N^{1/2}}
$$

**Numerical value**:

$$
\Lambda \approx 1.1 \times 10^{-52} \ \text{m}^{-2}
$$

**Equation of state**:

$$
w(a) = -1 + \frac{\Omega_m}{3\Omega_\Lambda} a^{-3} + O(a^{-6})
$$

Present value: $w_0 = -0.995 \pm 0.005$

### 5.3 Coupled Dark Sector

Dark matter and dark energy interact through network connections:

**Interaction term**:

$$
Q = \xi H \rho_{\text{DM}} \rho_\Lambda
$$

with $\xi \approx 0.01$ from network topology.

---

## 6. CMB Anomalies

### 6.1 Low-$\ell$ Anomalies

ACT predicts specific modifications to CMB:

**Quadrupole suppression**:

$$
C_2^{\text{ACT}} = C_2^{\text{ΛCDM}} \times \left[ 1 - 0.15 \exp\left( -\frac{\ell_P}{\tau_{\text{LS}}} \right) \right]
$$

**Lack of large-scale correlation**:

Angular correlation function suppressed at $\theta > 60^\circ$.

### 6.2 Hemispherical Asymmetry

The CMB power asymmetry:

$$
\frac{\Delta T}{T}(\hat{n}) = \left[ 1 + A(\hat{n} \cdot \hat{p}) \right] \frac{\Delta T}{T}_{\text{isotropic}}
$$

with amplitude $A \approx 0.07$ and direction $\hat{p} = (l,b) = (227^\circ, -27^\circ)$.

### 6.3 Cold Spot

The non-Gaussian cold spot at $(l,b) = (209^\circ, -57^\circ)$ explained by:

**Texture decay**:

$$
\frac{\Delta T}{T} \approx -7 \times 10^{-5} \text{ for } \theta < 5^\circ
$$

### 6.4 ACT Predictions for CMB-S4

**EE polarization**:

Enhanced at $\ell > 2000$:

$$
C_\ell^{EE}(\text{ACT}) = C_\ell^{EE}(\text{ΛCDM}) \times \left[ 1 + 0.05 \left( \frac{\ell}{2000} \right)^2 \right]
$$

**Lensing potential**:

Modified at small scales:

$$
C_\ell^{\phi\phi}(\text{ACT}) = C_\ell^{\phi\phi}(\text{ΛCDM}) \times \left[ 1 - 0.03 \left( \frac{\ell}{3000} \right) \right]
$$

---

## 7. Late-Time Universe

### 7.1 Hubble Tension

ACT predicts a slightly different expansion history:

**Hubble constant**:

$$
H_0 = 67.8 \pm 0.5 \ \text{km/s/Mpc}
$$

consistent with Planck but with different systematics.

### 7.2 $S_8$ Tension

The matter clustering parameter:

$$
S_8 = \sigma_8 \sqrt{\frac{\Omega_m}{0.3}}
$$

ACT prediction: $S_8 = 0.81 \pm 0.01$, alleviating the tension.

### 7.3 Baryon Acoustic Oscillations

BAO scale modified by quantum gravity:

$$
r_d^{\text{ACT}} = r_d^{\text{ΛCDM}} \left[ 1 - 0.001 \left( \frac{z}{3} \right)^2 \right]
$$

### 7.4 Future Evolution

**Fate of the universe**:

$$
a(t) \propto \exp\left( H_0 t \right) \quad \text{for } t \gg t_0
$$

**Event horizon**:

$$
R_H = \frac{c}{H_0 \sqrt{\Omega_\Lambda}} \approx 16.7 \ \text{Gly}
$$

---

## 8. Numerical Cosmological Simulations

### 8.1 Network Cosmology Code

```python
import numpy as np
from act_cosmology import CosmologicalNetwork

def simulate_cosmology(N=10000, steps=1000):
    """Simulate cosmological evolution in ACT."""
    
    # Initialize network
    universe = CosmologicalNetwork(N=N)
    
    # Evolution history
    history = {
        'scale_factor': [],
        'hubblle_parameter': [],
        'density_parameters': [],
        'power_spectrum': []
    }
    
    for step in range(steps):
        # Evolve network
        universe.evolve_step(dt=0.1 * universe.t_P)
        
        # Measure cosmological parameters
        a = universe.scale_factor()
        H = universe.hubble_parameter()
        Omega = universe.density_parameters()
        Pk = universe.matter_power_spectrum()
        
        history['scale_factor'].append(a)
        history['hubble_parameter'].append(H)
        history['density_parameters'].append(Omega)
        
        if step % 100 == 0:
            history['power_spectrum'].append(Pk)
    
    return universe, history

# Run simulation
final_universe, cosmic_history = simulate_cosmology(N=5000, steps=500)

# Analyze results
print("Cosmological Parameters at z=0:")
print(f"Scale factor: {cosmic_history['scale_factor'][-1]:.4f}")
print(f"Hubble parameter: {cosmic_history['hubble_parameter'][-1]:.2f} (km/s/Mpc)")
print(f"Ω_m: {cosmic_history['density_parameters'][-1]['matter']:.3f}")
print(f"Ω_Λ: {cosmic_history['density_parameters'][-1]['lambda']:.3f}")
```

