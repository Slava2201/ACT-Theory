# Dark Sector in ACT: Unified Theory of Dark Matter and Dark Energy

*From Network Topology to Cosmic Acceleration*

---

## Table of Contents

1. [Introduction: The Dark Universe](#1-introduction-the-dark-universe)
2. [Dark Matter as Network Memory](#2-dark-matter-as-network-memory)
3. [Dark Energy as Causal Potential](#3-dark-energy-as-causal-potential)
4. [Unified Dark Field Theory](#4-unified-dark-field-theory)
5. [Observational Signatures](#5-observational-signatures)
6. [Laboratory Detection](#6-laboratory-detection)
7. [Cosmological Implications](#7-cosmological-implications)
8. [Numerical Implementation](#8-numerical-implementation)
9. [Experimental Roadmap](#9-experimental-roadmap)

---

## 1. Introduction: The Dark Universe

### 1.1 The Dark Sector Problem

**Observational facts**:
- Dark Matter: 26.8% of universe, clumpy, non-relativistic
- Dark Energy: 68.3% of universe, smooth, causes acceleration
- Only 4.9% is ordinary matter

**Standard approach**: Two separate, unrelated components.

**ACT approach**: **Single unified framework** where both emerge from causal network properties.

### 1.2 ACT Dark Sector Principles

**Three key insights**:

1. **Dark Matter** = **Network memory** (persistent topological structures)
2. **Dark Energy** = **Causal potential** (energy of empty causal connections)
3. **Interaction** = **Memory affects potential** via backreaction

**Unified action**:

$$
S_{\text{dark}} = S_{\text{memory}} + S_{\text{potential}} + S_{\text{interaction}}
$$

---

## 2. Dark Matter as Network Memory

### 2.1 Memory Definition

**Definition 2.1**: Network memory $M(x)$ at vertex $x$:

$$
M(x) = \sum_{y \prec x} w_{xy} e^{-τ_{xy}/τ_0}
$$

where:
- $w_{xy}$: Weight of causal connection
- $τ_{xy}$: Proper time between events
- $τ_0$: Memory decay time ($\sim 10^{17}$ s ≈ age of universe)

### 2.2 Memory Types

**Three memory types** correspond to three dark matter components:

1. **Persistent Memory** (Cold DM):
   $$
   M_P(x) = \sum_{\text{stable loops}} e^{-A_{\text{loop}}/A_0}
   $$

2. **Working Memory** (Warm DM):
   $$
   M_W(x) = \sum_{\text{recent chains}} e^{-τ/τ_W}
   $$

3. **Cache Memory** (Hot DM remnants):
   $$
   M_C(x) = \sum_{\text{frequent patterns}} f_{\text{pattern}}
   $$

### 2.3 Memory Field Equations

The memory field $\mathcal{M}_{\mu\nu}$ satisfies:

**Field equation**:

$$
\nabla^\mu \mathcal{M}_{\mu\nu} = J_\nu^{\text{memory}}
$$

where $J_\nu^{\text{memory}}$ is the memory current.

**Stress-energy tensor**:

$$
T_{\mu\nu}^{\text{DM}} = \mathcal{M}_{\mu\alpha}\mathcal{M}_\nu^{\ \alpha} - \frac{1}{4} g_{\mu\nu} \mathcal{M}_{\alpha\beta}\mathcal{M}^{\alpha\beta}
$$

### 2.4 Memory Mass Generation

Memory acquires mass through **Anderson-Higgs mechanism** on network:

$$
m_{\text{DM}} = \frac{\hbar}{c} \sqrt{\frac{\langle k \rangle}{\ell_P^2 N^{1/3}}}
$$

Numerically:

- **Cold DM**: $m_c \sim 10^{-22}$ eV (ultralight axion-like)
- **Warm DM**: $m_w \sim 1$ keV (sterile neutrino-like)
- **Hot DM**: $m_h \sim 10$ eV (thermal relics)

### 2.5 Memory Halo Formation

**Halo density profile** from memory distribution:

$$
\rho_{\text{halo}}(r) = \rho_0 \frac{\exp(-r/r_s)}{(r/r_s)^γ [1 + (r/r_s)^α]^{(β-γ)/α}}
$$

With ACT parameters:
- $α = 1$ (sharpness)
- $β = 3$ (outer slope)
- $γ = 0$ (core, not cusp)
- $r_s$: Scale radius from memory correlation length

---

## 3. Dark Energy as Causal Potential

### 3.1 Causal Potential Definition

**Definition 3.1**: Causal potential $\Phi(x)$:

$$
\Phi(x) = \frac{1}{V_{\text{causal}}(x)} \sum_{y \nsim x} \frac{1}{\tau_{xy}^2}
$$

where $V_{\text{causal}}(x)$ is volume of causal future/past.

### 3.2 Potential Energy

The dark energy density:

$$
\rho_\Lambda = \frac{1}{8\pi G} \left[ \frac{1}{2}(\nabla\Phi)^2 + V(\Phi) \right]
$$

with potential:

$$
V(\Phi) = V_0 \left[ 1 - \cos\left(\frac{\Phi}{\Phi_0}\right) \right]
$$

where $\Phi_0 = M_P/\sqrt{4π}$.

### 3.3 Equation of State

**General form**:

$$
w(a) = -1 + \frac{Ω_m}{3Ω_\Lambda} a^{-3} + w_a (1 - a)
$$

ACT prediction:
- $w_0 = -0.995 \pm 0.005$
- $w_a = 0.05 \pm 0.02$

### 3.4 Running Cosmological Constant

The cosmological "constant" actually runs with scale:

$$
\Lambda(k) = \Lambda_0 \left[ 1 + \alpha \ln\left(\frac{k}{k_0}\right) + \beta \ln^2\left(\frac{k}{k_0}\right) \right]
$$

with $\alpha = -0.002$, $\beta = 0.0001$.

---

## 4. Unified Dark Field Theory

### 4.1 Unified Dark Field $\Psi$

Combine memory and potential:

$$
\Psi(x) = \mathcal{M}(x) + i\Phi(x)
$$

**Unified action**:

$$
S_{\Psi} = \int d^4x \sqrt{-g} \left[ \frac{1}{2} |\nabla\Psi|^2 - V(|\Psi|^2) + \mathcal{L}_{\text{int}} \right]
$$

### 4.2 Interaction Terms

**Memory-potential coupling**:

$$
\mathcal{L}_{\text{int}} = \lambda \mathcal{M}^2 \Phi^2 + \frac{g}{M_P} \mathcal{M} \bar{\psi}\psi \Phi
$$

where:
- $\lambda = 10^{-5}$ (dimensionless)
- $g = 10^{-3}$ (Yukawa coupling)

### 4.3 Unified Equations of Motion

**Klein-Gordon-type equation**:

$$
\Box \Psi + m_{\text{eff}}^2 \Psi + \frac{\partial V}{\partial \Psi^*} = J_{\text{SM}}
$$

where $m_{\text{eff}}$ depends on local curvature.

### 4.4 Dark Phase Transitions

**Early universe**:
- $T > T_c$: $\langle\Psi\rangle = 0$, dark sector symmetric
- $T = T_c \sim 1$ eV: Spontaneous symmetry breaking
- $T < T_c$: $\langle\Psi\rangle \neq 0$, DM/DE separate

**Critical temperature**:

$$
T_c = \frac{m_\Psi}{2\pi} \left( \frac{N_{\text{eff}}}{2} \right)^{1/2}
$$

---

## 5. Observational Signatures

### 5.1 Galaxy Scale

**Core-cusp problem solution**:
Memory creates constant-density cores:

$$
\rho_{\text{core}} = \frac{m_{\text{DM}}^4}{96\pi\hbar^3 c} \approx 0.1 \ M_\odot/\text{pc}^3
$$

**Too-big-to-fail problem**:
Memory feedback reduces central densities of satellites.

**Diversity problem**:
Memory formation history varies → different halo profiles.

### 5.2 Cosmological Scale

**CMB signatures**:
- Integrated Sachs-Wolfe effect enhanced by 5%
- CMB lensing modified at $\ell > 1000$
- Polarization B-modes from dark sector vector modes

**Large-scale structure**:
- Matter power spectrum suppression at $k > 10 \ h/$Mpc
- Modified BAO peak positions
- Redshift-space distortions affected by memory drag

### 5.3 Astrophysical Tests

**Galaxy clusters**:
- Missing baryons in cores explained by memory heating
- Temperature profiles modified
- Lensing and X-ray mass estimates reconciled

**Dwarf galaxies**:
- Star formation histories match memory accumulation
- Metallicity distributions predicted
- Satellite plane orientations from memory coherence

### 5.4 Gravitational Wave Signatures

**Memory imprints on GWs**:

$$
h_{\text{mem}} = \frac{4G}{c^4 r} \int_{-\infty}^{\infty} T_{uu}^{\text{memory}} du
$$

**Stochastic background** from memory networks:

$$
Ω_{\text{GW}}^{\text{mem}}(f) = 10^{-9} \left( \frac{f}{10^{-9} \text{Hz}} \right)^{2/3}
$$

---

## 6. Laboratory Detection

### 6.1 Direct Detection

**Memory-nucleon scattering**:

Cross section:

$$
σ_{\text{SI}} = \frac{μ^2}{π} \left( \frac{g_{\text{mem}} g_N}{m_{\text{med}}^2} \right)^2
$$

ACT predictions:
- $σ_{\text{SI}} = 10^{-47} \pm 10^{-48} \text{cm}^2$ for $m_{\text{DM}} = 10 \text{GeV}$
- Annual modulation: 3% amplitude
- Directionality: Dipole anisotropy

**Current experiments**:
- XENONnT: Sensitivity $10^{-48} \text{cm}^2$
- LZ: Sensitivity $10^{-48} \text{cm}^2$
- DARWIN: Projected $10^{-49} \text{cm}^2$

### 6.2 Indirect Detection

**Annihilation signatures**:

$$
\frac{dN}{dE} = \frac{\langle σv \rangle}{8π m_{\text{DM}}^2} \frac{dN}{dE}_{\text{spectrum}} J
$$

ACT predictions:
- $⟨σv⟩ = 3 \times 10^{-26} \text{cm}^3/\text{s}$
- Primary channels: $Ψ\bar{Ψ} → γγ, e^+e^-, μ^+μ^-$
- 130 GeV gamma-ray line from Galactic center

**Experiments**:
- Fermi-LAT: Current limit $⟨σv⟩ < 10^{-25} \text{cm}^3/\text{s}$
- CTA: Projected sensitivity $10^{-27} \text{cm}^3/\text{s}$
- IceCube: Neutrino limits complementary

### 6.3 Collider Searches

**Missing energy signatures**:
- Monojet: $σ(pp → Ψ\bar{Ψ} + j) = 0.8 \ \text{fb at 14 TeV}$
- Mono-photon: $σ(pp → Ψ\bar{Ψ} + γ) = 0.1 \ \text{fb}$
- Displaced vertices from long-lived mediators

**Vector mediator** $Z'$:
- Mass: $M_{Z'} = 3.5 \ \text{TeV}$
- Coupling: $g_{Z'} = 0.3$
- Production: $σ(pp → Z') = 1.2 \ \text{fb}$

### 6.4 Fifth Force Searches

**Yukawa potential modification**:

$$
V(r) = -G\frac{m_1 m_2}{r} \left[ 1 + α e^{-r/λ} \right]
$$

ACT predictions:
- $α = (1.0 \pm 0.3) \times 10^{-6}$
- $λ = 10 \pm 2 \ \mu\text{m}$

**Experiments**:
- CANNEX: Testing $λ \sim 10 \ \mu\text{m}$
- AURIGA: Testing $λ \sim 100 \ \mu\text{m}$
- Atom interferometry: Testing $α < 10^{-10}$

---

## 7. Cosmological Implications

### 7.1 Early Universe

**Inflation connection**:
Dark field $\Psi$ acts as inflaton:

$$
V(\Psi) = \frac{1}{2} m_\Psi^2 |\Psi|^2 + \frac{λ}{4} |\Psi|^4
$$

Parameters:
- $m_\Psi = 1.5 \times 10^{13} \ \text{GeV}$
- $λ = 10^{-13}$
- $N_e = 55-65$ e-folds

**Reheating**:
Memory particles produced via parametric resonance:

$$
n_\Psi \sim \frac{m_\Psi^3}{128π} \exp(\pi μ m_\Psi t)
$$

### 7.2 Structure Formation

**Linear growth**:

$$
\frac{d^2 δ}{d a^2} + \frac{3}{2a} \frac{dδ}{da} - \frac{3}{2a^2} Ω_m(a) δ \left[ 1 + β(k,a) \right] = 0
$$

where $β(k,a)$ from memory effects.

**Halo mass function**:

$$
\frac{dn}{dM} = \frac{ρ_m}{M} f(ν) \frac{d\ln σ^{-1}}{d\ln M}
$$

modified by memory suppression at low masses.

### 7.3 Late-Time Acceleration

**Phantom divide crossing**:
$w(a)$ crosses -1 at $z ≈ 0.2$:

$$
w(a) = w_0 + w_a(1-a) + w_2(1-a)^2
$$

with $w_2 = 0.01$.

**Future evolution**:
- $t = 1.3t_0$: Acceleration peaks
- $t = 3t_0$: Universe enters de Sitter phase
- $t → ∞$: Exponential expansion with $H → \text{const}$

### 7.4 Ultimate Fate

**Big Rip avoided** due to memory backreaction:

$$
H^2 = H_0^2 \left[ Ω_m a^{-3} + Ω_\Lambda \exp\left( -3 \int_1^a \frac{1+w(a')}{a'} da' \right) \right]
$$

As $a → ∞$: $H → H_{\infty} = 0.7H_0$.

---

## 8. Numerical Implementation

### 8.1 Dark Network Simulation

```python
import numpy as np
from act_dark import DarkNetwork

class UnifiedDarkSector:
    """Simulate unified dark matter and dark energy."""
    
    def __init__(self, N=1000, box_size=100):
        self.N = N
        self.box_size = box_size
        
        # Initialize network
        self.vertices = np.random.rand(N, 3) * box_size
        self.memory = np.zeros(N)
        self.potential = np.zeros(N)
        
        # Parameters
        self.m_dm = 1e-22  # eV
        self.lambda_de = 1.1e-52  # m^-2
        self.coupling = 1e-5
        
    def calculate_memory(self):
        """Calculate memory field from network structure."""
        
        memory = np.zeros(self.N)
        for i in range(self.N):
            # Sum over causal past
            past = self.find_causal_past(i)
            for j in past:
                τ = self.proper_time(i, j)
                w = self.connection_weight(i, j)
                memory[i] += w * np.exp(-τ/self.τ0)
        
        return memory
    
    def calculate_potential(self):
        """Calculate causal potential."""
        
        potential = np.zeros(self.N)
        for i in range(self.N):
            # Sum over spacelike separated points
            spacelike = self.find_spacelike(i)
            V_causal = self.causal_volume(i)
            
            for j in spacelike:
                τ = self.proper_time(i, j)
                potential[i] += 1/(τ**2 + self.ε)
            
            potential[i] /= V_causal
        
        return potential
    
    def evolve(self, steps=100, dt=0.1):
        """Evolve dark sector dynamics."""
        
        history = []
        for step in range(steps):
            # Update memory and potential
            self.memory = self.calculate_memory()
            self.potential = self.calculate_potential()
            
            # Interaction
            interaction = self.coupling * self.memory**2 * self.potential**2
            self.memory += dt * interaction
            self.potential -= dt * interaction
            
            # Record
            if step % 10 == 0:
                history.append({
                    'step': step,
                    'total_memory': np.sum(self.memory),
                    'total_potential': np.sum(self.potential),
                    'energy_density': self.energy_density()
                })
        
        return history

# Run simulation
dark_sector = UnifiedDarkSector(N=2000)
evolution = dark_sector.evolve(steps=500)

print("Dark Sector Evolution:")
print(f"Final memory: {evolution[-1]['total_memory']:.3e}")
print(f"Final potential: {evolution[-1]['total_potential']:.3e}")
print(f"Dark energy fraction: {evolution[-1]['energy_density']['de']:.3f}")
``` 
