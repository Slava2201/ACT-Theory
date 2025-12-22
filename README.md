# Algebraic Causality Theory (ACT)

*A Fundamental Theory of Quantum Gravity and Emergent Spacetime*

![ACT Framework](docs/images/act_framework.png)

## Abstract

Algebraic Causality Theory (ACT) is a novel approach to quantum gravity that posits spacetime as emergent from fundamental causal relations between quantum events. The theory provides:

1. **Unification** of general relativity and quantum mechanics
2. **Emergence** of the Standard Model from network topology
3. **Prediction** of dark matter as topological defects
4. **Testable signatures** for LHC and gravitational wave observatories

## Mathematical Foundations

### Core Principles

- **Causal Sets**: Discrete partially ordered sets (posets) as fundamental entities
- **Algebraic Quantum Field Theory**: Local operators on causal sets
- **Emergent Geometry**: Spacetime metric from causal intervals

### Fundamental Equations

#### 1. Causal Action Principle

The dynamics follows from the action:

$$
S[\mathcal{C}] = \alpha \sum_{x \prec y} V(x,y) - \beta \sum_{\Delta} R(\Delta) + \gamma \sum_{\mathcal{D}} Q(\mathcal{D})
$$

where:
- $\mathcal{C}$ is the causal set
- $V(x,y)$ is the causal volume between events $x$ and $y$
- $R(\Delta)$ is the Regge curvature on simplex $\Delta$
- $Q(\mathcal{D})$ is the topological charge of defect $\mathcal{D}$

#### 2. Quantum Amplitude

The path integral over causal structures:

$$
Z = \int \mathcal{D}[\mathcal{C}] \, e^{iS[\mathcal{C}]/\hbar}
$$

#### 3. Emergent Metric

The continuum metric emerges as:

$$
g_{\mu\nu}(x) = \lim_{\rho \to \infty} \frac{1}{\rho} \sum_{y \in C(x,\rho)} \frac{(x^\mu - y^\mu)(x^\nu - y^\nu)}{V(x,y)}
$$

where $C(x,\rho)$ is the causal diamond of size $\rho$.

#### 4. Einstein Equations from Causal Structure

The Einstein equations emerge in the continuum limit:

$$
R_{\mu\nu} - \frac{1}{2} g_{\mu\nu} R + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4} \langle T_{\mu\nu} \rangle_{\mathcal{C}}
$$

where $\langle T_{\mu\nu} \rangle_{\mathcal{C}}$ is the expectation value of the stress-energy tensor over causal sets.

#### 5. Quantum Field Equations

Matter fields emerge as:

$$
(i\gamma^\mu \nabla_\mu - m)\psi(x) = \lambda \phi(x)\psi(x) + \sum_{y \prec x} K(x,y)\psi(y)
$$

with non-local kernel $K(x,y)$ encoding causal structure.

#### 6. Dark Matter Defect Equation

Topological defects satisfy:

$$
\Box \Phi + \frac{\partial V(\Phi)}{\partial \Phi^*} = j_{\text{top}}(x)
$$

where $j_{\text{top}}(x)$ is the topological current.

#### 7. Modified Dispersion Relation

From quantum gravity effects:

$$
E^2 = p^2 c^2 + m^2 c^4 + \alpha \frac{p^4 c^2}{M_{\text{pl}}^2} + \beta \frac{p^6 c^4}{M_{\text{pl}}^4}
$$

#### 8. Non-commutative Geometry

Spacetime non-commutativity:

$$
[x^\mu, x^\nu] = i\theta^{\mu\nu}, \quad \theta^{\mu\nu} = \ell_P^2 \begin{pmatrix}
0 & 1 & 0 & 0 \\
-1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & -1 & 0
\end{pmatrix}
$$

#### 9. Holographic Entropy Bound

From the holographic principle:

$$
S_{\text{max}} = \frac{A}{4\ell_P^2} \quad \text{with} \quad A = 4\pi R_S^2
$$

#### 10. Regge Calculus

Curvature from deficit angles:

$$
R = 2 \sum_{\Delta} \epsilon_\Delta \delta_\Delta
$$

where $\delta_\Delta = 2\pi - \sum_i \theta_i$ is the deficit angle.

## Derivation Overview

### From Discrete to Continuum

Starting from the discrete action:

$$
S_{\text{disc}} = \sum_{\langle ij \rangle} J_{ij} A_{ij} - \sum_{\Delta} \lambda V_\Delta + \sum_{\mathcal{D}} \kappa Q_{\mathcal{D}}
$$

we take the continuum limit $N \to \infty$, $\ell_P \to 0$ while keeping $N\ell_P^3 = V$ fixed:

$$
S_{\text{cont}} = \int d^4x \sqrt{-g} \left[ \frac{R - 2\Lambda}{16\pi G} + \mathcal{L}_{\text{matter}} + \mathcal{L}_{\text{top}} \right]
$$

### Emergence of Gauge Symmetries

Local SU(N) symmetries emerge from network connectivity:

$$
U_{ij} = \exp\left(i \int_{i}^{j} A_\mu dx^\mu\right)
$$

with field strength:

$$
F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu - i[A_\mu, A_\nu]
$$

## Numerical Implementation

### Discrete Action Implementation

In code, the action is computed as:

```python
def calculate_action(self):
    total_action = 0.0
    for tetra in self.tetrahedra:
        # Regge term
        volume = self._tetrahedron_volume(tetra)
        deficit = self._deficit_angle(tetra)
        total_action += self.alpha * volume - self.beta * deficit
    
    # Topological term
    total_action += self.gamma * self._topological_charge()
    return total_action
```

Metric Reconstruction
The emergent metric tensor is reconstructed via:
$$ g_{\mu\nu}(x_i) = \frac{1}{N_{\text{neigh}}} \sum_{j \in N(x_i)} \tau_{ij}^2 \Delta x_{ij}^\mu \Delta x_{ij}^\nu $$

