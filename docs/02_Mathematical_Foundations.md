# 02. Mathematical Foundations of Algebraic Causality Theory

# Mathematical Foundations of ACT Theory

## Table of Contents

1. [Causal Set Theory](#1-causal-set-theory)
2. [Regge Calculus](#2-regge-calculus)
3. [Dirac Operator on Causal Sets](#3-dirac-operator-on-causal-sets)
4. [Geometric Quantization](#4-geometric-quantization)
5. [Group Theory and Symmetries](#5-group-theory-and-symmetries)
6. [Topological Invariants](#6-topological-invariants)

## 1. Causal Set Theory

### 1.1 Basic Definitions

A **causal set** $\mathcal{C}$ is a locally finite partially ordered set:

$$
\mathcal{C} = (P, \prec)
$$

where:
- $P$ is a set of elements (events)
- $\prec$ is a partial order representing causal precedence
- **Local finiteness**: $\forall x,y \in P$, the set $\{z \in P \mid x \prec z \prec y\}$ is finite

### 1.2 Causal Interval

The number of elements in the causal interval between $x$ and $y$:

$$
N(x,y) = |\{z \in P \mid x \prec z \prec y\}|
$$

This provides a measure of **proper time**.

### 1.3 Sprinkling Process

To obtain a causal set from a Lorentzian manifold $(M,g)$:

1. **Poisson sprinkling**: Randomly select points with density $\rho$
2. **Induce causal relations**: $x \prec y$ iff $y$ is in the future light cone of $x$

The expected number of points in a volume $V$ is:

$$
\mathbb{E}[N] = \rho V
$$

with $\rho \sim \ell_P^{-4}$ at the Planck scale.

## 2. Regge Calculus

### 2.1 Simplicial Decomposition

Spacetime is approximated by a **simplicial complex**:

$$
\mathcal{M} \approx \bigcup_i \sigma_i^{(n)}
$$

where $\sigma_i^{(n)}$ are $n$-simplices (triangles, tetrahedra, etc.).

### 2.2 Regge Action

For a 4D simplicial complex:

$$
S_{\text{Regge}} = \frac{1}{8\pi G} \sum_{\text{hinges } h} A_h \delta_h - \Lambda \sum_{\text{simplices } s} V_s
$$

where:
- $A_h$ = area of hinge (2-simplex)
- $\delta_h$ = deficit angle at hinge
- $V_s$ = volume of 4-simplex
- $\Lambda$ = cosmological constant

### 2.3 Deficit Angle

The deficit angle $\delta_h$ measures curvature:

$$
\delta_h = 2\pi - \sum_{s \supset h} \theta_{h,s}
$$

where $\theta_{h,s}$ is the dihedral angle of simplex $s$ at hinge $h$.

## 3. Dirac Operator on Causal Sets

### 3.1 Definition

**Definition 3.1 (Causal Dirac Operator):**
The discrete Dirac operator $D$ acts on spinors $\psi_x$ associated to each element:

$$
(D\psi)_x = \sum_{y \prec x \text{ or } x \prec y} \kappa(x,y) \psi_y
$$

where $\kappa(x,y)$ is a kernel encoding causal relations and distances.

### 3.2 Matrix Representation

In matrix form:

$$
D_{xy} = 
\begin{cases} 
\dfrac{i}{\ell_P} C(x,y) & \text{if } x \prec y \text{ or } y \prec x \\ 
0 & \text{otherwise}
\end{cases}
$$

with $C(x,y)$ encoding the causal structure:

$$
C(x,y) = \frac{1}{\sqrt{N(x,y)}} e^{i\phi(x,y)}
$$

where $\phi(x,y)$ is a phase factor.

### 3.3 Spectrum and Dimension

The spectral dimension $d_s$ is obtained from the return probability:

$$
P(t) = \text{Tr}(e^{-tD^2}) \sim t^{-d_s/2} \quad \text{as } t \to 0
$$

For ACT theory:

$$
d_s = 
\begin{cases}
4 & \text{at Planck scale} \\
2 & \text{at large scales}
\end{cases}
$$

### 3.4 Continuum Limit

In the continuum limit, $D$ reduces to the standard Dirac operator:

$$
\lim_{N \to \infty} D = i\gamma^\mu \partial_\mu + m
$$

where $\gamma^\mu$ are Dirac matrices.

## 4. Geometric Quantization

### 4.1 Quantization Condition

The fine-structure constant emerges from geometric quantization:

$$
\frac{1}{\alpha} = 4\pi \frac{\langle V \rangle}{\ell_P^2} \ln N
$$

where:
- $\alpha$ = fine-structure constant
- $\langle V \rangle$ = average simplex volume
- $\ell_P$ = Planck length
- $N$ = number of simplices

### 4.2 Derivation

Starting from the path integral:

$$
Z = \int \mathcal{D}A \exp\left(iS[A]\right)
$$

with action:

$$
S[A] = \frac{1}{4e^2} \int F_{\mu\nu} F^{\mu\nu} \sqrt{-g} d^4x
$$

On a simplicial complex, this becomes:

$$
S_{\text{lattice}}[U] = \frac{1}{e^2} \sum_{\text{plaquettes } p} \text{Re Tr}(1 - U_p)
$$

where $U_p$ is the product of link variables around plaquette $p$.

### 4.3 Gauge Couplings

The gauge couplings emerge from group theory:

$$
\frac{1}{\alpha_i} = \frac{C_2(G_i)}{4\pi} \frac{\langle A \rangle}{\ell_P^2}
$$

where $C_2(G_i)$ is the quadratic Casimir of gauge group $G_i$.

## 5. Group Theory and Symmetries

### 5.1 Emergent Gauge Groups

Standard Model gauge groups emerge from the fundamental group of the causal set:

$$
SU(3)_C \times SU(2)_L \times U(1)_Y \subset \pi_1(\mathcal{C})
$$

### 5.2 Representation Theory

Particle representations correspond to irreducible representations of stabilizer subgroups:

| Particle | Representation | Origin |
|----------|----------------|---------|
| Quarks | $\mathbf{3}$ of $SU(3)$ | Triangulation vertices |
| Leptons | $\mathbf{1}$ of $SU(3)$ | Tetrahedron centers |
| Gauge bosons | Adjoint rep | Simplex edges |

### 5.3 Symmetry Breaking

Electroweak symmetry breaking emerges naturally:

$$
SU(2)_L \times U(1)_Y \to U(1)_{\text{EM}}
$$

through a geometric Higgs mechanism.

## 6. Topological Invariants

### 6.1 Euler Characteristic

For a simplicial complex:

$$
\chi = \sum_{k=0}^n (-1)^k f_k
$$

where $f_k$ is the number of $k$-simplices.

### 6.2 Betti Numbers

The $k$-th Betti number $b_k$ counts $k$-dimensional holes:

$$
b_k = \dim H_k(\mathcal{C})
$$

where $H_k$ is the $k$-th homology group.

### 6.3 Chern Classes

Gauge fields are associated with Chern classes:

$$
c_k(F) = \frac{1}{k!} \text{Tr}\left(\frac{iF}{2\pi}\right)^k
$$

which are topological invariants.

## 7. Mathematical Consistency

### 7.1 Continuum Limit

The continuum limit exists if:

$$
\lim_{N \to \infty} \frac{\langle V \rangle}{\ell_P^4} = \text{finite}
$$

### 7.2 Renormalization Group Flow

The $\beta$-function for the gravitational coupling:

$$
\beta_G = \mu \frac{\partial G}{\partial \mu} = \frac{G^2}{\ell_P^2} (a + b G \Lambda + \cdots)
$$

### 7.3 Fixed Points

The theory has UV and IR fixed points:

- **UV fixed point**: $G^* \sim \ell_P^2$, asymptotically safe
- **IR fixed point**: $G \to G_N$, Newton's constant

## 8. Computational Methods

### 8.1 Monte Carlo Simulations

The partition function is evaluated using Markov Chain Monte Carlo:

$$
\langle \mathcal{O} \rangle = \frac{1}{Z} \sum_{\mathcal{T}} \mathcal{O}[\mathcal{T}] e^{-S[\mathcal{T}]}
$$

### 8.2 Numerical Techniques

- **Heat kernel methods**: For spectral dimension
- **Dynamical triangulations**: For path integral
- **Renormalization group**: For continuum limit

## Appendix A: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| $\mathcal{C}$ | Causal set |
| $\prec$ | Causal relation |
| $\ell_P$ | Planck length |
| $D$ | Dirac operator |
| $S_{\text{Regge}}$ | Regge action |
| $\alpha$ | Fine-structure constant |
| $G$ | Gravitational constant |
| $\Lambda$ | Cosmological constant |

## References

1. Bombelli, L., Lee, J., Meyer, D., & Sorkin, R. D. (1987). *Spacetime as a causal set*
2. Regge, T. (1961). *General relativity without coordinates*
3. Connes, A. (1994). *Noncommutative geometry*
4. Ambjørn, J., Jurkiewicz, J., & Loll, R. (2005). *Quantum Gravity as Sum over Spacetimes*
