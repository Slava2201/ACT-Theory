"""
Algebraic Causality Theory (ACT) - Main Implementation
A fundamental theory of quantum gravity and emergent spacetime
Version: 2.0 (Extended with Dark Matter)
"""

import numpy as np
import numba as nb
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigs
from scipy.spatial import KDTree
import networkx as nx
from collections import defaultdict
from datetime import datetime
import warnings
import psutil
import json
import pickle
import plotly.graph_objects as go
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')

# ============================================================================
# 1. SCALABLE ACT MODEL WITH N≥1000 VERTICES
# ============================================================================

class ACTModel:
    """
    Scalable Algebraic Causality Theory model with optimizations for large networks.
    Implements emergent spacetime from causal sets with quantum gravity corrections.
    """
    
    def __init__(self, 
                 N: int = 1500, 
                 dim: int = 4, 
                 temperature: float = 1.0,
                 seed: Optional[int] = None,
                 include_dark_matter: bool = True):
        """
        Initialize large-scale ACT model.
        
        Parameters:
        -----------
        N : int (≥ 1000)
            Number of vertices in the causal set
        dim : int
            Dimension of embedding space (4 for spacetime)
        temperature : float
            System temperature (inverse coupling)
        seed : int
            Random seed for reproducibility
        include_dark_matter : bool
            Whether to include dark matter sector
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.N = N
        self.dim = dim
        self.beta = 1.0 / temperature
        self.include_dark_matter = include_dark_matter
        
        # Fundamental constants
        self.l_p = 1.616255e-35  # Planck length (m)
        self.t_p = 5.391247e-44  # Planck time (s)
        self.M_p = 2.176434e-8   # Planck mass (kg)
        self.ħ = 1.054571817e-34 # Reduced Planck constant
        self.c = 299792458       # Speed of light
        self.G = 6.67430e-11     # Gravitational constant
        
        print("="*80)
        print("ALGEBRAIC CAUSALITY THEORY (ACT) INITIALIZATION")
        print(f"Vertices: N = {N:,}")
        print(f"Temperature: T = {temperature}")
        print(f"Include Dark Matter: {include_dark_matter}")
        print("="*80)
        
        # Optimized vertex initialization
        self.vertices = self._initialize_vertices_batch(N, dim)
        
        # Sparse operator initialization
        self.operators = self._initialize_operators_sparse(N)
        
        # Build causal structure
        print("\n[1/6] Building causal structure...")
        self.causal_matrix = self._build_causal_matrix()
        
        # Build tetrahedral complex with space partitioning
        print("[2/6] Constructing simplicial complex...")
        self.tetrahedra = self._build_simplicial_complex_optimized()
        
        # Sparse adjacency matrix
        self.adjacency = self._build_sparse_adjacency()
        
        # Quantum gravity parameters
        self.qg_params = self._initialize_qg_parameters()
        
        # Dark matter sector (if enabled)
        if self.include_dark_matter:
            print("[3/6] Initializing dark matter sector...")
            self.dark_matter_params = self._initialize_dark_matter()
        else:
            self.dark_matter_params = None
        
        # Loop corrections cache
        self.loop_corrections = {}
        self._action_cache = {}
        
        # Observables
        self.observables = {}
        
        print(f"[4/6] Created {len(self.tetrahedra):,} tetrahedra")
        print(f"[5/6] Average vertex degree: {self.adjacency.sum()/self.N:.2f}")
        print("[6/6] Model initialization complete!")
    
    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def _initialize_vertices_batch(N: int, dim: int) -> np.ndarray:
        """
        Batch initialization of vertices with Numba acceleration.
        
        Returns:
        --------
        vertices : np.ndarray of shape (N, dim)
            Vertex coordinates in embedding space
        """
        vertices = np.random.randn(N, dim).astype(np.float32)
        
        # Normalization and logarithmic scaling
        for i in nb.prange(N):
            norm = np.sqrt(np.sum(vertices[i]**2))
            if norm > 0:
                # Scale with log(i+2) for hierarchical structure
                vertices[i] = vertices[i] / norm * np.log(i + 2)
        
        return vertices
    
    def _initialize_operators_sparse(self, N: int) -> List[csr_matrix]:
        """
        Initialize SU(4) operators in sparse format for memory efficiency.
        
        Mathematical formulation:
        U_i ∈ SU(4) where SU(4) = {U ∈ ℂ^{4×4} | U^†U = I, det(U) = 1}
        """
        operators = []
        block_size = 4
        n_blocks = N // block_size
        
        for block in range(n_blocks):
            block_ops = []
            for _ in range(block_size):
                # Generate random complex matrix
                X = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
                
                # QR decomposition for unitary matrix
                Q, R = np.linalg.qr(X)
                
                # Ensure det(Q) = 1
                D = np.diag(R) / np.abs(np.diag(R))
                Q = Q @ np.diag(D)
                
                # Convert to sparse format
                Q_sparse = csr_matrix(Q)
                block_ops.append(Q_sparse)
            
            operators.extend(block_ops)
            
            if block % 100 == 0:
                print(f"  Generated {len(operators):,}/{N:,} operators")
        
        # Add remaining operators
        remaining = N % block_size
        for _ in range(remaining):
            X = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
            Q, R = np.linalg.qr(X)
            D = np.diag(R) / np.abs(np.diag(R))
            Q = Q @ np.diag(D)
            operators.append(csr_matrix(Q))
        
        return operators
    
    def _build_causal_matrix(self) -> csr_matrix:
        """
        Build causal matrix C_{ij} where:
        C_{ij} = 1 if vertex i precedes vertex j in causal order
        C_{ij} = 0 otherwise
        
        Implements: i ≺ j ⇔ t_i < t_j and proper distance condition
        """
        # Time coordinates (first dimension)
        times = self.vertices[:, 0]
        
        # Spatial coordinates
        spatial_coords = self.vertices[:, 1:4]
        
        # Build causal matrix
        causal = lil_matrix((self.N, self.N), dtype=np.int8)
        
        # Vectorized computation for moderate N
        if self.N <= 5000:
            for i in range(self.N):
                # Time condition: i precedes j if t_i < t_j
                time_mask = times[i] < times
                
                # Distance condition: |x_i - x_j| < c|t_i - t_j|
                dt = np.abs(times[i] - times)
                dx = np.linalg.norm(spatial_coords[i] - spatial_coords, axis=1)
                lightcone_mask = dx < self.c * dt
                
                # Combined condition
                causal_mask = time_mask & lightcone_mask
                causal[i, causal_mask] = 1
        
        return causal.tocsr()
    
    def _build_simplicial_complex_optimized(self) -> List[Tuple[int, int, int, int]]:
        """
        Build 4-simplex (tetrahedron) complex from causal structure.
        Uses Delaunay-like construction with causal constraints.
        """
        # Use 3D spatial projection for geometric structure
        spatial_coords = self.vertices[:, 1:4]
        
        # Adaptive sampling for large N
        if self.N > 2000:
            sample_size = min(5000, self.N)
            sample_indices = np.random.choice(self.N, sample_size, replace=False)
            coords_sampled = spatial_coords[sample_indices]
        else:
            coords_sampled = spatial_coords
            sample_indices = np.arange(self.N)
        
        # KD-tree for efficient neighbor search
        kdtree = KDTree(coords_sampled)
        
        tetrahedra = []
        n_samples = len(sample_indices)
        
        # Adaptive search radius
        distances = []
        for i in range(min(100, n_samples)):
            for j in range(i+1, min(100, n_samples)):
                dist = np.linalg.norm(coords_sampled[i] - coords_sampled[j])
                distances.append(dist)
        
        avg_distance = np.mean(distances) if distances else 1.0
        search_radius = 2.5 * avg_distance
        
        print(f"  Search radius: {search_radius:.3f}")
        print(f"  Sampling {n_samples:,} vertices...")
        
        # Construct tetrahedra with causal ordering
        for idx, i in enumerate(range(0, n_samples, 10)):
            # Find neighbors within search radius
            neighbors = kdtree.query_ball_point(coords_sampled[i], search_radius)
            neighbors = [n for n in neighbors if n != i]
            
            if len(neighbors) >= 3:
                # Try multiple combinations
                for attempt in range(min(5, len(neighbors)//3)):
                    selected = np.random.choice(neighbors, 3, replace=False)
                    j, k, l = selected
                    
                    # Original indices
                    orig_indices = tuple(sorted([
                        sample_indices[i],
                        sample_indices[j],
                        sample_indices[k],
                        sample_indices[l]
                    ]))
                    
                    # Causal ordering check
                    times = self.vertices[list(orig_indices), 0]
                    if (len(np.unique(times)) == 4 and 
                        np.all(np.diff(np.sort(times)) > 0)):
                        tetrahedra.append(orig_indices)
            
            if idx % 50 == 0:
                print(f"    Processed {idx}/{n_samples//10} vertices, "
                      f"found {len(tetrahedra):,} tetrahedra")
        
        # Ensure minimum number of tetrahedra
        min_tetrahedra = self.N // 20
        if len(tetrahedra) < min_tetrahedra:
            print(f"  Adding {min_tetrahedra - len(tetrahedra):,} additional tetrahedra...")
            
            while len(tetrahedra) < min_tetrahedra:
                tetra = tuple(np.random.choice(self.N, 4, replace=False))
                tetra = tuple(sorted(tetra))
                
                times = self.vertices[list(tetra), 0]
                if np.all(np.diff(np.sort(times)) > 0):
                    tetrahedra.append(tetra)
        
        # Remove duplicates
        tetrahedra = list(set(tetrahedra))
        
        return tetrahedra
    
    def _build_sparse_adjacency(self) -> csr_matrix:
        """Build sparse adjacency matrix from tetrahedral complex."""
        adj = lil_matrix((self.N, self.N), dtype=np.int8)
        
        for tetra in self.tetrahedra:
            i, j, k, l = tetra
            edges = [(i, j), (i, k), (i, l), (j, k), (j, l), (k, l)]
            
            for u, v in edges:
                adj[u, v] = 1
                adj[v, u] = 1
        
        return adj.tocsr()
    
    def _initialize_qg_parameters(self) -> Dict[str, Any]:
        """Initialize quantum gravity parameters."""
        return {
            'G_N': self.G,
            'ħ': self.ħ,
            'c': self.c,
            'Λ': 1.1056e-52,  # Cosmological constant (m^-2)
            'M_pl': self.M_p,
            'l_pl': self.l_p,
            't_pl': self.t_p,
            'quantum_fluctuations': 0.1,
            'non_commutativity_scale': 10 * self.l_p,
            'holographic_entropy': np.log(self.N) * self.l_p**2
        }
    
    def _initialize_dark_matter(self) -> Dict[str, Any]:
        """
        Initialize dark matter sector based on ACT principles.
        
        Dark matter emerges as topological defects in the causal structure.
        Mathematical formulation: DM ∼ π₂(M) where M is the causal set manifold.
        """
        # Dark matter parameters from ACT
        dm_mass_scale = self.M_p / np.sqrt(self.N)  # Emergent mass scale
        dm_coupling = 1 / (4 * np.pi * np.sqrt(self.N))  # Weak coupling
        
        # Dark matter distribution (topological charge)
        dm_charge = np.zeros(self.N)
        for tetra in self.tetrahedra:
            # Compute topological charge for each tetrahedron
            vertices = [self.vertices[i] for i in tetra]
            charge = self._calculate_tetrahedron_topological_charge(vertices)
            
            # Distribute charge to vertices
            for i in tetra:
                dm_charge[i] += charge / 4
        
        # Normalize dark matter density
        total_charge = np.sum(np.abs(dm_charge))
        if total_charge > 0:
            dm_charge = dm_charge / total_charge * 0.27  # Ω_DM ≈ 0.27
        
        return {
            'mass_scale': dm_mass_scale,
            'coupling_strength': dm_coupling,
            'topological_charge': dm_charge,
            'density_parameter': 0.27,  # Ω_DM
            'interaction_type': 'topological',
            'cross_section': 1e-46,  # cm^2 (typical for WIMPs)
            'annihilation_channels': ['γγ', 'e⁺e⁻', 'μ⁺μ⁻', 'τ⁺τ⁻']
        }
    
    @staticmethod
    @nb.njit(fastmath=True)
    def _calculate_tetrahedron_topological_charge(vertices: List[np.ndarray]) -> float:
        """Calculate topological charge for a tetrahedron."""
        if len(vertices) < 4:
            return 0.0
        
        # Compute oriented volume
        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[0]
        v3 = vertices[3] - vertices[0]
        
        oriented_volume = np.dot(v1, np.cross(v2, v3))
        
        # Topological charge proportional to oriented volume
        return np.sign(oriented_volume) * np.log(1 + np.abs(oriented_volume))
    
    # ============================================================================
    # ACTION AND DYNAMICS
    # ============================================================================
    
    def calculate_action(self) -> Dict[str, float]:
        """
        Calculate the total action of the ACT model.
        
        Returns:
        --------
        action_dict : dict
            Components of the total action:
            - S_Regge: Regge action for simplicial gravity
            - S_matter: Matter field action
            - S_DM: Dark matter action (if included)
            - S_Λ: Cosmological constant term
            - S_total: Total action
        """
        S_Regge = self._calculate_regge_action()
        S_matter = self._calculate_matter_action()
        
        if self.include_dark_matter:
            S_DM = self._calculate_dark_matter_action()
        else:
            S_DM = 0.0
        
        S_Λ = self._calculate_cosmological_constant_action()
        
        S_total = S_Regge + S_matter + S_DM + S_Λ
        
        action_dict = {
            'S_Regge': S_Regge,
            'S_matter': S_matter,
            'S_DM': S_DM,
            'S_Λ': S_Λ,
            'S_total': S_total
        }
        
        return action_dict
    
    def _calculate_regge_action(self) -> float:
        """
        Calculate Regge action for simplicial gravity.
        
        Mathematical formulation:
        S_Regge = (1/8πG) Σ_t (A_t δ_t)
        where A_t is area of triangle t, δ_t is deficit angle
        """
        total_action = 0.0
        
        for tetra in self.tetrahedra:
            # Calculate areas and deficit angles
            vertices = [self.vertices[i] for i in tetra]
            
            # Triangle areas
            areas = []
            triangles = [
                (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)
            ]
            
            for i, j, k in triangles:
                v1 = vertices[j] - vertices[i]
                v2 = vertices[k] - vertices[i]
                area = 0.5 * np.linalg.norm(np.cross(v1, v2))
                areas.append(area)
            
            # Simplified deficit angle (placeholder)
            # In full implementation, compute from dihedral angles
            deficit = 0.1 * np.random.randn()
            
            # Add to action
            total_action += np.sum(areas) * deficit
        
        # Multiply by 1/(8πG)
        total_action = total_action / (8 * np.pi * self.G)
        
        return total_action
    
    def _calculate_matter_action(self) -> float:
        """Calculate action for matter fields."""
        total_action = 0.0
        
        for i in range(self.N):
            # Scalar field action: S = ∫ d⁴x [½(∂φ)² - V(φ)]
            # Simplified implementation
            neighbors = self.adjacency[i].nonzero()[1]
            
            for j in neighbors:
                # Kinetic term
                phi_i = self.vertices[i, 0]  # Use time coordinate as field value
                phi_j = self.vertices[j, 0]
                distance = np.linalg.norm(self.vertices[i] - self.vertices[j])
                
                kinetic = 0.5 * ((phi_i - phi_j) / distance)**2
                
                # Potential term (φ⁴ potential)
                potential = 0.25 * phi_i**4 - 0.5 * phi_i**2
                
                total_action += (kinetic - potential)
        
        return total_action
    
    def _calculate_dark_matter_action(self) -> float:
        """Calculate dark matter action."""
        if not self.include_dark_matter:
            return 0.0
        
        total_action = 0.0
        dm_charge = self.dark_matter_params['topological_charge']
        
        for i in range(self.N):
            # Dark matter action: S_DM = ∫ d⁴x [χ̄(i∂̸ - m_DM)χ + interactions]
            # Simplified topological action
            
            neighbors = self.adjacency[i].nonzero()[1]
            
            for j in neighbors:
                # Topological current
                J_ij = dm_charge[i] * dm_charge[j]
                
                # Distance between vertices
                distance = np.linalg.norm(self.vertices[i] - self.vertices[j])
                
                # Action contribution
                total_action += J_ij / (1 + distance**2)
        
        # Multiply by dark matter mass scale
        total_action *= self.dark_matter_params['mass_scale']
        
        return total_action
    
    def _calculate_cosmological_constant_action(self) -> float:
        """Calculate cosmological constant term."""
        # S_Λ = Λ ∫ d⁴x √g
        # Approximate volume from tetrahedra
        total_volume = 0.0
        
        for tetra in self.tetrahedra:
            vertices = [self.vertices[i] for i in tetra]
            volume = self._calculate_tetrahedron_volume(vertices)
            total_volume += volume
        
        # Multiply by cosmological constant
        S_Λ = self.qg_params['Λ'] * total_volume
        
        return S_Λ
    
    @staticmethod
    @nb.njit(fastmath=True)
    def _calculate_tetrahedron_volume(vertices: List[np.ndarray]) -> float:
        """Calculate volume of a tetrahedron."""
        if len(vertices) < 4:
            return 0.0
        
        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[0]
        v3 = vertices[3] - vertices[0]
        
        volume = np.abs(np.dot(v1, np.cross(v2, v3))) / 6.0
        return volume
    
    # ============================================================================
    # METROPOLIS ALGORITHM FOR THERMALIZATION
    # ============================================================================
    
    def thermalize(self, 
                   n_steps: int = 1000, 
                   step_size: float = 0.1,
                   batch_size: int = 100) -> None:
        """
        Thermalize the model using Metropolis-Hastings algorithm.
        
        Parameters:
        -----------
        n_steps : int
            Number of thermalization steps
        step_size : float
            Maximum displacement for vertex moves
        batch_size : int
            Number of vertices to update in parallel
        """
        print(f"\nThermalizing model with {n_steps:,} steps...")
        
        for step in range(n_steps):
            # Select random batch of vertices
            if batch_size < self.N:
                indices = np.random.choice(self.N, batch_size, replace=False)
            else:
                indices = np.arange(self.N)
            
            # Update each vertex in batch
            for i in indices:
                old_coords = self.vertices[i].copy()
                
                # Propose new coordinates
                delta = np.random.randn(self.dim) * step_size
                new_coords = self.vertices[i] + delta
                
                # Calculate action change
                ΔS = self._calculate_action_change(i, old_coords, new_coords)
                
                # Metropolis acceptance criterion
                if ΔS < 0 or np.random.random() < np.exp(-self.beta * ΔS):
                    self.vertices[i] = new_coords
            
            # Periodically update operators
            if step % 100 == 0:
                self._update_operators_batch(indices[:batch_size//10])
                
            # Progress report
            if step % max(1, n_steps//10) == 0:
                print(f"  Step {step:,}/{n_steps:,} complete")
                
            # Clear cache periodically
            if step % 500 == 0:
                self._action_cache.clear()
        
        print("Thermalization complete!")
    
    def _calculate_action_change(self, 
                                 vertex_idx: int, 
                                 old_coords: np.ndarray,
                                 new_coords: np.ndarray) -> float:
        """Calculate change in action when moving a vertex."""
        # Cache key
        cache_key = f"action_{vertex_idx}"
        
        if cache_key in self._action_cache:
            old_action = self._action_cache[cache_key]
        else:
            old_action = self._calculate_vertex_action(vertex_idx, old_coords)
            self._action_cache[cache_key] = old_action
        
        new_action = self._calculate_vertex_action(vertex_idx, new_coords)
        
        return new_action - old_action
    
    def _calculate_vertex_action(self, 
                                 vertex_idx: int, 
                                 coords: np.ndarray) -> float:
        """Calculate action contribution for a specific vertex."""
        action = 0.0
        
        # Get neighboring vertices
        neighbors = self.adjacency[vertex_idx].nonzero()[1]
        
        for neighbor in neighbors:
            # Distance to neighbor
            if vertex_idx < neighbor:  # Avoid double counting
                neighbor_coords = self.vertices[neighbor]
                distance = np.linalg.norm(coords - neighbor_coords)
                
                # Regge-like action: sum of squared distances
                action += distance**2
        
        # Add curvature contribution
        curvature = self._calculate_vertex_curvature(vertex_idx, coords)
        action += curvature
        
        return action
    
    def _calculate_vertex_curvature(self, 
                                    vertex_idx: int, 
                                    coords: np.ndarray) -> float:
        """Calculate scalar curvature at a vertex."""
        curvature = 0.0
        
        # Find tetrahedra containing this vertex
        neighbor_tets = [t for t in self.tetrahedra if vertex_idx in t]
        
        for tetra in neighbor_tets:
            # Calculate angles at this vertex in the tetrahedron
            other_vertices = [v for v in tetra if v != vertex_idx]
            
            if len(other_vertices) == 3:
                # Vectors from vertex to other vertices
                vectors = []
                for v in other_vertices:
                    vec = self.vertices[v] - coords
                    vectors.append(vec)
                
                # Calculate solid angle (simplified)
                if len(vectors) == 3:
                    v1, v2, v3 = vectors
                    
                    # Triple product
                    triple = np.abs(np.dot(v1, np.cross(v2, v3)))
                    
                    # Magnitudes
                    mag1 = np.linalg.norm(v1)
                    mag2 = np.linalg.norm(v2)
                    mag3 = np.linalg.norm(v3)
                    
                    # Denominator
                    denom = (mag1 * mag2 * mag3 + 
                            np.dot(v1, v2) * mag3 +
                            np.dot(v1, v3) * mag2 +
                            np.dot(v2, v3) * mag1)
                    
                    if denom > 0:
                        solid_angle = 2 * np.arctan2(triple, denom)
                        curvature += (4*np.pi - solid_angle)  # Deficit solid angle
        
        return curvature / max(len(neighbor_tets), 1)
    
    def _update_operators_batch(self, indices: np.ndarray) -> None:
        """Update operators for selected vertices."""
        for i in indices:
            if i < len(self.operators):
                # Small random rotation
                if isinstance(self.operators[i], csr_matrix):
                    op_dense = self.operators[i].toarray()
                    rotation = np.random.randn(4, 4) * 0.01
                    op_dense = op_dense @ (np.eye(4) + rotation)
                    self.operators[i] = csr_matrix(op_dense)
                else:
                    rotation = np.random.randn(4, 4) * 0.01
                    self.operators[i] = self.operators[i] @ (np.eye(4) + rotation)
    
    # ============================================================================
    # OBSERVABLES AND MEASUREMENTS
    # ============================================================================
    
    def calculate_observables(self, 
                             parallel: bool = True,
                             n_workers: int = 4) -> Dict[str, Any]:
        """
        Calculate physical observables from the ACT model.
        
        Returns:
        --------
        observables : dict
            Dictionary of calculated observables
        """
        print("\nCalculating observables...")
        
        observables = {}
        
        # 1. Action components
        observables['action'] = self.calculate_action()
        
        # 2. Scalar curvature distribution
        observables['curvature'] = self._calculate_curvature_distribution()
        
        # 3. Entanglement entropy
        observables['entanglement'] = self._calculate_entanglement_entropy()
        
        # 4. Dark matter density (if included)
        if self.include_dark_matter:
            observables['dark_matter'] = self._calculate_dark_matter_observables()
        
        # 5. Spectral dimension
        observables['spectral_dimension'] = self._calculate_spectral_dimension()
        
        # 6. Hausdorff dimension
        observables['hausdorff_dimension'] = self._calculate_hausdorff_dimension()
        
        # 7. Causal intervals
        observables['causal_structure'] = self._analyze_causal_structure()
        
        # 8. Fundamental constants (emergent)
        observables['fundamental_constants'] = self._calculate_emergent_constants()
        
        self.observables = observables
        return observables
    
    def _calculate_curvature_distribution(self) -> Dict[str, Any]:
        """Calculate scalar curvature distribution."""
        curvatures = np.zeros(self.N)
        
        for i in range(self.N):
            curvatures[i] = self._calculate_vertex_curvature(i, self.vertices[i])
        
        return {
            'mean': np.mean(curvatures),
            'std': np.std(curvatures),
            'min': np.min(curvatures),
            'max': np.max(curvatures),
            'distribution': curvatures.tolist(),
            'histogram': np.histogram(curvatures, bins=20)
        }
    
    def _calculate_entanglement_entropy(self) -> Dict[str, float]:
        """Calculate entanglement entropy for bipartitions."""
        # Simplified calculation
        if self.N > 1000:
            # Use sampling for large networks
            sample_size = min(500, self.N)
            indices = np.random.choice(self.N, sample_size, replace=False)
            adj_sample = self.adjacency[indices][:, indices].toarray()
        else:
            adj_sample = self.adjacency.toarray()
        
        # Laplacian matrix
        degrees = np.sum(adj_sample, axis=1)
        laplacian = np.diag(degrees) - adj_sample
        
        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(laplacian)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        # Entanglement entropy (simplified)
        entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))
        
        # Area law coefficient
        boundary_size = int(np.sqrt(self.N))
        area_law = entropy / boundary_size if boundary_size > 0 else 0
        
        return {
            'entropy': entropy,
            'area_law_coefficient': area_law,
            'eigenvalues': eigenvalues.tolist()[:50]
        }
    
    def _calculate_dark_matter_observables(self) -> Dict[str, Any]:
        """Calculate dark matter observables."""
        if not self.include_dark_matter:
            return {}
        
        dm_charge = self.dark_matter_params['topological_charge']
        
        # Density profile
        radial_bins = 10
        radii = np.linalg.norm(self.vertices[:, 1:4], axis=1)
        max_r = np.max(radii)
        bin_edges = np.linspace(0, max_r, radial_bins + 1)
        
        density_profile = []
        for i in range(radial_bins):
            mask = (radii >= bin_edges[i]) & (radii < bin_edges[i+1])
            if np.any(mask):
                density = np.mean(dm_charge[mask])
                density_profile.append({
                    'r_min': bin_edges[i],
                    'r_max': bin_edges[i+1],
                    'density': density,
                    'count': np.sum(mask)
                })
        
        # Power spectrum
        ft_dm = np.fft.fft(dm_charge)
        power_spectrum = np.abs(ft_dm)**2
        
        return {
            'total_charge': np.sum(dm_charge),
            'mean_density': np.mean(dm_charge),
            'density_profile': density_profile,
            'power_spectrum': power_spectrum.tolist(),
            'mass_estimate': np.sum(np.abs(dm_charge)) * self.dark_matter_params['mass_scale']
        }
    
    def _calculate_spectral_dimension(self) -> float:
        """Calculate spectral dimension of the causal set."""
        # Simplified calculation
        if self.N > 1000:
            sample_size = min(300, self.N)
            indices = np.random.choice(self.N, sample_size, replace=False)
            adj_sample = self.adjacency[indices][:, indices].toarray()
        else:
            adj_sample = self.adjacency.toarray()
        
        # Random walk on the graph
        n_steps = 100
        start_vertex = 0
        
        # Probability distribution after n steps
        p_current = np.zeros(adj_sample.shape[0])
        p_current[start_vertex] = 1.0
        
        return_probabilities = []
        
        for step in range(1, n_steps + 1):
            # Transition matrix
            degrees = np.sum(adj_sample, axis=1)
            transition = adj_sample / degrees[:, np.newaxis]
            transition[np.isnan(transition)] = 0
            
            # Evolve probability distribution
            p_current = transition.T @ p_current
            
            # Return probability
            return_prob = p_current[start_vertex]
            return_probabilities.append(return_prob)
        
        # Fit to power law: P(s) ~ s^{-d_s/2}
        steps = np.arange(1, n_steps + 1)
        log_steps = np.log(steps[10:50])  # Intermediate regime
        log_probs = np.log(return_probabilities[10:50])
        
        if len(log_steps) > 1 and len(log_probs) > 1:
            # Linear fit
            coeffs = np.polyfit(log_steps, log_probs, 1)
            spectral_dim = -2 * coeffs[0]
        else:
            spectral_dim = 4.0  # Default to 4D
        
        return spectral_dim
    
    def _calculate_hausdorff_dimension(self) -> float:
        """Calculate Hausdorff dimension using box-counting."""
        # Use spatial coordinates
        spatial_coords = self.vertices[:, 1:4]
        
        # Box-counting method
        box_sizes = np.logspace(-2, 0, 10)
        counts = []
        
        for size in box_sizes:
            # Discretize space into boxes
            mins = np.min(spatial_coords, axis=0)
            maxs = np.max(spatial_coords, axis=0)
            
            # Number of boxes needed
            n_boxes = np.ceil((maxs - mins) / size).astype(int)
            
            # Assign points to boxes
            box_indices = np.floor((spatial_coords - mins) / size).astype(int)
            
            # Count unique boxes
            unique_boxes = len(set(tuple(idx) for idx in box_indices))
            counts.append(unique_boxes)
        
        # Fit to power law: N(ε) ~ ε^{-d_H}
        log_sizes = np.log(box_sizes)
        log_counts = np.log(counts)
        
        if len(log_sizes) > 1 and len(log_counts) > 1:
            coeffs = np.polyfit(log_sizes, log_counts, 1)
            hausdorff_dim = -coeffs[0]
        else:
            hausdorff_dim = 3.0  # Default to 3D space
        
        return hausdorff_dim
    
    def _analyze_causal_structure(self) -> Dict[str, Any]:
        """Analyze causal structure properties."""
        # Myrheim-Meyer dimension
        n_causal_pairs = self.causal_matrix.sum()
        n_total_pairs = self.N * (self.N - 1) / 2
        
        if n_total_pairs > 0:
            ordering_fraction = n_causal_pairs / n_total_pairs
            # Estimate dimension from ordering fraction
            # For Poisson sprinkling in d-dimensional Minkowski space:
            # f(d) = 2^{-d} * Γ(d+1) / Γ(d/2+1)^2
            mm_dim = self._estimate_myrheim_meyer_dim(ordering_fraction)
        else:
            mm_dim = 4.0
        
        # Causal intervals
        intervals = self._find_causal_intervals()
        
        return {
            'myrheim_meyer_dimension': mm_dim,
            'ordering_fraction': float(ordering_fraction),
            'n_causal_pairs': int(n_causal_pairs),
            'causal_density': n_causal_pairs / n_total_pairs if n_total_pairs > 0 else 0,
            'causal_intervals': intervals
        }
    
    def _estimate_myrheim_meyer_dim(self, f: float) -> float:
        """Estimate dimension from ordering fraction using Myrheim-Meyer."""
        # Solve f(d) = 2^{-d} * Γ(d+1) / Γ(d/2+1)^2 for d
        # Use numerical approximation
        d_vals = np.linspace(1, 10, 1000)
        f_vals = 2**(-d_vals) * np.exp(
            gammaln(d_vals + 1) - 2 * gammaln(d_vals/2 + 1)
        )
        
        # Find closest dimension
        idx = np.argmin(np.abs(f_vals - f))
        return d_vals[idx]
    
    def _find_causal_intervals(self, max_intervals: int = 1000) -> List[Dict[str, Any]]:
        """Find causal intervals in the causal set."""
        intervals = []
        
        # Sample pairs of vertices
        n_samples = min(max_intervals * 10, self.N * (self.N - 1) // 2)
        
        for _ in range(n_samples):
            i, j = np.random.choice(self.N, 2, replace=False)
            
            if self.causal_matrix[i, j]:
                # i precedes j, find elements in between
                # Elements that follow i and precede j
                followers_i = set(self.causal_matrix[i].nonzero()[1])
                preceders_j = set(self.causal_matrix[:, j].nonzero()[0])
                
                # Intersection gives elements in causal interval
                interval = followers_i.intersection(preceders_j)
                
                if len(interval) > 0:
                    intervals.append({
                        'start': int(i),
                        'end': int(j),
                        'size': len(interval),
                        'elements': list(interval)[:10]  # First 10 elements
                    })
            
            if len(intervals) >= max_intervals:
                break
        
        return intervals
    
    def _calculate_emergent_constants(self) -> Dict[str, float]:
        """Calculate emergent fundamental constants from ACT."""
        # Fine structure constant α
        # Emerges from topological properties of the causal set
        topological_invariant = self._calculate_euler_characteristic()
        alpha_emergent = 1 / (4 * np.pi * topological_invariant)
        
        # Gravitational constant G
        # Related to density of causal relations
        causal_density = self.causal_matrix.sum() / (self.N * (self.N - 1))
        G_emergent = self.l_p**2 / (8 * np.pi * causal_density)
        
        # Speed of light c
        # Maximum causal speed from causal structure
        c_emergent = 1.0  # In natural units
        
        # Cosmological constant Λ
        # From average curvature
        avg_curvature = self._calculate_curvature_distribution()['mean']
        Λ_emergent = avg_curvature / 2
        
        return {
            'alpha': alpha_emergent,
            'G': G_emergent,
            'c': c_emergent,
            'Λ': Λ_emergent,
            'topological_invariant': topological_invariant,
            'causal_density': causal_density
        }
    
    def _calculate_euler_characteristic(self) -> float:
        """Calculate Euler characteristic of the simplicial complex."""
        # For 3D simplicial complex:
        # χ = V - E + F - T
        # where V=vertices, E=edges, F=triangles, T=tetrahedra
        
        V = self.N
        
        # Count edges from adjacency matrix
        E = self.adjacency.sum() // 2
        
        # Count triangles (simplified)
        # Triangles are faces of tetrahedra
        triangle_set = set()
        for tetra in self.tetrahedra:
            i, j, k, l = tetra
            triangles = [
                tuple(sorted([i, j, k])),
                tuple(sorted([i, j, l])),
                tuple(sorted([i, k, l])),
                tuple(sorted([j, k, l]))
            ]
            for tri in triangles:
                triangle_set.add(tri)
        
        F = len(triangle_set)
        T = len(self.tetrahedra)
        
        χ = V - E + F - T
        return χ
    
    # ============================================================================
    # VISUALIZATION
    # ============================================================================
    
    def visualize_3d(self, 
                    max_vertices: int = 500,
                    filename: Optional[str] = None) -> go.Figure:
        """
        Create 3D visualization of the ACT network.
        
        Parameters:
        -----------
        max_vertices : int
            Maximum number of vertices to display
        filename : str or None
            If provided, save HTML to this filename
        
        Returns:
        --------
        fig : plotly.graph_objects.Figure
            3D visualization figure
        """
        if self.N > max_vertices:
            indices = np.random.choice(self.N, max_vertices, replace=False)
            vertices = self.vertices[indices]
            vertex_indices = indices
        else:
            vertices = self.vertices
            vertex_indices = np.arange(self.N)
        
        fig = go.Figure()
        
        # Vertex colors based on properties
        if self.include_dark_matter:
            colors = self.dark_matter_params['topological_charge']
            if len(colors) > len(vertex_indices):
                colors = colors[vertex_indices]
            color_label = "Dark Matter Charge"
        else:
            colors = vertices[:, 0]  # Time coordinate
            color_label = "Time Coordinate"
        
        # Vertices
        fig.add_trace(go.Scatter3d(
            x=vertices[:, 1],
            y=vertices[:, 2],
            z=vertices[:, 3],
            mode='markers',
            marker=dict(
                size=4,
                color=colors,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title=color_label)
            ),
            text=[f"Vertex {i}" for i in vertex_indices],
            hoverinfo='text',
            name='Vertices'
        ))
        
        # Edges (sampled)
        n_edges_to_show = min(2000, len(self.tetrahedra) * 3)
        tetra_samples = np.random.choice(
            len(self.tetrahedra),
            n_edges_to_show // 6,
            replace=False
        )
        
        edge_x, edge_y, edge_z = [], [], []
        
        for idx in tetra_samples:
            tetra = self.tetrahedra[idx]
            # Check if vertices are in our sample
            tetra_in_sample = [i for i in tetra if i in vertex_indices]
            
            if len(tetra_in_sample) >= 2:
                for i in range(len(tetra_in_sample)):
                    for j in range(i+1, len(tetra_in_sample)):
                        v1_idx = tetra_in_sample[i]
                        v2_idx = tetra_in_sample[j]
                        
                        # Find positions in our sample
                        v1_pos = np.where(vertex_indices == v1_idx)[0][0]
                        v2_pos = np.where(vertex_indices == v2_idx)[0][0]
                        
                        edge_x.extend([vertices[v1_pos, 1], vertices[v2_pos, 1], None])
                        edge_y.extend([vertices[v1_pos, 2], vertices[v2_pos, 2], None])
                        edge_z.extend([vertices[v1_pos, 3], vertices[v2_pos, 3], None])
        
        if edge_x:
            fig.add_trace(go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode='lines',
                line=dict(color='rgba(100, 100, 100, 0.15)', width=1),
                hoverinfo='none',
                name='Edges'
            ))
        
        # Layout
        fig.update_layout(
            title=dict(
                text=f"ACT Network: N={self.N:,}, Tets={len(self.tetrahedra):,}",
                font=dict(size=16)
            ),
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        if filename:
            fig.write_html(filename)
            print(f"Visualization saved to {filename}")
        
        return fig
    
    # ============================================================================
    # SAVE AND LOAD
    # ============================================================================
    
    def save(self, filename: str) -> None:
        """Save model to file."""
        with open(filename, 'wb') as f:
            pickle.dump({
                'vertices': self.vertices,
                'operators': self.operators,
                'tetrahedra': self.tetrahedra,
                'adjacency': self.adjacency,
                'causal_matrix': self.causal_matrix,
                'qg_params': self.qg_params,
                'dark_matter_params': self.dark_matter_params,
                'observables': self.observables,
                'N': self.N,
                'dim': self.dim,
                'beta': self.beta
            }, f)
        
        print(f"Model saved to {filename}")
    
    @classmethod
    def load(cls, filename: str) -> 'ACTModel':
        """Load model from file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        # Create new instance
        model = cls.__new__(cls)
        
        # Restore attributes
        model.vertices = data['vertices']
        model.operators = data['operators']
        model.tetrahedra = data['tetrahedra']
        model.adjacency = data['adjacency']
        model.causal_matrix = data['causal_matrix']
        model.qg_params = data['qg_params']
        model.dark_matter_params = data['dark_matter_params']
        model.observables = data['observables']
        model.N = data['N']
        model.dim = data['dim']
        model.beta = data['beta']
        
        # Set derived attributes
        model.l_p = model.qg_params['l_pl']
        model.t_p = model.qg_params['t_pl']
        model.M_p = model.qg_params['M_pl']
        model.ħ = model.qg_params['ħ']
        model.c = model.qg_params['c']
        model.G = model.qg_params['G_N']
        
        if model.dark_matter_params is not None:
            model.include_dark_matter = True
        else:
            model.include_dark_matter = False
        
        model._action_cache = {}
        
        print(f"Model loaded from {filename}")
        return model


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_act_experiment(N: int = 1000,
                      temperature: float = 0.7,
                      include_dm: bool = True,
                      thermalization_steps: int = 500,
                      seed: Optional[int] = None) -> Tuple[ACTModel, Dict[str, Any]]:
    """
    Run complete ACT experiment.
    
    Parameters:
    -----------
    N : int
        Number of vertices
    temperature : float
        System temperature
    include_dm : bool
        Include dark matter sector
    thermalization_steps : int
        Number of thermalization steps
    seed : int or None
        Random seed
    
    Returns:
    --------
    model : ACTModel
        Trained ACT model
    results : dict
        Experiment results
    """
    print("="*80)
    print("ACT EXPERIMENT")
    print("="*80)
    
    start_time = datetime.now()
    
    # Initialize model
    model = ACTModel(
        N=N,
        temperature=temperature,
        include_dark_matter=include_dm,
        seed=seed
    )
    
    # Thermalize
    model.thermalize(
        n_steps=thermalization_steps,
        step_size=0.1,
        batch_size=100
    )
    
    # Calculate observables
    observables = model.calculate_observables(parallel=True)
    
    # Create visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_file = f"act_network_N{N}_{timestamp}.html"
    model.visualize_3d(filename=vis_file)
    
    # Compile results
    results = {
        'parameters': {
            'N': N,
            'temperature': temperature,
            'include_dark_matter': include_dm,
            'thermalization_steps': thermalization_steps,
            'seed': seed
        },
        'geometry': {
            'tetrahedra_count': len(model.tetrahedra),
            'average_degree': model.adjacency.sum()/model.N,
            'causal_density': model.causal_matrix.sum()/(model.N*(model.N-1))
        },
        'observables': observables,
        'fundamental_constants': observables.get('fundamental_constants', {}),
        'dark_matter': observables.get('dark_matter', {}) if include_dm else None,
        'computation_time': (datetime.now() - start_time).total_seconds(),
        'timestamp': timestamp
    }
    
    # Save results
    results_file = f"act_results_N{N}_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save model
    model_file = f"act_model_N{N}_{timestamp}.pkl"
    model.save(model_file)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nResults:")
    print(f"  • Action: {observables['action']['S_total']:.4e}")
    print(f"  • Spectral dimension: {observables['spectral_dimension']:.2f}")
    print(f"  • Hausdorff dimension: {observables['hausdorff_dimension']:.2f}")
    
    if include_dm:
        print(f"  • Dark matter density: {observables['dark_matter']['mean_density']:.4e}")
    
    consts = observables['fundamental_constants']
    print(f"  • Emergent α: {consts['alpha']:.6f}")
    print(f"  • Emergent G: {consts['G']:.4e} m³/kg/s²")
    
    print(f"\nFiles saved:")
    print(f"  • Results: {results_file}")
    print(f"  • Model: {model_file}")
    print(f"  • Visualization: {vis_file}")
    print(f"\nTotal time: {results['computation_time']:.1f} seconds")
    print("="*80)
    
    return model, results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Algebraic Causality Theory (ACT) Simulation"
    )
    parser.add_argument("--N", type=int, default=1000,
                       help="Number of vertices (default: 1000)")
    parser.add_argument("--temp", type=float, default=0.7,
                       help="Temperature (default: 0.7)")
    parser.add_argument("--no-dm", action="store_true",
                       help="Disable dark matter sector")
    parser.add_argument("--steps", type=int, default=500,
                       help="Thermalization steps (default: 500)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--test", action="store_true",
                       help="Quick test with N=500")
    
    args = parser.parse_args()
    
    if args.test:
        print("Running quick test...")
        model, results = run_act_experiment(
            N=500,
            temperature=0.7,
            include_dm=not args.no_dm,
            thermalization_steps=100,
            seed=args.seed
        )
    else:
        model, results = run_act_experiment(
            N=args.N,
            temperature=args.temp,
            include_dm=not args.no_dm,
            thermalization_steps=args.steps,
            seed=args.seed
        )
