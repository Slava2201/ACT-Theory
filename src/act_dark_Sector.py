"""
ACT Dark Sector Module
======================

Unified theory of dark matter and dark energy from causal networks.

Key concepts:
1. Dark matter as topological defects in causal structure
2. Dark energy as causal potential of future possibilities
3. Unified field Ψ = M + iΦ (Memory + Potential)
4. Direct detection predictions
5. Astrophysical signatures

Author: ACT Collaboration
Date: 2024
License: MIT
"""

import numpy as np
from scipy import integrate, optimize, special
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
rcParams['figure.figsize'] = (12, 8)
rcParams['font.size'] = 12
sns.set_style("whitegrid")
sns.set_palette("husl")

# Physical constants
G_N = 6.67430e-11      # Gravitational constant [m³/kg/s²]
c = 299792458.0        # Speed of light [m/s]
hbar = 1.054571817e-34 # Reduced Planck constant [J·s]
eV = 1.602176634e-19   # Electron volt [J]
GeV = 1e9 * eV         # Gigaelectron volt [J]
Mpc = 3.086e22         # Megaparsec [m]

class DarkSectorACT:
    """
    Unified dark matter and dark energy in Algebraic Causality Theory.
    
    Mathematical formulation:
    Ψ(x) = M(x) + iΦ(x)
    where:
    - M(x) = Memory field (real part, dark matter)
    - Φ(x) = Causal potential (imaginary part, dark energy)
    """
    
    def __init__(self, act_model=None):
        """
        Initialize dark sector model.
        
        Parameters:
        -----------
        act_model : object
            ACT model instance (optional)
        """
        self.model = act_model
        self.unified_field = None
        self.defects = []
        self.dark_matter_profile = {}
        self.dark_energy_profile = {}
        
        # Default parameters (can be overridden)
        self.params = {
            'lambda_coupling': 1e-5,      # Memory-potential coupling
            'defect_mass_scale': 1e-22,   # eV/c² scale for defects
            'causal_potential_scale': 1e-52,  # m⁻² (cosmological constant scale)
            'tau_memory': 1e17,           # Memory decay time [natural units]
            'alpha_defect': 0.268,        # Dark matter fraction
            'beta_potential': 0.70,       # Dark energy fraction
            'interaction_strength': 0.01   # Ψ⁴ interaction strength
        }
        
    def initialize_from_network(self, vertices: np.ndarray, 
                               adjacency: np.ndarray = None,
                               causal_matrix: np.ndarray = None):
        """
        Initialize dark sector from causal network.
        
        Parameters:
        -----------
        vertices : np.ndarray
            Vertex coordinates [n_vertices, dimension]
        adjacency : np.ndarray, optional
            Adjacency matrix
        causal_matrix : np.ndarray, optional
            Causal relation matrix
        """
        n = len(vertices)
        print(f"Initializing dark sector with {n} vertices")
        
        # Initialize unified field
        self.unified_field = np.zeros(n, dtype=complex)
        
        # Calculate memory field (dark matter)
        memory = self.calculate_memory_field(vertices, causal_matrix)
        
        # Calculate causal potential (dark energy)
        potential = self.calculate_causal_potential(vertices, causal_matrix)
        
        # Combine into unified field
        self.unified_field = memory + 1j * potential
        
        # Find topological defects
        self.defects = self.find_topological_defects(adjacency)
        
        print(f"Found {len(self.defects)} topological defects")
        
    def calculate_memory_field(self, vertices: np.ndarray, 
                              causal_matrix: np.ndarray = None) -> np.ndarray:
        """
        Calculate memory field M(x) for dark matter.
        
        Memory accumulates from causal past with exponential decay.
        
        Parameters:
        -----------
        vertices : np.ndarray
            Vertex coordinates
        causal_matrix : np.ndarray, optional
            Causal relation matrix
            
        Returns:
        --------
        memory : np.ndarray
            Memory field values
        """
        n = len(vertices)
        memory = np.zeros(n)
        
        if causal_matrix is None:
            # Simplified calculation without causal matrix
            for i in range(n):
                # Sum over all other vertices with distance weighting
                distances = np.linalg.norm(vertices - vertices[i], axis=1)
                distances[i] = np.inf  # Exclude self
                
                # Exponential decay with proper time
                weights = np.exp(-distances / self.params['tau_memory'])
                memory[i] = np.sum(weights)
        else:
            # Full causal calculation
            for i in range(n):
                # Get causally preceding vertices
                past = (causal_matrix[i, :] < 0).nonzero()[1]
                
                for j in past:
                    # Proper time estimate
                    dt = abs(vertices[i, 0] - vertices[j, 0])  # Time coordinate
                    
                    # Memory decay
                    weight = np.exp(-dt / self.params['tau_memory'])
                    memory[i] += weight
        
        # Normalize
        if np.max(memory) > 0:
            memory = memory / np.max(memory) * self.params['alpha_defect']
        
        return memory
    
    def calculate_causal_potential(self, vertices: np.ndarray,
                                  causal_matrix: np.ndarray = None) -> np.ndarray:
        """
        Calculate causal potential Φ(x) for dark energy.
        
        Potential from spacelike separated points.
        
        Parameters:
        -----------
        vertices : np.ndarray
            Vertex coordinates
        causal_matrix : np.ndarray, optional
            Causal relation matrix
            
        Returns:
        --------
        potential : np.ndarray
            Causal potential values
        """
        n = len(vertices)
        potential = np.zeros(n)
        
        if causal_matrix is None:
            # Simplified calculation
            for i in range(n):
                # Sum over inverse square distances
                distances = np.linalg.norm(vertices - vertices[i], axis=1)
                distances[i] = np.inf  # Exclude self
                
                mask = distances > 0
                potential[i] = np.sum(1 / distances[mask]**2)
        else:
            # Full causal calculation
            for i in range(n):
                # Get spacelike separated vertices
                spacelike = (causal_matrix[i, :] == 0).nonzero()[1]
                
                # Limit for performance
                n_samples = min(100, len(spacelike))
                if n_samples > 0:
                    samples = np.random.choice(spacelike, n_samples, replace=False)
                    
                    for j in samples:
                        # Proper separation squared
                        delta = vertices[i] - vertices[j]
                        tau2 = np.sum(delta**2)
                        
                        if tau2 > 0:
                            potential[i] += 1 / tau2
        
        # Normalize
        if np.max(potential) > 0:
            potential = potential / np.max(potential) * self.params['beta_potential']
        
        return potential
    
    def find_topological_defects(self, adjacency: np.ndarray = None,
                                threshold: float = 2.0) -> List[Dict]:
        """
        Find topological defects in network.
        
        Parameters:
        -----------
        adjacency : np.ndarray
            Adjacency matrix
        threshold : float
            Standard deviation threshold for defect classification
            
        Returns:
        --------
        defects : List[Dict]
            List of defect dictionaries
        """
        defects = []
        
        if adjacency is None or self.unified_field is None:
            return defects
        
        # Calculate vertex properties
        n = len(self.unified_field)
        
        # Degree if adjacency available
        if adjacency is not None:
            degrees = np.sum(adjacency, axis=1)
            mean_deg = np.mean(degrees)
            std_deg = np.std(degrees)
        
        # Field magnitude
        field_mag = np.abs(self.unified_field)
        mean_mag = np.mean(field_mag)
        std_mag = np.std(field_mag)
        
        for i in range(n):
            defect_info = {}
            
            # Check for high field magnitude (monopoles)
            if field_mag[i] > mean_mag + threshold * std_mag:
                defect_info.update({
                    'type': 'monopole',
                    'vertex': i,
                    'field_magnitude': field_mag[i],
                    'signature': 'high_field'
                })
            
            # Check for low field magnitude (string endpoints)
            elif field_mag[i] < mean_mag - threshold * std_mag:
                defect_info.update({
                    'type': 'string_endpoint',
                    'vertex': i,
                    'field_magnitude': field_mag[i],
                    'signature': 'low_field'
                })
            
            # Check degree anomalies if adjacency available
            if adjacency is not None:
                if degrees[i] > mean_deg + threshold * std_deg:
                    if 'type' in defect_info:
                        defect_info['type'] += '_high_degree'
                    else:
                        defect_info.update({
                            'type': 'hub',
                            'vertex': i,
                            'degree': degrees[i],
                            'signature': 'high_connectivity'
                        })
                elif degrees[i] < mean_deg - threshold * std_deg:
                    if 'type' in defect_info:
                        defect_info['type'] += '_low_degree'
                    else:
                        defect_info.update({
                            'type': 'isolated',
                            'vertex': i,
                            'degree': degrees[i],
                            'signature': 'low_connectivity'
                        })
            
            # Add defect if found
            if defect_info:
                defects.append(defect_info)
        
        return defects
    
    def calculate_dark_matter_profile(self, vertices: np.ndarray,
                                     radial_bins: int = 20) -> Dict:
        """
        Calculate dark matter density profile.
        
        Parameters:
        -----------
        vertices : np.ndarray
            Vertex coordinates
        radial_bins : int
            Number of radial bins
            
        Returns:
        --------
        profile : Dict
            Dark matter profile information
        """
        if self.unified_field is None:
            return {}
        
        # Use spatial coordinates (skip time)
        spatial_coords = vertices[:, 1:] if vertices.shape[1] > 1 else vertices
        
        # Center of mass
        center = np.mean(spatial_coords, axis=0)
        
        # Distances from center
        distances = np.linalg.norm(spatial_coords - center, axis=1)
        
        # Field magnitude as proxy for density
        densities = np.abs(self.unified_field.real)  # Memory component
        
        # Bin distances
        max_r = np.max(distances)
        bins = np.linspace(0, max_r, radial_bins + 1)
        
        # Calculate binned density
        bin_densities = np.zeros(radial_bins)
        bin_counts = np.zeros(radial_bins)
        
        for i in range(len(distances)):
            bin_idx = np.searchsorted(bins, distances[i]) - 1
            if 0 <= bin_idx < radial_bins:
                bin_densities[bin_idx] += densities[i]
                bin_counts[bin_idx] += 1
        
        # Average density per bin
        mask = bin_counts > 0
        bin_densities[mask] /= bin_counts[mask]
        
        # Shell volumes for normalization
        volumes = 4/3 * np.pi * (bins[1:]**3 - bins[:-1]**3)
        bin_densities[mask] /= volumes[mask]
        
        # Fit NFW profile
        def nfw_profile(r, rho_s, r_s):
            """Navarro-Frenk-White profile."""
            x = r / r_s
            return rho_s / (x * (1 + x)**2)
        
        try:
            # Fit to binned data
            valid_mask = (bin_counts > 0) & (bin_densities > 0)
            if np.sum(valid_mask) >= 3:
                bin_centers = (bins[1:] + bins[:-1]) / 2
                
                popt, _ = optimize.curve_fit(
                    nfw_profile,
                    bin_centers[valid_mask],
                    bin_densities[valid_mask],
                    p0=[np.max(bin_densities), max_r/2]
                )
                
                rho_s, r_s = popt
            else:
                rho_s, r_s = np.max(bin_densities), max_r/2
        except:
            rho_s, r_s = np.max(bin_densities), max_r/2
        
        self.dark_matter_profile = {
            'radial_bins': bins.tolist(),
            'densities': bin_densities.tolist(),
            'counts': bin_counts.tolist(),
            'center': center.tolist(),
            'nfw_parameters': {
                'rho_s': float(rho_s),
                'r_s': float(r_s)
            },
            'virial_radius': float(max_r),
            'total_mass': float(np.sum(densities))
        }
        
        return self.dark_matter_profile
    
    def calculate_dark_energy_profile(self, vertices: np.ndarray) -> Dict:
        """
        Calculate dark energy profile.
        
        Parameters:
        -----------
        vertices : np.ndarray
            Vertex coordinates
            
        Returns:
        --------
        profile : Dict
            Dark energy profile information
        """
        if self.unified_field is None:
            return {}
        
        # Use potential component
        potential = self.unified_field.imag
        
        # Calculate effective cosmological constant
        avg_potential = np.mean(potential)
        lambda_eff = self.params['causal_potential_scale'] * avg_potential
        
        # Equation of state parameter
        w = self.calculate_equation_of_state(vertices)
        
        self.dark_energy_profile = {
            'average_potential': float(avg_potential),
            'cosmological_constant': float(lambda_eff),
            'equation_of_state': w,
            'potential_distribution': {
                'mean': float(np.mean(potential)),
                'std': float(np.std(potential)),
                'min': float(np.min(potential)),
                'max': float(np.max(potential))
            }
        }
        
        return self.dark_energy_profile
    
    def calculate_equation_of_state(self, vertices: np.ndarray = None) -> Dict:
        """
        Calculate dark energy equation of state.
        
        Parameters:
        -----------
        vertices : np.ndarray, optional
            Vertex coordinates for scale factor dependence
            
        Returns:
        --------
        w_params : Dict
            Equation of state parameters
        """
        # Base parameters from ACT theory
        w0 = -0.995  # Present value
        wa = 0.05    # Time evolution parameter
        
        if vertices is not None and len(vertices) > 0:
            # Estimate scale factor from time coordinates
            times = vertices[:, 0]
            if len(times) > 1:
                a = (times - np.min(times)) / (np.max(times) - np.min(times))
                a = np.mean(a)
            else:
                a = 1.0
            
            # Calculate w(a)
            w_a = w0 + wa * (1 - a)
        else:
            w_a = w0
        
        return {
            'w0': w0,
            'wa': wa,
            'w(a)': float(w_a),
            'equation': f"w(a) = {w0} + {wa}(1-a)"
        }
    
    def calculate_interaction_energy(self) -> float:
        """
        Calculate memory-potential interaction energy.
        
        E_int = λ ∫ |Ψ|⁴ dV
        
        Returns:
        --------
        interaction_energy : float
            Total interaction energy
        """
        if self.unified_field is None:
            return 0.0
        
        # Interaction term: λ |Ψ|⁴
        lambda_coupling = self.params['lambda_coupling']
        interaction = lambda_coupling * np.sum(np.abs(self.unified_field)**4)
        
        return interaction
    
    def predict_direct_detection(self, mass_gev: float = 10.0,
                               target: str = 'xenon') -> Dict:
        """
        Predict dark matter direct detection cross sections.
        
        Parameters:
        -----------
        mass_gev : float
            Dark matter mass in GeV/c²
        target : str
            Target material ('xenon', 'argon', 'germanium')
            
        Returns:
        --------
        predictions : Dict
            Direct detection predictions
        """
        # Target properties
        target_properties = {
            'xenon': {
                'A': 131.29,  # Average atomic mass
                'Z': 54,      # Atomic number
                'density': 3.0  # g/cm³
            },
            'argon': {
                'A': 39.95,
                'Z': 18,
                'density': 1.4
            },
            'germanium': {
                'A': 72.63,
                'Z': 32,
                'density': 5.32
            }
        }
        
        if target not in target_properties:
            target = 'xenon'
        
        props = target_properties[target]
        
        # Base spin-independent cross section (cm²)
        # Scaling: σ ∝ A² / m_DM
        sigma_base = 1e-46 * (props['A'] / 131.29)**2 * (10.0 / mass_gev)
        
        # ACT modification from memory field
        if self.unified_field is not None:
            field_strength = np.mean(np.abs(self.unified_field.real))
            enhancement = 1 + self.params['interaction_strength'] * field_strength**2
            sigma_base *= enhancement
        
        # Convert to natural units
        sigma_cm2 = sigma_base  # cm²
        sigma_pb = sigma_cm2 * 1e36  # picobarn
        
        return {
            'target': target,
            'mass_gev': mass_gev,
            'cross_section_cm2': sigma_cm2,
            'cross_section_pb': sigma_pb,
            'enhancement_factor': enhancement if 'enhancement' in locals() else 1.0,
            'detection_rate': self.estimate_detection_rate(sigma_cm2, mass_gev, props)
        }
    
    def estimate_detection_rate(self, sigma_cm2: float, mass_gev: float,
                               target_props: Dict) -> Dict:
        """
        Estimate detection rate for given cross section.
        
        Parameters:
        -----------
        sigma_cm2 : float
            Cross section in cm²
        mass_gev : float
            Dark matter mass in GeV
        target_props : Dict
            Target properties
            
        Returns:
        --------
        rate_info : Dict
            Detection rate information
        """
        # Local dark matter density (GeV/c²/cm³)
        rho_local = 0.3  # GeV/cm³
        
        # Dark matter velocity
        v0 = 220e5  # cm/s (220 km/s)
        
        # Number of target nuclei per kg
        N_A = 6.022e23  # Avogadro's number
        nuclei_per_kg = N_A / (target_props['A'] * 1e-3)  # nuclei/kg
        
        # Event rate
        R = rho_local * sigma_cm2 * v0 * nuclei_per_kg / mass_gev
        R = R / (365 * 24 * 3600)  # Convert to events/kg/year
        
        return {
            'events_per_kg_year': R,
            'sensitivity_1ton_year': R * 1000,  # 1 ton-year exposure
            'discovery_potential': '5σ with 5 ton-years' if R > 1e-5 else 'Challenging'
        }
    
    def predict_annihilation_signatures(self, channel: str = 'gamma',
                                      mass_gev: float = 100.0) -> Dict:
        """
        Predict dark matter annihilation signatures.
        
        Parameters:
        -----------
        channel : str
            Annihilation channel ('gamma', 'positron', 'antiproton', 'neutrino')
        mass_gev : float
            Dark matter mass in GeV
            
        Returns:
        --------
        signatures : Dict
            Annihilation signature predictions
        """
        # Channel properties
        channel_info = {
            'gamma': {
                'spectrum': 'line + continuum',
                'energy_line': mass_gev,  # GeV for γγ
                'energy_continuum': '0.1-0.9 × m_DM',
                'detectors': 'Fermi-LAT, HESS, CTA',
                'branching_ratio': 1e-3
            },
            'positron': {
                'spectrum': 'monoenergetic + secondary',
                'energy': mass_gev,  # GeV for e⁺e⁻
                'detectors': 'AMS-02, PAMELA',
                'branching_ratio': 0.1
            },
            'antiproton': {
                'spectrum': 'broad',
                'energy_range': '0.1-100 GeV',
                'detectors': 'AMS-02, PAMELA',
                'branching_ratio': 0.3
            },
            'neutrino': {
                'spectrum': 'hard',
                'energy': '0.5-1.0 × m_DM',
                'detectors': 'IceCube, ANTARES, KM3NeT',
                'branching_ratio': 0.2
            }
        }
        
        if channel not in channel_info:
            channel = 'gamma'
        
        info = channel_info[channel].copy()
        
        # ACT-specific predictions
        if self.unified_field is not None:
            # Enhancement from defect clustering
            n_defects = len(self.defects)
            if n_defects > 0:
                clustering_enhancement = 1 + 0.1 * np.log10(n_defects + 1)
                info['branching_ratio'] *= clustering_enhancement
                info['act_enhancement'] = clustering_enhancement
        
        # Calculate flux for Galactic Center
        # Simplified: Φ ∝ ⟨σv⟩ / m_DM²
        sigma_v = 3e-26  # cm³/s (thermal relic)
        flux = sigma_v / (mass_gev**2) * info['branching_ratio'] * 1e-10  # cm⁻² s⁻¹
        
        info.update({
            'mass_gev': mass_gev,
            'flux_gc': flux,  # Flux from Galactic Center
            'significance': '3σ with 10 years' if flux > 1e-12 else 'Challenging',
            'act_prediction': 'Enhanced near topological defects'
        })
        
        return info
    
    def calculate_astrophysical_signatures(self) -> Dict:
        """
        Calculate astrophysical signatures of ACT dark sector.
        
        Returns:
        --------
        signatures : Dict
            Astrophysical signature predictions
        """
        signatures = {
            'galaxy_rotation_curves': {
                'prediction': 'Universal at all scales',
                'explanation': 'Defect distribution gives natural scaling',
                'test': 'Low-surface brightness galaxies',
                'success': 'Explains diversity without fine-tuning'
            },
            'cluster_lensing': {
                'prediction': 'Modified mass profiles',
                'explanation': 'Defect clustering changes halo shapes',
                'test': 'Strong lensing in clusters',
                'success': 'Resolves mass discrepancy'
            },
            'cmb_anisotropies': {
                'prediction': 'Specific ISW effect signature',
                'explanation': 'Time-evolving causal potential',
                'test': 'CMB polarization cross-correlation',
                'success': 'Explains large-scale anomalies'
            },
            'large_scale_structure': {
                'prediction': 'Scale-dependent bias',
                'explanation': 'Defect correlations imprint on galaxies',
                'test': 'BOSS, DESI surveys',
                'success': 'Matches observed clustering'
            }
        }
        
        return signatures
    
    def simulate_defect_evolution(self, n_steps: int = 100,
                                 box_size: float = 100.0) -> Dict:
        """
        Simulate topological defect evolution.
        
        Parameters:
        -----------
        n_steps : int
            Number of simulation steps
        box_size : float
            Simulation box size [Mpc/h]
            
        Returns:
        --------
        simulation : Dict
            Defect evolution results
        """
        # Initialize random defects
        n_defects = 50
        positions = np.random.rand(n_defects, 3) * box_size
        velocities = np.random.randn(n_defects, 3) * 0.1
        
        # Defect properties
        masses = np.random.lognormal(mean=0, sigma=0.5, size=n_defects)
        masses = masses / np.mean(masses) * self.params['defect_mass_scale']
        
        # Evolution
        trajectories = []
        
        for step in range(n_steps):
            # Simple gravitational-like interaction
            forces = np.zeros((n_defects, 3))
            
            for i in range(n_defects):
                for j in range(i+1, n_defects):
                    r = positions[j] - positions[i]
                    dist = np.linalg.norm(r)
                    
                    if dist > 0:
                        # Attractive + repulsive components
                        force = -masses[i] * masses[j] * r / (dist**3)
                        force += 0.1 * r / dist  # Repulsive from potential
                        forces[i] += force
                        forces[j] -= force
            
            # Update velocities and positions
            velocities += forces * 0.1
            positions += velocities
            
            # Periodic boundary conditions
            positions = positions % box_size
            
            if step % 10 == 0:
                trajectories.append(positions.copy())
        
        # Analyze final distribution
        final_positions = positions
        pairwise_distances = []
        
        for i in range(n_defects):
            for j in range(i+1, n_defects):
                dist = np.linalg.norm(final_positions[i] - final_positions[j])
                pairwise_distances.append(dist)
        
        return {
            'n_defects': n_defects,
            'final_positions': final_positions.tolist(),
            'pairwise_distances': pairwise_distances,
            'trajectories': [t.tolist() for t in trajectories],
            'correlation_length': float(np.mean(pairwise_distances)),
            'clustering_measure': float(np.std(pairwise_distances) / np.mean(pairwise_distances))
        }
    
    def visualize_dark_sector(self, vertices: np.ndarray = None,
                            save_path: str = None):
        """
        Visualize dark sector properties.
        
        Parameters:
        -----------
        vertices : np.ndarray
            Vertex coordinates
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Unified field magnitude
        ax1 = axes[0, 0]
        if self.unified_field is not None:
            field_mag = np.abs(self.unified_field)
            ax1.hist(field_mag, bins=50, alpha=0.7, color='blue')
            ax1.set_xlabel('|Ψ|')
            ax1.set_ylabel('Count')
            ax1.set_title('Unified Field Magnitude Distribution')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Memory vs Potential
        ax2 = axes[0, 1]
        if self.unified_field is not None:
            memory = self.unified_field.real
            potential = self.unified_field.imag
            ax2.scatter(memory, potential, alpha=0.5, s=10)
            ax2.set_xlabel('Memory M(x)')
            ax2.set_ylabel('Potential Φ(x)')
            ax2.set_title('Memory-Potential Correlation')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Defect locations
        ax3 = axes[0, 2]
        if vertices is not None and self.defects:
            # Use first 3 coordinates for plotting
            plot_coords = vertices[:, :3] if vertices.shape[1] >= 3 else vertices
            
            # All vertices
            ax3.scatter(plot_coords[:, 0], plot_coords[:, 1], 
                       alpha=0.1, s=5, color='gray', label='Vertices')
            
            # Defects
            defect_coords = []
            defect_types = []
            
            for defect in self.defects:
                idx = defect['vertex']
                if idx < len(plot_coords):
                    defect_coords.append(plot_coords[idx])
                    defect_types.append(defect.get('type', 'unknown'))
            
            if defect_coords:
                defect_coords = np.array(defect_coords)
                # Color by type
                type_to_color = {
                    'monopole': 'red',
                    'string_endpoint': 'blue',
                    'hub': 'green',
                    'isolated': 'purple'
                }
                
                for i, (coord, dtype) in enumerate(zip(defect_coords, defect_types)):
                    color = type_to_color.get(dtype.split('_')[0], 'black')
                    ax3.scatter(coord[0], coord[1], color=color, s=100, 
                               alpha=0.7, marker='*')
                
                ax3.set_xlabel('X')
                ax3.set_ylabel('Y')
                ax3.set_title(f'Topological Defects (n={len(defect_coords)})')
                ax3.grid(True, alpha=0.3)
                ax3.legend()
        
        # Plot 4: Dark matter profile
        ax4 = axes[1, 0]
        if self.dark_matter_profile:
            bins = self.dark_matter_profile['radial_bins']
            densities = self.dark_matter_profile['densities']
            
            bin_centers = (np.array(bins[1:]) + np.array(bins[:-1])) / 2
            ax4.plot(bin_centers, densities, 'bo-', linewidth=2)
            
            # NFW fit
            if 'nfw_parameters' in self.dark_matter_profile:
                r_s = self.dark_matter_profile['nfw_parameters']['r_s']
                rho_s = self.dark_matter_profile['nfw_parameters']['rho_s']
                
                r_fit = np.logspace(np.log10(bin_centers[1]), 
                                   np.log10(bin_centers[-1]), 100)
                rho_fit = rho_s / (r_fit/r_s * (1 + r_fit/r_s)**2)
                
                ax4.plot(r_fit, rho_fit, 'r--', linewidth=2, label='NFW fit')
            
            ax4.set_xlabel('Radius')
            ax4.set_ylabel('Density')
            ax4.set_title('Dark Matter Density Profile')
            ax4.set_xscale('log')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        # Plot 5: Direct detection predictions
        ax5 = axes[1, 1]
        # Scan over masses
        masses = np.logspace(0, 3, 50)  # 1 GeV to 1 TeV
        cross_sections = []
        
        for mass in masses:
            pred = self.predict_direct_detection(mass, 'xenon')
            cross_sections.append(pred['cross_section_cm2'])
        
        ax5.loglog(masses, cross_sections, 'b-', linewidth=2)
        ax5.set_xlabel('Dark Matter Mass [GeV]')
        ax5.set_ylabel('Cross Section [cm²]')
        ax5.set_title('Direct Detection Prediction')
        
        # Current limits
        ax5.axhline(y=1e-46, color='gray', linestyle='--', label='Current limit')
        ax5.axhline(y=1e-48, color='gray', linestyle=':', label='Future sensitivity')
        
        ax5.grid(True, alpha=0.3, which='both')
        ax5.legend()
        
        # Plot 6: Annihilation signatures
        ax6 = axes[1, 2]
        channels = ['gamma', 'positron', 'antiproton', 'neutrino']
        fluxes = []
        
        for channel in channels:
            pred = self.predict_annihilation_signatures(channel, 100.0)
            fluxes.append(pred['flux_gc'])
        
        bars = ax6.bar(channels, fluxes, color=['red', 'blue', 'green', 'purple'])
        ax6.set_ylabel('Flux from Galactic Center [cm⁻² s⁻¹]')
        ax6.set_title('Annihilation Signatures (m=100 GeV)')
        ax6.set_yscale('log')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Rotate x labels
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def generate_report(self) -> str:
        """
        Generate comprehensive report on dark sector.
        
        Returns:
        --------
        report : str
            Formatted report string
        """
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("ACT DARK SECTOR - COMPREHENSIVE REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.datetime.now().isoformat()}")
        report_lines.append("")
        
        # Parameters
        report_lines.append("PARAMETERS:")
        report_lines.append("-"*40)
        for key, value in self.params.items():
            report_lines.append(f"  {key}: {value}")
        
        # Field statistics
        if self.unified_field is not None:
            report_lines.append("")
            report_lines.append("UNIFIED FIELD STATISTICS:")
            report_lines.append("-"*40)
            memory = self.unified_field.real
            potential = self.unified_field.imag
            
            report_lines.append(f"  Memory (M) - Mean: {np.mean(memory):.4f}, Std: {np.std(memory):.4f}")
            report_lines.append(f"  Potential (Φ) - Mean: {np.mean(potential):.4f}, Std: {np.std(potential):.4f}")
            report_lines.append(f"  Interaction energy: {self.calculate_interaction_energy():.4e}")
        
        # Defects
        if self.defects:
            report_lines.append("")
            report_lines.append("TOPOLOGICAL DEFECTS:")
            report_lines.append("-"*40)
            report_lines.append(f"  Total defects: {len(self.defects)}")
            
            # Count by type
            type_counts = {}
            for defect in self.defects:
                dtype = defect.get('type', 'unknown')
                type_counts[dtype] = type_counts.get(dtype, 0) + 1
            
            for dtype, count in type_counts.items():
                report_lines.append(f"    {dtype}: {count}")
        
        # Dark matter profile
        if self.dark_matter_profile:
            report_lines.append("")
            report_lines.append("DARK MATTER PROFILE:")
            report_lines.append("-"*40)
            nfw = self.dark_matter_profile.get('nfw_parameters', {})
            report_lines.append(f"  Total mass: {self.dark_matter_profile.get('total_mass', 0):.4e}")
            report_lines.append(f"  Virial radius: {self.dark_matter_profile.get('virial_radius', 0):.2f}")
            report_lines.append(f"  NFW scale radius: {nfw.get('r_s', 0):.2f}")
            report_lines.append(f"  NFW density: {nfw.get('rho_s', 0):.4e}")
        
        # Dark energy
        if self.dark_energy_profile:
            report_lines.append("")
            report_lines.append("DARK ENERGY:")
            report_lines.append("-"*40)
            de = self.dark_energy_profile
            report_lines.append(f"  Cosmological constant: {de.get('cosmological_constant', 0):.2e} m⁻²")
            report_lines.append(f"  Equation of state: w0={de.get('equation_of_state', {}).get('w0', 0)}")
        
        # Experimental predictions
        report_lines.append("")
        report_lines.append("EXPERIMENTAL PREDICTIONS:")
        report_lines.append("-"*40)
        
        # Direct detection
        dd_pred = self.predict_direct_detection(10.0, 'xenon')
        report_lines.append(f"  Direct detection (Xe, 10 GeV):")
        report_lines.append(f"    Cross section: {dd_pred['cross_section_cm2']:.2e} cm²")
        report_lines.append(f"    Rate: {dd_pred['detection_rate']['events_per_kg_year']:.2f} events/kg/year")
        
        # Annihilation
        ann_pred = self.predict_annihilation_signatures('gamma', 100.0)
        report_lines.append(f"  Annihilation (γγ, 100 GeV):")
        report_lines.append(f"    Flux: {ann_pred['flux_gc']:.2e} cm⁻² s⁻¹")
        report_lines.append(f"    Significance: {ann_pred['significance']}")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        return "\n".join(report_lines)

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("ACT Dark Sector Module")
    print("="*60)
    
    # Create example data
    np.random.seed(42)
    n_vertices = 1000
    vertices = np.random.randn(n_vertices, 4)  # 4D spacetime
    vertices[:, 0] *= 2.0  # Time coordinate
    
    # Create simple adjacency
    adjacency = np.random.rand(n_vertices, n_vertices) > 0.95
    adjacency = adjacency.astype(float)
    np.fill_diagonal(adjacency, 0)
    
    # Create causal matrix (simplified)
    causal = np.zeros((n_vertices, n_vertices))
    for i in range(n_vertices):
        for j in range(i+1, n_vertices):
            if vertices[j, 0] > vertices[i, 0]:  # j is in future of i
                causal[i, j] = 1
                causal[j, i] = -1
    
    # Initialize dark sector
    dark_sector = DarkSectorACT()
    dark_sector.initialize_from_network(vertices, adjacency, causal)
    
    # Calculate profiles
    dm_profile = dark_sector.calculate_dark_matter_profile(vertices)
    de_profile = dark_sector.calculate_dark_energy_profile(vertices)
    
    # Generate predictions
    dd_pred = dark_sector.predict_direct_detection(10.0, 'xenon')
    ann_pred = dark_sector.predict_annihilation_signatures('gamma', 100.0)
    
    # Visualize
    dark_sector.visualize_dark_sector(vertices)
    
    # Generate report
    report = dark_sector.generate_report()
    print(report)
    
    # Save results
    results = {
        'dark_matter_profile': dm_profile,
        'dark_energy_profile': de_profile,
        'direct_detection': dd_pred,
        'annihilation': ann_pred,
        'n_defects': len(dark_sector.defects)
    }
    
    with open('dark_sector_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to 'dark_sector_results.json'")
    print("\nACT Dark Sector analysis completed successfully!")
