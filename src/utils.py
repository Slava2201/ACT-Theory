"""
ACT Utilities Module
====================

Utility functions for Algebraic Causality Theory framework.

Includes:
- Mathematical tools
- Network analysis
- Statistical methods
- Visualization helpers
- File I/O operations
- Performance optimizations

Author: ACT Collaboration
Date: 2024
License: MIT
"""

import numpy as np
from scipy import sparse, spatial, stats, optimize, integrate
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from collections import defaultdict, Counter
import numba as nb
from numba import jit, prange
import time
import warnings
import pickle
import json
import h5py
import csv
from pathlib import Path
from datetime import datetime
import hashlib
import inspect
import functools
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
from matplotlib import rcParams, cm, colors
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ACT_utils")

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure plotting
rcParams['figure.figsize'] = (12, 8)
rcParams['font.size'] = 12
sns.set_style("whitegrid")
sns.set_palette("husl")

# ============================================================================
# 1. MATHEMATICAL UTILITIES
# ============================================================================

class MathUtils:
    """Mathematical utility functions for ACT."""
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def fast_norm(x: np.ndarray) -> float:
        """Fast Euclidean norm calculation."""
        return np.sqrt(np.sum(x**2))
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def fast_distance_matrix(points: np.ndarray) -> np.ndarray:
        """Fast distance matrix calculation using Numba."""
        n = points.shape[0]
        dist = np.zeros((n, n))
        for i in prange(n):
            for j in prange(i+1, n):
                d = 0.0
                for k in range(points.shape[1]):
                    diff = points[i, k] - points[j, k]
                    d += diff * diff
                dist[i, j] = np.sqrt(d)
                dist[j, i] = dist[i, j]
        return dist
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def causal_interval(p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Calculate causal interval between two points.
        
        Returns:
        - Positive if p1 causally precedes p2
        - Negative if p2 causally precedes p1
        - Zero if spacelike separated
        """
        dt = p2[0] - p1[0]
        if dt == 0:
            return 0.0
        
        # Spatial distance squared
        dx2 = 0.0
        for i in range(1, len(p1)):
            diff = p2[i] - p1[i]
            dx2 += diff * diff
        
        # Check timelike separation
        if dt**2 > dx2:
            return dt / np.sqrt(dt**2 - dx2) if dt > 0 else -np.sqrt(dt**2 - dx2)
        return 0.0
    
    @staticmethod
    def hausdorff_dimension(points: np.ndarray, 
                           r_min: float = None, 
                           r_max: float = None,
                           n_bins: int = 20) -> float:
        """
        Estimate Hausdorff dimension using box-counting.
        
        Parameters:
        -----------
        points : np.ndarray
            Point cloud
        r_min, r_max : float
            Min and max box sizes
        n_bins : int
            Number of box sizes to test
            
        Returns:
        --------
        dimension : float
            Estimated Hausdorff dimension
        """
        if len(points) < 10:
            return points.shape[1] if len(points.shape) > 1 else 1
        
        # Default box size range
        if r_min is None:
            r_min = np.min(np.std(points, axis=0)) / 100
        if r_max is None:
            r_max = np.max(np.ptp(points, axis=0))
        
        # Generate box sizes
        radii = np.logspace(np.log10(r_min), np.log10(r_max), n_bins)
        
        counts = []
        for r in radii:
            # Simple box counting
            mins = np.min(points, axis=0)
            maxs = np.max(points, axis=0)
            
            # Number of boxes needed
            n_boxes = np.prod(np.ceil((maxs - mins) / r))
            counts.append(n_boxes)
        
        # Fit log-log to get dimension
        log_r = np.log(radii)
        log_n = np.log(counts)
        
        # Linear fit
        coeffs = np.polyfit(log_r, log_n, 1)
        dimension = -coeffs[0]  # Negative slope
        
        return dimension
    
    @staticmethod
    def curvature_tensor(vertices: np.ndarray, 
                        tetrahedra: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Calculate curvature tensor from simplicial complex.
        
        Parameters:
        -----------
        vertices : np.ndarray
            Vertex coordinates
        tetrahedra : List[Tuple]
            List of tetrahedron vertex indices
            
        Returns:
        --------
        curvature : np.ndarray
            Ricci curvature tensor estimate
        """
        n_vertices = len(vertices)
        curvature = np.zeros((n_vertices, n_vertices))
        
        for tetra in tetrahedra:
            if len(tetra) != 4:
                continue
            
            v0, v1, v2, v3 = tetra
            
            # Calculate volume
            vec1 = vertices[v1] - vertices[v0]
            vec2 = vertices[v2] - vertices[v0]
            vec3 = vertices[v3] - vertices[v0]
            
            volume = np.abs(np.dot(vec1, np.cross(vec2, vec3))) / 6.0
            
            if volume > 0:
                # Simplified curvature contribution
                for i in range(4):
                    for j in range(4):
                        if i != j:
                            curvature[tetra[i], tetra[j]] += 1.0 / volume
        
        # Normalize
        row_sums = curvature.sum(axis=1)
        row_sums[row_sums == 0] = 1
        curvature = curvature / row_sums[:, np.newaxis]
        
        return curvature
    
    @staticmethod
    def entanglement_entropy(adjacency: np.ndarray, 
                            subsystem: np.ndarray = None) -> float:
        """
        Calculate entanglement entropy from adjacency matrix.
        
        Parameters:
        -----------
        adjacency : np.ndarray
            Adjacency matrix
        subsystem : np.ndarray
            Indices of subsystem A
            
        Returns:
        --------
        entropy : float
            Entanglement entropy S_A
        """
        if adjacency.shape[0] < 2:
            return 0.0
        
        # Use Laplacian
        degree = np.sum(adjacency, axis=1)
        laplacian = np.diag(degree) - adjacency
        
        if subsystem is None:
            # Use random subsystem
            n = adjacency.shape[0]
            n_a = max(1, n // 2)
            subsystem = np.random.choice(n, n_a, replace=False)
        
        # Reduced density matrix (simplified)
        laplacian_a = laplacian[np.ix_(subsystem, subsystem)]
        
        try:
            # Eigenvalues
            eigenvalues = np.linalg.eigvalsh(laplacian_a)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            
            # Von Neumann entropy
            entropy = -np.sum(eigenvalues * np.log(eigenvalues))
        except:
            entropy = 0.0
        
        return entropy
    
    @staticmethod
    def topological_invariants(graph: nx.Graph) -> Dict[str, Any]:
        """
        Calculate topological invariants of a graph.
        
        Returns:
        --------
        invariants : Dict
            Dictionary of topological invariants
        """
        invariants = {}
        
        try:
            # Betti numbers (simplified)
            n_components = nx.number_connected_components(graph)
            invariants['betti_0'] = n_components  # Number of components
            
            # Euler characteristic
            n_vertices = graph.number_of_nodes()
            n_edges = graph.number_of_edges()
            invariants['euler_characteristic'] = n_vertices - n_edges
            
            # Average degree
            degrees = [d for _, d in graph.degree()]
            invariants['average_degree'] = np.mean(degrees)
            invariants['degree_distribution'] = np.histogram(degrees, bins=20)[0].tolist()
            
            # Clustering coefficient
            invariants['clustering_coefficient'] = nx.average_clustering(graph)
            
            # Diameter (for connected graphs)
            if n_components == 1:
                invariants['diameter'] = nx.diameter(graph)
            else:
                invariants['diameter'] = float('inf')
            
            # Spectral properties
            laplacian = nx.laplacian_matrix(graph).toarray()
            eigenvalues = np.linalg.eigvalsh(laplacian)
            invariants['spectral_gap'] = eigenvalues[1] if len(eigenvalues) > 1 else 0
            
        except Exception as e:
            logger.warning(f"Failed to calculate topological invariants: {e}")
        
        return invariants

# ============================================================================
# 2. NETWORK UTILITIES
# ============================================================================

class NetworkUtils:
    """Network analysis utilities for ACT."""
    
    @staticmethod
    def build_causal_network(vertices: np.ndarray, 
                            threshold: float = 1.0) -> csr_matrix:
        """
        Build causal network from vertex coordinates.
        
        Parameters:
        -----------
        vertices : np.ndarray
            Vertex coordinates (time in first column)
        threshold : float
            Causality threshold
            
        Returns:
        --------
        causal_matrix : csr_matrix
            Sparse causal adjacency matrix
        """
        n = len(vertices)
        rows = []
        cols = []
        data = []
        
        for i in range(n):
            for j in range(i+1, n):
                dt = vertices[j, 0] - vertices[i, 0]
                
                if dt > 0:  # j is in future of i
                    dx2 = np.sum((vertices[j, 1:] - vertices[i, 1:])**2)
                    
                    if dt**2 > dx2:  # Timelike separation
                        weight = dt / np.sqrt(dt**2 - dx2)
                        if weight > threshold:
                            rows.append(i)
                            cols.append(j)
                            data.append(1)
                            rows.append(j)
                            cols.append(i)
                            data.append(-1)  # Reverse direction
        
        # Create sparse matrix
        causal_matrix = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        return causal_matrix
    
    @staticmethod
    def find_cliques(adjacency: csr_matrix, 
                    min_size: int = 3, 
                    max_size: int = 10) -> List[List[int]]:
        """
        Find cliques in adjacency matrix.
        
        Parameters:
        -----------
        adjacency : csr_matrix
            Adjacency matrix
        min_size, max_size : int
            Min and max clique sizes
            
        Returns:
        --------
        cliques : List[List[int]]
            List of cliques
        """
        # Convert to networkx graph
        G = nx.from_scipy_sparse_array(adjacency)
        
        # Find cliques
        all_cliques = list(nx.find_cliques(G))
        
        # Filter by size
        cliques = [c for c in all_cliques if min_size <= len(c) <= max_size]
        
        # Sort by size (largest first)
        cliques.sort(key=len, reverse=True)
        
        return cliques
    
    @staticmethod
    def community_detection(adjacency: csr_matrix, 
                          method: str = 'louvain') -> Dict[int, int]:
        """
        Detect communities in network.
        
        Parameters:
        -----------
        adjacency : csr_matrix
            Adjacency matrix
        method : str
            Method to use ('louvain', 'label_propagation', 'girvan_newman')
            
        Returns:
        --------
        communities : Dict
            Mapping from node index to community id
        """
        G = nx.from_scipy_sparse_array(adjacency)
        
        if method == 'louvain':
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(G)
            except ImportError:
                logger.warning("python-louvain not installed. Using label propagation.")
                partition = nx.algorithms.community.label_propagation.asyn_lpa_communities(G)
                # Convert to dict
                communities = {}
                for i, comm in enumerate(partition):
                    for node in comm:
                        communities[node] = i
                partition = communities
        
        elif method == 'label_propagation':
            communities = nx.algorithms.community.label_propagation.asyn_lpa_communities(G)
            partition = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    partition[node] = i
        
        elif method == 'girvan_newman':
            communities = nx.algorithms.community.girvan_newman(G)
            # Take first level
            partition = {}
            for i, comm in enumerate(next(communities)):
                for node in comm:
                    partition[node] = i
        
        else:
            raise ValueError(f"Unknown community detection method: {method}")
        
        return partition
    
    @staticmethod
    def calculate_centrality(adjacency: csr_matrix, 
                           centrality_type: str = 'betweenness') -> np.ndarray:
        """
        Calculate centrality measures for nodes.
        
        Parameters:
        -----------
        adjacency : csr_matrix
            Adjacency matrix
        centrality_type : str
            Type of centrality ('degree', 'betweenness', 'closeness', 'eigenvector')
            
        Returns:
        --------
        centrality : np.ndarray
            Centrality values for each node
        """
        G = nx.from_scipy_sparse_array(adjacency)
        n = len(G)
        
        if centrality_type == 'degree':
            centrality = np.array([d for _, d in G.degree()])
        
        elif centrality_type == 'betweenness':
            centrality = np.array(list(nx.betweenness_centrality(G).values()))
        
        elif centrality_type == 'closeness':
            centrality = np.array(list(nx.closeness_centrality(G).values()))
        
        elif centrality_type == 'eigenvector':
            centrality = np.array(list(nx.eigenvector_centrality(G, max_iter=1000).values()))
        
        else:
            raise ValueError(f"Unknown centrality type: {centrality_type}")
        
        # Normalize
        if np.max(centrality) > 0:
            centrality = centrality / np.max(centrality)
        
        return centrality
    
    @staticmethod
    def random_network(n_nodes: int, 
                      connection_prob: float = 0.1,
                      seed: int = None) -> csr_matrix:
        """
        Generate random network.
        
        Parameters:
        -----------
        n_nodes : int
            Number of nodes
        connection_prob : float
            Probability of connection
        seed : int, optional
            Random seed
            
        Returns:
        --------
        adjacency : csr_matrix
            Random adjacency matrix
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random adjacency
        random_mat = np.random.rand(n_nodes, n_nodes)
        adjacency = (random_mat < connection_prob).astype(float)
        
        # Make symmetric and remove self-loops
        adjacency = np.maximum(adjacency, adjacency.T)
        np.fill_diagonal(adjacency, 0)
        
        return csr_matrix(adjacency)
    
    @staticmethod
    def scale_free_network(n_nodes: int, 
                          m: int = 2,
                          seed: int = None) -> csr_matrix:
        """
        Generate scale-free network using Barabási-Albert model.
        
        Parameters:
        -----------
        n_nodes : int
            Number of nodes
        m : int
            Number of edges to attach from new node
        seed : int, optional
            Random seed
            
        Returns:
        --------
        adjacency : csr_matrix
            Scale-free adjacency matrix
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate using networkx
        G = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
        adjacency = nx.adjacency_matrix(G)
        
        return adjacency

# ============================================================================
# 3. STATISTICAL UTILITIES
# ============================================================================

class StatisticalUtils:
    """Statistical analysis utilities for ACT."""
    
    @staticmethod
    def bootstrap_confidence(data: np.ndarray, 
                           statistic: Callable,
                           n_bootstrap: int = 1000,
                           confidence: float = 0.95,
                           seed: int = None) -> Dict[str, float]:
        """
        Calculate bootstrap confidence intervals.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data
        statistic : Callable
            Function to calculate statistic
        n_bootstrap : int
            Number of bootstrap samples
        confidence : float
            Confidence level
        seed : int, optional
            Random seed
            
        Returns:
        --------
        result : Dict
            Dictionary with statistic and confidence interval
        """
        if seed is not None:
            np.random.seed(seed)
        
        n = len(data)
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            sample = data[np.random.choice(n, n, replace=True)]
            bootstrap_stats.append(statistic(sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate percentiles
        alpha = (1 - confidence) / 2
        lower = np.percentile(bootstrap_stats, 100 * alpha)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha))
        
        return {
            'statistic': statistic(data),
            'mean': np.mean(bootstrap_stats),
            'std': np.std(bootstrap_stats),
            'confidence_interval': (float(lower), float(upper)),
            'confidence_level': confidence,
            'n_bootstrap': n_bootstrap
        }
    
    @staticmethod
    def mcmc_sampling(log_probability: Callable,
                     n_samples: int = 10000,
                     n_chains: int = 4,
                     initial_params: np.ndarray = None,
                     seed: int = None) -> Dict[str, Any]:
        """
        Perform MCMC sampling using emcee.
        
        Parameters:
        -----------
        log_probability : Callable
            Log-probability function
        n_samples : int
            Number of samples per chain
        n_chains : int
            Number of chains
        initial_params : np.ndarray
            Initial parameter values
        seed : int, optional
            Random seed
            
        Returns:
        --------
        samples : Dict
            MCMC sampling results
        """
        try:
            import emcee
            
            if seed is not None:
                np.random.seed(seed)
            
            # Determine parameter dimension
            if initial_params is None:
                # Try to infer from function signature
                sig = inspect.signature(log_probability)
                n_params = len([p for p in sig.parameters.values() 
                              if p.name != 'self'])
                initial_params = np.random.randn(n_params)
            else:
                n_params = len(initial_params)
            
            # Initialize walkers
            n_walkers = n_chains * 2
            pos = initial_params + 1e-4 * np.random.randn(n_walkers, n_params)
            
            # Run sampler
            sampler = emcee.EnsembleSampler(n_walkers, n_params, log_probability)
            sampler.run_mcmc(pos, n_samples, progress=True)
            
            # Process samples
            samples_flat = sampler.get_chain(discard=n_samples//2, flat=True)
            
            result = {
                'samples': samples_flat,
                'acceptance_fraction': np.mean(sampler.acceptance_fraction),
                'autocorrelation_time': sampler.get_autocorr_time(),
                'log_prob': sampler.get_log_prob(),
                'sampler': sampler
            }
            
            return result
            
        except ImportError:
            logger.warning("emcee not installed. Using simple random sampling.")
            # Fallback to random sampling
            if initial_params is None:
                initial_params = np.zeros(1)
            
            n_params = len(initial_params)
            samples = np.random.randn(n_samples, n_params)
            
            return {
                'samples': samples,
                'acceptance_fraction': 1.0,
                'autocorrelation_time': np.ones(n_params),
                'log_prob': np.zeros(n_samples),
                'sampler': None
            }
    
    @staticmethod
    def correlation_analysis(data: np.ndarray,
                           method: str = 'pearson') -> Dict[str, Any]:
        """
        Perform correlation analysis.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data (n_samples x n_features)
        method : str
            Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
        --------
        correlations : Dict
            Correlation analysis results
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        n_features = data.shape[1]
        
        if method == 'pearson':
            corr_matrix = np.corrcoef(data.T)
        elif method == 'spearman':
            from scipy.stats import spearmanr
            corr_matrix, _ = spearmanr(data)
        elif method == 'kendall':
            from scipy.stats import kendalltau
            corr_matrix = np.eye(n_features)
            for i in range(n_features):
                for j in range(i+1, n_features):
                    tau, _ = kendalltau(data[:, i], data[:, j])
                    corr_matrix[i, j] = tau
                    corr_matrix[j, i] = tau
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        
        return {
            'correlation_matrix': corr_matrix,
            'eigenvalues': eigenvalues,
            'condition_number': np.max(eigenvalues) / np.min(eigenvalues[eigenvalues > 0]),
            'method': method
        }
    
    @staticmethod
    def time_series_analysis(series: np.ndarray,
                           max_lag: int = 50) -> Dict[str, Any]:
        """
        Analyze time series data.
        
        Parameters:
        -----------
        series : np.ndarray
            Time series data
        max_lag : int
            Maximum lag for autocorrelation
            
        Returns:
        --------
        analysis : Dict
            Time series analysis results
        """
        n = len(series)
        
        # Basic statistics
        mean = np.mean(series)
        std = np.std(series)
        skew = stats.skew(series)
        kurtosis = stats.kurtosis(series)
        
        # Autocorrelation
        lags = np.arange(max_lag)
        autocorr = np.array([np.corrcoef(series[:-lag], series[lag:])[0, 1] 
                           for lag in lags if lag < n/2])
        
        # Power spectrum
        if n > 1:
            spectrum = np.abs(np.fft.fft(series))**2
            frequencies = np.fft.fftfreq(n)
        else:
            spectrum = np.array([0])
            frequencies = np.array([0])
        
        # Hurst exponent (estimate)
        def hurst_exponent(x):
            n = len(x)
            t = np.arange(1, n+1)
            y = np.cumsum(x - np.mean(x))
            rs = (np.max(y) - np.min(y)) / np.std(x)
            return np.log(rs) / np.log(n)
        
        hurst = hurst_exponent(series) if n > 10 else 0.5
        
        return {
            'basic_stats': {
                'mean': mean,
                'std': std,
                'skewness': skew,
                'kurtosis': kurtosis,
                'min': np.min(series),
                'max': np.max(series)
            },
            'autocorrelation': {
                'lags': lags[:len(autocorr)],
                'values': autocorr,
                'decay_time': np.argmax(autocorr < 0.5) if np.any(autocorr < 0.5) else max_lag
            },
            'spectral_analysis': {
                'frequencies': frequencies,
                'power_spectrum': spectrum,
                'dominant_frequency': frequencies[np.argmax(spectrum)]
            },
            'hurst_exponent': hurst,
            'stationary_test': {
                'adf_pvalue': stats.kpss(series, regression='c')[1] if n > 10 else 1.0,
                'is_stationary': hurst < 0.7
            }
        }

# ============================================================================
# 4. VISUALIZATION UTILITIES
# ============================================================================

class VisualizationUtils:
    """Visualization utilities for ACT."""
    
    @staticmethod
    def plot_network_3d(coordinates: np.ndarray,
                       adjacency: csr_matrix = None,
                       node_color: np.ndarray = None,
                       node_size: np.ndarray = None,
                       edge_color: str = 'gray',
                       edge_alpha: float = 0.1,
                       title: str = "Network Visualization",
                       save_path: str = None) -> plt.Figure:
        """
        Create 3D network visualization.
        
        Parameters:
        -----------
        coordinates : np.ndarray
            Node coordinates (at least 3D)
        adjacency : csr_matrix, optional
            Adjacency matrix for edges
        node_color : np.ndarray, optional
            Color values for nodes
        node_size : np.ndarray, optional
            Size values for nodes
        edge_color : str
            Color for edges
        edge_alpha : float
            Alpha transparency for edges
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract coordinates
        if coordinates.shape[1] > 3:
            x, y, z = coordinates[:, 1], coordinates[:, 2], coordinates[:, 3]
        else:
            x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        
        # Default node properties
        if node_color is None:
            node_color = coordinates[:, 0] if coordinates.shape[1] > 0 else 'blue'
        if node_size is None:
            node_size = 20
        
        # Plot nodes
        scatter = ax.scatter(x, y, z, 
                           c=node_color, 
                           s=node_size,
                           cmap='viridis',
                           alpha=0.8,
                           edgecolors='black',
                           linewidth=0.5)
        
        # Plot edges if adjacency provided
        if adjacency is not None:
            # Convert to COO for edge iteration
            adj_coo = adjacency.tocoo()
            
            for i, j in zip(adj_coo.row, adj_coo.col):
                if i < j:  # Plot each edge once
                    ax.plot([x[i], x[j]], 
                           [y[i], y[j]], 
                           [z[i], z[j]], 
                           color=edge_color, 
                           alpha=edge_alpha,
                           linewidth=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # Add colorbar if node_color is numeric
        if np.issubdtype(node_color.dtype, np.number):
            plt.colorbar(scatter, ax=ax, label='Node value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_heatmap(matrix: np.ndarray,
                    title: str = "Heatmap",
                    cmap: str = 'viridis',
                    log_scale: bool = False,
                    symmetric: bool = False,
                    save_path: str = None) -> plt.Figure:
        """
        Create heatmap visualization.
        
        Parameters:
        -----------
        matrix : np.ndarray
            Matrix to visualize
        title : str
            Plot title
        cmap : str
            Colormap
        log_scale : bool
            Whether to use log scale
        symmetric : bool
            Whether matrix is symmetric
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Prepare data
        data = matrix.copy()
        if log_scale:
            data = np.log10(np.abs(data) + 1e-10)
        
        # Plot heatmap
        im = ax.imshow(data, cmap=cmap, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        if log_scale:
            cbar.set_label('Log scale')
        
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_title(title)
        
        # Add grid for symmetric matrices
        if symmetric:
            ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_distribution(data: np.ndarray,
                         title: str = "Distribution",
                         bins: int = 50,
                         density: bool = True,
                         fit_distribution: str = None,
                         save_path: str = None) -> plt.Figure:
        """
        Plot distribution of data.
        
        Parameters:
        -----------
        data : np.ndarray
            Data to plot
        title : str
            Plot title
        bins : int
            Number of histogram bins
        density : bool
            Whether to normalize histogram
        fit_distribution : str, optional
            Distribution to fit ('normal', 'lognormal', 'exponential')
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        n, bins, patches = ax.hist(data, bins=bins, density=density, 
                                  alpha=0.7, edgecolor='black')
        
        # Fit distribution if requested
        if fit_distribution is not None:
            x = np.linspace(np.min(data), np.max(data), 1000)
            
            if fit_distribution == 'normal':
                mu, sigma = stats.norm.fit(data)
                y = stats.norm.pdf(x, mu, sigma)
                label = f'Normal fit: μ={mu:.3f}, σ={sigma:.3f}'
                
            elif fit_distribution == 'lognormal':
                shape, loc, scale = stats.lognorm.fit(data)
                y = stats.lognorm.pdf(x, shape, loc, scale)
                label = f'Log-normal fit'
                
            elif fit_distribution == 'exponential':
                loc, scale = stats.expon.fit(data)
                y = stats.expon.pdf(x, loc, scale)
                label = f'Exponential fit: λ={1/scale:.3f}'
                
            else:
                raise ValueError(f"Unknown distribution: {fit_distribution}")
            
            ax.plot(x, y, 'r-', linewidth=2, label=label)
            ax.legend()
        
        ax.set_xlabel('Value')
        ax.set_ylabel('Density' if density else 'Count')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = (f'Mean: {np.mean(data):.3f}\n'
                     f'Std: {np.std(data):.3f}\n'
                     f'Skew: {stats.skew(data):.3f}\n'
                     f'Kurtosis: {stats.kurtosis(data):.3f}')
        
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def create_interactive_plot(coordinates: np.ndarray,
                               node_properties: Dict[str, np.ndarray] = None,
                               title: str = "Interactive Network") -> go.Figure:
        """
        Create interactive 3D plot using Plotly.
        
        Parameters:
        -----------
        coordinates : np.ndarray
            Node coordinates
        node_properties : Dict, optional
            Additional node properties for coloring/sizing
        title : str
            Plot title
            
        Returns:
        --------
        fig : plotly.graph_objects.Figure
            Interactive figure
        """
        # Prepare data
        if coordinates.shape[1] > 3:
            x, y, z = coordinates[:, 1], coordinates[:, 2], coordinates[:, 3]
        else:
            x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        
        # Default properties
        if node_properties is None:
            node_properties = {'value': coordinates[:, 0]}
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter trace for each property
        for prop_name, prop_values in node_properties.items():
            fig.add_trace(go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=8,
                    color=prop_values,
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title=prop_name)
                ),
                name=prop_name,
                text=[f"Node {i}<br>{prop_name}: {v:.3f}" 
                     for i, v in enumerate(prop_values)],
                hoverinfo='text'
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            showlegend=True,
            hovermode='closest'
        )
        
        return fig

# ============================================================================
# 5. FILE I/O UTILITIES
# ============================================================================

class FileUtils:
    """File input/output utilities for ACT."""
    
    @staticmethod
    def save_model(model: Any, 
                  filename: str, 
                  format: str = 'pickle') -> None:
        """
        Save model to file.
        
        Parameters:
        -----------
        model : Any
            Model object to save
        filename : str
            Output filename
        format : str
            Format ('pickle', 'hdf5', 'json')
        """
        filename = Path(filename)
        
        if format == 'pickle':
            with open(filename, 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        elif format == 'hdf5':
            with h5py.File(filename, 'w') as f:
                # Save basic attributes
                f.attrs['saved_at'] = datetime.now().isoformat()
                f.attrs['model_type'] = type(model).__name__
                
                # Try to save model data
                if hasattr(model, '__dict__'):
                    for key, value in model.__dict__.items():
                        if isinstance(value, (np.ndarray, list, tuple, int, float, str)):
                            try:
                                f.create_dataset(key, data=value)
                            except:
                                logger.warning(f"Could not save {key} to HDF5")
        
        elif format == 'json':
            # Convert to JSON-serializable format
            def convert_to_serializable(obj):
                if isinstance(obj, (np.ndarray, np.generic)):
                    return obj.tolist()
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_serializable(x) for x in obj]
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                else:
                    return str(obj)
            
            data = convert_to_serializable(model.__dict__ if hasattr(model, '__dict__') else model)
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Model saved to {filename} ({format})")
    
    @staticmethod
    def load_model(filename: str, 
                  format: str = 'pickle') -> Any:
        """
        Load model from file.
        
        Parameters:
        -----------
        filename : str
            Input filename
        format : str
            Format ('pickle', 'hdf5', 'json')
            
        Returns:
        --------
        model : Any
            Loaded model
        """
        filename = Path(filename)
        
        if not filename.exists():
            raise FileNotFoundError(f"File not found: {filename}")
        
        if format == 'pickle':
            with open(filename, 'rb') as f:
                model = pickle.load(f)
        
        elif format == 'hdf5':
            model_data = {}
            with h5py.File(filename, 'r') as f:
                # Load attributes
                for key in f.attrs:
                    model_data[key] = f.attrs[key]
                
                # Load datasets
                for key in f.keys():
                    model_data[key] = f[key][()]
            
            # Create simple object
            class LoadedModel:
                def __init__(self, data):
                    self.__dict__.update(data)
            
            model = LoadedModel(model_data)
        
        elif format == 'json':
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Create simple object
            class LoadedModel:
                def __init__(self, data):
                    self.__dict__.update(data)
            
            model = LoadedModel(data)
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Model loaded from {filename}")
        return model
    
    @staticmethod
    def save_dataframe(df: pd.DataFrame, 
                      filename: str, 
                      format: str = 'csv') -> None:
        """
        Save DataFrame to file.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to save
        filename : str
            Output filename
        format : str
            Format ('csv', 'parquet', 'hdf5', 'excel')
        """
        filename = Path(filename)
        
        if format == 'csv':
            df.to_csv(filename, index=False)
        elif format == 'parquet':
            df.to_parquet(filename, index=False)
        elif format == 'hdf5':
            df.to_hdf(filename, key='data', mode='w')
        elif format == 'excel':
            df.to_excel(filename, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"DataFrame saved to {filename} ({format})")
    
    @staticmethod
    def load_dataframe(filename: str, 
                      format: str = 'csv') -> pd.DataFrame:
        """
        Load DataFrame from file.
        
        Parameters:
        -----------
        filename : str
            Input filename
        format : str
            Format ('csv', 'parquet', 'hdf5', 'excel')
            
        Returns:
        --------
        df : pd.DataFrame
            Loaded DataFrame
        """
        filename = Path(filename)
        
        if not filename.exists():
            raise FileNotFoundError(f"File not found: {filename}")
        
        if format == 'csv':
            df = pd.read_csv(filename)
        elif format == 'parquet':
            df = pd.read_parquet(filename)
        elif format == 'hdf5':
            df = pd.read_hdf(filename)
        elif format == 'excel':
            df = pd.read_excel(filename)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"DataFrame loaded from {filename}")
        return df
    
    @staticmethod
    def generate_report(data: Dict[str, Any], 
                       filename: str = None) -> str:
        """
        Generate formatted report from data.
        
        Parameters:
        -----------
        data : Dict
            Data to include in report
        filename : str, optional
            Output filename
            
        Returns:
        --------
        report : str
            Formatted report
        """
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("ACT ANALYSIS REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append("")
        
        def add_section(title: str, content: Dict, indent: int = 0):
            """Helper to add section to report."""
            indent_str = " " * indent
            report_lines.append(f"{indent_str}{title}:")
            report_lines.append(f"{indent_str}{'-'*len(title)}")
            
            for key, value in content.items():
                if isinstance(value, dict):
                    report_lines.append(f"{indent_str}  {key}:")
                    for k2, v2 in value.items():
                        report_lines.append(f"{indent_str}    {k2}: {v2}")
                elif isinstance(value, (list, tuple, np.ndarray)):
                    if len(value) > 5:
                        report_lines.append(f"{indent_str}  {key}: [{value[0]}, ..., {value[-1]}] (n={len(value)})")
                    else:
                        report_lines.append(f"{indent_str}  {key}: {value}")
                else:
                    report_lines.append(f"{indent_str}  {key}: {value}")
            
            report_lines.append("")
        
        # Add sections
        for section, content in data.items():
            if isinstance(content, dict):
                add_section(section, content)
            else:
                report_lines.append(f"{section}: {content}")
        
        report_lines.append("="*80)
        
        report = "\n".join(report_lines)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {filename}")
        
        return report

# ============================================================================
# 6. PERFORMANCE UTILITIES
# ============================================================================

class PerformanceUtils:
    """Performance optimization utilities."""
    
    @staticmethod
    def time_function(func: Callable, 
                     *args, 
                     **kwargs) -> Dict[str, Any]:
        """
        Time function execution.
        
        Parameters:
        -----------
        func : Callable
            Function to time
        *args, **kwargs
            Function arguments
            
        Returns:
        --------
        timing : Dict
            Timing information
        """
        import time
        
        start_time = time.perf_counter()
        start_cpu = time.process_time()
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.perf_counter()
        end_cpu = time.process_time()
        
        timing = {
            'wall_time': end_time - start_time,
            'cpu_time': end_cpu - start_cpu,
            'success': success,
            'function': func.__name__
        }
        
        if success:
            timing['result'] = result
        else:
            timing['error'] = error
        
        return timing
    
    @staticmethod
    def memory_usage() -> Dict[str, float]:
        """
        Get current memory usage.
        
        Returns:
        --------
        memory : Dict
            Memory usage in MB
        """
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        return {
            'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def parallel_matrix_operation(A: np.ndarray, 
                                 B: np.ndarray, 
                                 operation: str = 'multiply') -> np.ndarray:
        """
        Parallel matrix operation using Numba.
        
        Parameters:
        -----------
        A, B : np.ndarray
            Input matrices
        operation : str
            Operation ('multiply', 'add', 'subtract')
            
        Returns:
        --------
        result : np.ndarray
            Result matrix
        """
        if A.shape != B.shape:
            raise ValueError("Matrices must have same shape")
        
        n, m = A.shape
        result = np.zeros((n, m))
        
        if operation == 'multiply':
            for i in prange(n):
                for j in prange(m):
                    result[i, j] = A[i, j] * B[i, j]
        elif operation == 'add':
            for i in prange(n):
                for j in prange(m):
                    result[i, j] = A[i, j] + B[i, j]
        elif operation == 'subtract':
            for i in prange(n):
                for j in prange(m):
                    result[i, j] = A[i, j] - B[i, j]
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        return result
    
    @staticmethod
    def cache_results(func: Callable = None, 
                     maxsize: int = 128,
                     use_hash: bool = True):
        """
        Decorator to cache function results.
        
        Parameters:
        -----------
        maxsize : int
            Maximum cache size
        use_hash : bool
            Whether to use hash for cache keys
        """
        def decorator(f):
            cache = {}
            
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                # Create cache key
                if use_hash:
                    # Use hash for efficiency
                    key = hashlib.md5(
                        str(args).encode() + str(sorted(kwargs.items())).encode()
                    ).hexdigest()
                else:
                    # Use tuple (less efficient but more precise)
                    key = (args, tuple(sorted(kwargs.items())))
                
                # Check cache
                if key in cache:
                    return cache[key]
                
                # Compute and cache
                result = f(*args, **kwargs)
                
                # Manage cache size
                if len(cache) >= maxsize:
                    # Remove oldest (simple FIFO)
                    oldest_key = next(iter(cache))
                    del cache[oldest_key]
                
                cache[key] = result
                return result
            
            wrapper.cache_info = lambda: {
                'size': len(cache),
                'maxsize': maxsize
            }
            wrapper.clear_cache = lambda: cache.clear()
            
            return wrapper
        
        if func is None:
            return decorator
        else:
            return decorator(func)

# ============================================================================
# 7. CONFIGURATION UTILITIES
# ============================================================================

class ConfigUtils:
    """Configuration management utilities."""
    
    @staticmethod
    def load_config(config_file: str = 'config.json') -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Parameters:
        -----------
        config_file : str
            Configuration file path
            
        Returns:
        --------
        config : Dict
            Configuration dictionary
        """
        config_file = Path(config_file)
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_file}")
        else:
            config = {}
            logger.warning(f"Configuration file not found: {config_file}")
        
        return config
    
    @staticmethod
    def save_config(config: Dict[str, Any], 
                   config_file: str = 'config.json') -> None:
        """
        Save configuration to file.
        
        Parameters:
        -----------
        config : Dict
            Configuration dictionary
        config_file : str
            Output file path
        """
        config_file = Path(config_file)
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {config_file}")
    
    @staticmethod
    def merge_configs(default: Dict[str, Any], 
                     user: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge default and user configurations.
        
        Parameters:
        -----------
        default : Dict
            Default configuration
        user : Dict
            User configuration
            
        Returns:
        --------
        merged : Dict
            Merged configuration
        """
        def deep_merge(a, b):
            """Recursively merge dictionaries."""
            result = a.copy()
            
            for key, value in b.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            
            return result
        
        return deep_merge(default, user)

# ============================================================================
# 8. EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("ACT Utilities Module")
    print("="*60)
    
    # Example 1: Mathematical utilities
    print("\n1. Mathematical Utilities:")
    print("-"*40)
    
    points = np.random.randn(100, 3)
    distances = MathUtils.fast_distance_matrix(points[:10, :])
    print(f"Distance matrix shape: {distances.shape}")
    print(f"Mean distance: {np.mean(distances):.3f}")
    
    # Example 2: Network utilities
    print("\n2. Network Utilities:")
    print("-"*40)
    
    adj = NetworkUtils.random_network(20, 0.2, seed=42)
    cliques = NetworkUtils.find_cliques(adj, min_size=3, max_size=5)
    print(f"Found {len(cliques)} cliques")
    if cliques:
        print(f"Largest clique: {cliques[0]}")
    
    # Example 3: Statistical utilities
    print("\n3. Statistical Utilities:")
    print("-"*40)
    
    data = np.random.normal(0, 1, 1000)
    stats = StatisticalUtils.bootstrap_confidence(
        data, np.mean, n_bootstrap=100, seed=42
    )
    print(f"Mean: {stats['statistic']:.3f}")
    print(f"95% CI: {stats['confidence_interval']}")
    
    # Example 4: Visualization
    print("\n4. Visualization Utilities:")
    print("-"*40)
    
    coords = np.random.randn(50, 3)
    fig = VisualizationUtils.plot_distribution(
        coords[:, 0], title="Random Distribution", fit_distribution='normal'
    )
    print("Distribution plot created")
    
    # Example 5: File I/O
    print("\n5. File I/O Utilities:")
    print("-"*40)
    
    test_data = {'a': 1, 'b': [2, 3, 4], 'c': np.array([5, 6, 7])}
    FileUtils.save_model(test_data, 'test_data.pkl', 'pickle')
    loaded = FileUtils.load_model('test_data.pkl', 'pickle')
    print(f"Data saved and loaded successfully: {loaded['a']}")
    
    # Clean up
    import os
    if os.path.exists('test_data.pkl'):
        os.remove('test_data.pkl')
    
    print("\n" + "="*60)
    print("ACT Utilities module loaded successfully!")
    print("Available classes:")
    print("  - MathUtils: Mathematical functions")
    print("  - NetworkUtils: Network analysis")
    print("  - StatisticalUtils: Statistical methods")
    print("  - VisualizationUtils: Plotting functions")
    print("  - FileUtils: File I/O operations")
    print("  - PerformanceUtils: Optimization tools")
    print("  - ConfigUtils: Configuration management")
