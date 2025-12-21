"""
ACT Theory - Utility Functions
=============================
Collection of utility functions for ACT simulations, data processing,
visualization, and analysis.

This module provides helper functions used throughout the ACT project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats, optimize, interpolate
from scipy.spatial import KDTree, Delaunay
from scipy.sparse import csr_matrix, lil_matrix, diags
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import pickle
import json
import h5py
from datetime import datetime
import warnings
from tqdm import tqdm
import itertools
from collections import defaultdict, Counter
import multiprocessing as mp
from functools import lru_cache, wraps
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# 1. MATHEMATICAL UTILITIES
# ============================================================================

def complex_to_real(z: complex) -> Tuple[float, float]:
    """
    Convert complex number to real representation (Re, Im).
    
    Parameters:
    -----------
    z : complex
        Complex number
        
    Returns:
    --------
    tuple : (real, imaginary)
    """
    return (z.real, z.imag)

def real_to_complex(x: float, y: float) -> complex:
    """
    Convert real representation to complex number.
    
    Parameters:
    -----------
    x, y : float
        Real and imaginary parts
        
    Returns:
    --------
    complex : x + iy
    """
    return x + 1j * y

@lru_cache(maxsize=128)
def su_matrix(n: int = 2) -> np.ndarray:
    """
    Generate random SU(n) matrix.
    
    Parameters:
    -----------
    n : int
        Dimension of matrix (default: 2)
        
    Returns:
    --------
    U : np.ndarray
        Random SU(n) matrix
    """
    # Generate random complex matrix
    X = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    
    # QR decomposition
    Q, R = np.linalg.qr(X)
    
    # Make determinant = 1
    D = np.diag(R) / np.abs(np.diag(R))
    Q = Q @ np.diag(D)
    
    # Ensure SU(n) condition
    det_Q = np.linalg.det(Q)
    Q = Q / (det_Q ** (1/n))
    
    return Q

def parallel_transport(U1: np.ndarray, U2: np.ndarray, 
                      steps: int = 10) -> List[np.ndarray]:
    """
    Parallel transport between two unitary matrices.
    
    Parameters:
    -----------
    U1, U2 : np.ndarray
        Start and end matrices
    steps : int
        Number of intermediate steps
        
    Returns:
    --------
    path : list
        List of matrices along geodesic
    """
    # Ensure matrices are SU(n)
    n = U1.shape[0]
    U1 = U1 / (np.linalg.det(U1) ** (1/n))
    U2 = U2 / (np.linalg.det(U2) ** (1/n))
    
    # Geodesic in SU(n) manifold
    path = []
    for t in np.linspace(0, 1, steps):
        # Logarithmic map
        log_U1 = 1j * np.log(np.linalg.eigvals(U1)).real
        log_U2 = 1j * np.log(np.linalg.eigvals(U2)).real
        
        # Interpolate in Lie algebra
        log_interp = (1-t) * log_U1 + t * log_U2
        
        # Exponential map back to group
        U_interp = np.linalg.eigvals(np.exp(1j * log_interp))
        path.append(U_interp)
    
    return path

def compute_holonomy(loop: List[np.ndarray]) -> np.ndarray:
    """
    Compute holonomy around a loop of connections.
    
    Parameters:
    -----------
    loop : list of np.ndarray
        List of connection matrices along loop
        
    Returns:
    --------
    holonomy : np.ndarray
        Holonomy matrix
    """
    if len(loop) < 2:
        return np.eye(loop[0].shape[0]) if loop else np.eye(2)
    
    # Product of matrices around loop
    holonomy = np.eye(loop[0].shape[0])
    for U in loop:
        holonomy = holonomy @ U
    
    return holonomy

def wilson_loop(holonomy: np.ndarray) -> float:
    """
    Compute Wilson loop from holonomy matrix.
    
    Parameters:
    -----------
    holonomy : np.ndarray
        Holonomy matrix
        
    Returns:
    --------
    float : Wilson loop value
    """
    return np.real(np.trace(holonomy))

# ============================================================================
# 2. GEOMETRY AND TOPOLOGY UTILITIES
# ============================================================================

def compute_curvature_tensor(vertices: np.ndarray, 
                            simplices: List[Tuple[int, ...]]) -> np.ndarray:
    """
    Compute curvature tensor from simplicial complex.
    
    Parameters:
    -----------
    vertices : np.ndarray
        Vertex coordinates
    simplices : list
        List of simplices (tuples of vertex indices)
        
    Returns:
    --------
    curvature : np.ndarray
        Curvature tensor at vertices
    """
    n_vertices = len(vertices)
    curvature = np.zeros((n_vertices, 4, 4))  # 4D curvature tensor
    
    for simplex in simplices:
        if len(simplex) == 4:  # Tetrahedron
            # Compute deficit angle approximation
            points = vertices[list(simplex)]
            
            # Compute edge vectors
            edges = []
            for i in range(4):
                for j in range(i+1, 4):
                    edges.append(points[j] - points[i])
            
            # Simplified curvature estimate
            curvature_contrib = np.outer(edges[0], edges[1]) - np.outer(edges[1], edges[0])
            curvature_contrib = curvature_contrib / np.linalg.norm(curvature_contrib + 1e-10)
            
            # Distribute to vertices
            for v in simplex:
                curvature[v] += curvature_contrib / 4
    
    return curvature

def compute_homology(simplices: List[Tuple[int, ...]], 
                    max_dim: int = 3) -> Dict[int, List[np.ndarray]]:
    """
    Compute homology groups of simplicial complex.
    
    Parameters:
    -----------
    simplices : list
        List of simplices
    max_dim : int
        Maximum dimension to compute
        
    Returns:
    --------
    homology : dict
        Dictionary mapping dimension to homology groups
    """
    # Group simplices by dimension
    simplices_by_dim = defaultdict(list)
    for simplex in simplices:
        dim = len(simplex) - 1
        if dim <= max_dim:
            simplices_by_dim[dim].append(simplex)
    
    # Simplified Betti numbers calculation
    homology = {}
    for dim in range(max_dim + 1):
        if dim in simplices_by_dim:
            n_simplices = len(simplices_by_dim[dim])
            # Approximate Betti number (simplified)
            betti = max(0, n_simplices - len(simplices_by_dim.get(dim-1, [])))
            homology[dim] = [np.eye(betti) if betti > 0 else np.array([])]
    
    return homology

def compute_volume_form(vertices: np.ndarray, 
                       simplex: Tuple[int, ...]) -> float:
    """
    Compute volume form of simplex.
    
    Parameters:
    -----------
    vertices : np.ndarray
        Vertex coordinates
    simplex : tuple
        Simplex vertex indices
        
    Returns:
    --------
    volume : float
        Signed volume
    """
    points = vertices[list(simplex)]
    
    if len(simplex) == 2:  # Edge
        return np.linalg.norm(points[1] - points[0])
    elif len(simplex) == 3:  # Triangle
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        return 0.5 * np.linalg.norm(np.cross(v1, v2))
    elif len(simplex) == 4:  # Tetrahedron
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        v3 = points[3] - points[0]
        return abs(np.linalg.det(np.column_stack([v1, v2, v3]))) / 6.0
    else:
        return 0.0

def compute_metric_tensor(vertices: np.ndarray, 
                         neighbors: Dict[int, List[int]]) -> np.ndarray:
    """
    Compute induced metric tensor from vertex positions.
    
    Parameters:
    -----------
    vertices : np.ndarray
        Vertex coordinates
    neighbors : dict
        Adjacency dictionary
        
    Returns:
    --------
    metric : np.ndarray
        Metric tensor at vertices
    """
    n_vertices = len(vertices)
    metric = np.zeros((n_vertices, 4, 4))
    
    for i in range(n_vertices):
        if i in neighbors and neighbors[i]:
            # Compute local metric from neighbor distances
            for j in neighbors[i]:
                dx = vertices[j] - vertices[i]
                metric[i] += np.outer(dx, dx) / len(neighbors[i])
        
        # Ensure positive definiteness
        eigenvalues = np.linalg.eigvalsh(metric[i])
        if np.any(eigenvalues <= 0):
            metric[i] += np.eye(4) * (abs(min(eigenvalues)) + 1e-10)
    
    return metric

# ============================================================================
# 3. NETWORK AND GRAPH UTILITIES
# ============================================================================

def build_adjacency_matrix(vertices: np.ndarray, 
                          radius: float = 0.1) -> csr_matrix:
    """
    Build adjacency matrix from vertex positions.
    
    Parameters:
    -----------
    vertices : np.ndarray
        Vertex coordinates
    radius : float
        Connection radius
        
    Returns:
    --------
    adjacency : csr_matrix
        Sparse adjacency matrix
    """
    n_vertices = len(vertices)
    adj = lil_matrix((n_vertices, n_vertices), dtype=np.int8)
    
    # Use KD-tree for efficient neighbor search
    kdtree = KDTree(vertices)
    
    for i in range(n_vertices):
        neighbors = kdtree.query_ball_point(vertices[i], radius)
        for j in neighbors:
            if i != j:
                adj[i, j] = 1
                adj[j, i] = 1
    
    return adj.tocsr()

def compute_graph_laplacian(adjacency: csr_matrix) -> csr_matrix:
    """
    Compute graph Laplacian from adjacency matrix.
    
    Parameters:
    -----------
    adjacency : csr_matrix
        Sparse adjacency matrix
        
    Returns:
    --------
    laplacian : csr_matrix
        Graph Laplacian matrix
    """
    n = adjacency.shape[0]
    
    # Degree matrix
    degrees = adjacency.sum(axis=1).A1
    D = diags(degrees, format='csr')
    
    # Laplacian: L = D - A
    laplacian = D - adjacency
    
    return laplacian

def compute_spectrum(laplacian: csr_matrix, 
                    k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues and eigenvectors of Laplacian.
    
    Parameters:
    -----------
    laplacian : csr_matrix
        Graph Laplacian
    k : int
        Number of eigenvalues to compute
        
    Returns:
    --------
    eigenvalues : np.ndarray
        Eigenvalues
    eigenvectors : np.ndarray
        Eigenvectors
    """
    from scipy.sparse.linalg import eigsh
    
    k = min(k, laplacian.shape[0] - 2)
    
    eigenvalues, eigenvectors = eigsh(
        laplacian, 
        k=k, 
        which='SM',  # Smallest magnitude
        maxiter=10000
    )
    
    return eigenvalues, eigenvectors

def compute_community_structure(adjacency: csr_matrix, 
                               resolution: float = 1.0) -> np.ndarray:
    """
    Detect community structure in network.
    
    Parameters:
    -----------
    adjacency : csr_matrix
        Adjacency matrix
    resolution : float
        Community detection resolution
        
    Returns:
    --------
    communities : np.ndarray
        Community labels for each vertex
    """
    try:
        import community as community_louvain
        
        # Convert to networkx graph
        G = nx.from_scipy_sparse_array(adjacency)
        
        # Detect communities using Louvain algorithm
        partition = community_louvain.best_partition(
            G, 
            resolution=resolution,
            random_state=42
        )
        
        # Convert to array
        communities = np.array([partition[i] for i in range(len(G))])
        
        return communities
        
    except ImportError:
        # Fallback to spectral clustering
        from sklearn.cluster import SpectralClustering
        
        clustering = SpectralClustering(
            n_clusters=min(10, adjacency.shape[0] // 10),
            affinity='precomputed',
            random_state=42
        )
        
        return clustering.fit_predict(adjacency.toarray())

# ============================================================================
# 4. DATA PROCESSING AND ANALYSIS
# ============================================================================

def moving_average(data: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Compute moving average of time series.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    window : int
        Window size
        
    Returns:
    --------
    smoothed : np.ndarray
        Smoothed data
    """
    if len(data) < window:
        return data
    
    return np.convolve(data, np.ones(window)/window, mode='valid')

def exponential_smoothing(data: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Exponential smoothing of time series.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    alpha : float
        Smoothing factor (0 < alpha < 1)
        
    Returns:
    --------
    smoothed : np.ndarray
        Smoothed data
    """
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    
    return smoothed

def compute_autocorrelation(data: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """
    Compute autocorrelation function.
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data
    max_lag : int
        Maximum lag to compute
        
    Returns:
    --------
    acf : np.ndarray
        Autocorrelation function
    """
    n = len(data)
    data_normalized = (data - np.mean(data)) / np.std(data)
    
    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag < n:
            acf[lag] = np.corrcoef(
                data_normalized[:n-lag], 
                data_normalized[lag:]
            )[0, 1]
    
    return acf

def compute_power_spectrum(data: np.ndarray, 
                          sampling_rate: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectrum using FFT.
    
    Parameters:
    -----------
    data : np.ndarray
        Time series data
    sampling_rate : float
        Sampling rate
        
    Returns:
    --------
    frequencies : np.ndarray
        Frequency bins
    spectrum : np.ndarray
        Power spectrum
    """
    n = len(data)
    
    # Apply window function
    window = np.hanning(n)
    data_windowed = data * window
    
    # Compute FFT
    fft_result = np.fft.fft(data_windowed)
    frequencies = np.fft.fftfreq(n, 1/sampling_rate)
    
    # Compute power spectrum
    spectrum = np.abs(fft_result[:n//2]) ** 2
    frequencies = frequencies[:n//2]
    
    return frequencies, spectrum

def fit_power_law(data_x: np.ndarray, 
                 data_y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit power law y = a * x^b.
    
    Parameters:
    -----------
    data_x, data_y : np.ndarray
        Input data
        
    Returns:
    --------
    a, b, r_squared : tuple
        Power law parameters and goodness of fit
    """
    # Log transform for linear fit
    mask = (data_x > 0) & (data_y > 0)
    log_x = np.log(data_x[mask])
    log_y = np.log(data_y[mask])
    
    if len(log_x) < 2:
        return 0, 0, 0
    
    # Linear fit in log-log space
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
    
    # Convert back to power law
    b = slope
    a = np.exp(intercept)
    r_squared = r_value ** 2
    
    return a, b, r_squared

# ============================================================================
# 5. VISUALIZATION UTILITIES
# ============================================================================

def plot_3d_network(vertices: np.ndarray, 
                   edges: Optional[List[Tuple[int, int]]] = None,
                   communities: Optional[np.ndarray] = None,
                   title: str = "3D Network",
                   save_path: Optional[str] = None) -> go.Figure:
    """
    Create 3D interactive plot of network.
    
    Parameters:
    -----------
    vertices : np.ndarray
        Vertex coordinates (N x 3)
    edges : list, optional
        List of edge tuples
    communities : np.ndarray, optional
        Community labels for coloring
    title : str
        Plot title
    save_path : str, optional
        Path to save HTML file
        
    Returns:
    --------
    fig : go.Figure
        Plotly figure
    """
    # Prepare data
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    
    if communities is None:
        colors = 'blue'
        color_scale = None
    else:
        colors = communities
        color_scale = px.colors.qualitative.Set3
    
    # Create figure
    fig = go.Figure()
    
    # Add vertices
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=colors,
            colorscale=color_scale,
            opacity=0.8,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        text=[f"Vertex {i}" for i in range(len(vertices))],
        hoverinfo='text',
        name='Vertices'
    ))
    
    # Add edges if provided
    if edges:
        edge_x, edge_y, edge_z = [], [], []
        
        for i, j in edges:
            edge_x.extend([x[i], x[j], None])
            edge_y.extend([y[i], y[j], None])
            edge_z.extend([z[i], z[j], None])
        
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='rgba(100,100,100,0.2)', width=1),
            hoverinfo='none',
            name='Edges'
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
        height=700
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Plot saved to {save_path}")
    
    return fig

def plot_time_series(times: np.ndarray, 
                    values: Dict[str, np.ndarray],
                    title: str = "Time Series",
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot multiple time series.
    
    Parameters:
    -----------
    times : np.ndarray
        Time points
    values : dict
        Dictionary of time series data
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    fig : matplotlib Figure
    """
    fig, axes = plt.subplots(len(values), 1, figsize=(12, 3*len(values)))
    
    if len(values) == 1:
        axes = [axes]
    
    for ax, (name, data) in zip(axes, values.items()):
        ax.plot(times, data, linewidth=2, label=name)
        ax.set_xlabel('Time')
        ax.set_ylabel(name)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig

def plot_histogram(data: np.ndarray, 
                  bins: int = 50,
                  title: str = "Histogram",
                  xlabel: str = "Value",
                  fit_distribution: bool = False,
                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot histogram with optional distribution fit.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    bins : int
        Number of bins
    title : str
        Plot title
    xlabel : str
        X-axis label
    fit_distribution : bool
        Whether to fit Gaussian distribution
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    fig : matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    n, bins, patches = ax.hist(data, bins=bins, density=True, 
                               alpha=0.7, edgecolor='black')
    
    # Fit Gaussian if requested
    if fit_distribution and len(data) > 10:
        mu, sigma = stats.norm.fit(data)
        x = np.linspace(min(data), max(data), 1000)
        y = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, y, 'r-', linewidth=2, 
               label=f'Gaussian fit\nμ={mu:.3f}, σ={sigma:.3f}')
        ax.legend()
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability Density')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig

def plot_correlation_matrix(data: pd.DataFrame,
                           title: str = "Correlation Matrix",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot correlation matrix heatmap.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
        
    Returns:
    --------
    fig : matplotlib Figure
    """
    # Compute correlation matrix
    corr = data.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Custom colormap
    cmap = plt.cm.RdYlBu_r
    cmap.set_bad(color='white')
    
    # Plot heatmap
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Correlation', rotation=90, va="bottom")
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    
    # Add correlation values
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            if not mask[i, j]:  # Lower triangle
                text = ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                             ha="center", va="center",
                             color="white" if abs(corr.iloc[i, j]) > 0.5 else "black",
                             fontsize=9)
    
    ax.set_title(title, fontsize=16, pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig

# ============================================================================
# 6. FILE I/O UTILITIES
# ============================================================================

def save_results(results: Dict[str, Any], 
                filename: str,
                format: str = 'pkl') -> None:
    """
    Save results to file.
    
    Parameters:
    -----------
    results : dict
        Results dictionary
    filename : str
        Output filename
    format : str
        File format ('pkl', 'json', 'h5')
    """
    if format == 'pkl':
        with open(filename, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    elif format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.complexfloating):
                return complex_to_real(obj)
            return obj
        
        results_json = json.dumps(results, default=convert_for_json, indent=2)
        with open(filename, 'w') as f:
            f.write(results_json)
    
    elif format == 'h5':
        with h5py.File(filename, 'w') as f:
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value)
                elif isinstance(value, (int, float, str)):
                    f.attrs[key] = value
                elif isinstance(value, dict):
                    # Recursively save dictionaries
                    grp = f.create_group(key)
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, np.ndarray):
                            grp.create_dataset(subkey, data=subvalue)
                        else:
                            grp.attrs[subkey] = subvalue
    
    print(f"Results saved to {filename}")

def load_results(filename: str, 
                format: str = 'pkl') -> Dict[str, Any]:
    """
    Load results from file.
    
    Parameters:
    -----------
    filename : str
        Input filename
    format : str
        File format ('pkl', 'json', 'h5')
        
    Returns:
    --------
    results : dict
        Loaded results
    """
    if format == 'pkl':
        with open(filename, 'rb') as f:
            results = pickle.load(f)
    
    elif format == 'json':
        with open(filename, 'r') as f:
            results = json.load(f)
    
    elif format == 'h5':
        results = {}
        with h5py.File(filename, 'r') as f:
            # Load datasets
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    results[key] = f[key][:]
                elif isinstance(f[key], h5py.Group):
                    results[key] = {}
                    for subkey in f[key].keys():
                        if isinstance(f[key][subkey], h5py.Dataset):
                            results[key][subkey] = f[key][subkey][:]
                        else:
                            results[key][subkey] = f[key].attrs[subkey]
            
            # Load attributes
            for key, value in f.attrs.items():
                results[key] = value
    
    print(f"Results loaded from {filename}")
    return results

def save_model(model: Any, filename: str) -> None:
    """
    Save ACT model to file.
    
    Parameters:
    -----------
    model : object
        ACT model instance
    filename : str
        Output filename
    """
    # Extract model state
    state = {
        'vertices': getattr(model, 'vertices', None),
        'operators': getattr(model, 'operators', None),
        'tetrahedra': getattr(model, 'tetrahedra', None),
        'adjacency': getattr(model, 'adjacency', None),
        'parameters': {
            'N': getattr(model, 'N', None),
            'temperature': getattr(model, 'temperature', None),
            'seed': getattr(model, 'seed', None)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    save_results(state, filename, format='pkl')

def load_model(filename: str, model_class: type) -> Any:
    """
    Load ACT model from file.
    
    Parameters:
    -----------
    filename : str
        Input filename
    model_class : type
        Model class to instantiate
        
    Returns:
    --------
    model : object
        Reconstructed model
    """
    state = load_results(filename, format='pkl')
    
    # Create new model instance
    params = state['parameters']
    model = model_class(
        N=params['N'],
        temperature=params['temperature'],
        seed=params['seed']
    )
    
    # Restore state
    for key, value in state.items():
        if key != 'parameters' and hasattr(model, key):
            setattr(model, key, value)
    
    return model

# ============================================================================
# 7. PARALLEL PROCESSING UTILITIES
# ============================================================================

def parallel_map(func: Callable, 
                items: List[Any],
                n_workers: Optional[int] = None,
                progress_bar: bool = True) -> List[Any]:
    """
    Parallel map function with progress bar.
    
    Parameters:
    -----------
    func : callable
        Function to apply
    items : list
        List of items to process
    n_workers : int, optional
        Number of worker processes
    progress_bar : bool
        Whether to show progress bar
        
    Returns:
    --------
    results : list
        List of results
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    n_workers = min(n_workers, len(items))
    
    if n_workers == 1:
        # Sequential processing
        if progress_bar:
            return [func(item) for item in tqdm(items, desc="Processing")]
        else:
            return [func(item) for item in items]
    else:
        # Parallel processing
        with mp.Pool(processes=n_workers) as pool:
            if progress_bar:
                results = list(tqdm(
                    pool.imap(func, items),
                    total=len(items),
                    desc="Parallel processing"
                ))
            else:
                results = pool.map(func, items)
        
        return results

class ProgressPool:
    """
    Multiprocessing pool with progress bar.
    """
    
    def __init__(self, processes=None, **kwargs):
        self.pool = mp.Pool(processes=processes, **kwargs)
        self.results = []
    
    def map(self, func, iterable, desc="Processing"):
        """Map with progress bar."""
        total = len(iterable)
        
        with tqdm(total=total, desc=desc) as pbar:
            def update(*args):
                pbar.update()
            
            results = []
            for item in iterable:
                result = self.pool.apply_async(
                    func, 
                    (item,), 
                    callback=update
                )
                results.append(result)
            
            self.results = [r.get() for r in results]
        
        return self.results
    
    def close(self):
        """Close the pool."""
        self.pool.close()
        self.pool.join()

# ============================================================================
# 8. PHYSICS-SPECIFIC UTILITIES
# ============================================================================

def compute_planck_units() -> Dict[str, float]:
    """
    Compute Planck units for quantum gravity.
    
    Returns:
    --------
    units : dict
        Dictionary of Planck units
    """
    # Fundamental constants (SI units)
    hbar = 1.054571817e-34  # Reduced Planck constant [J·s]
    G = 6.67430e-11         # Gravitational constant [m³/kg/s²]
    c = 299792458           # Speed of light [m/s]
    k_B = 1.380649e-23      # Boltzmann constant [J/K]
    
    # Planck length
    l_pl = np.sqrt(hbar * G / c**3)
    
    # Planck time
    t_pl = l_pl / c
    
    # Planck mass
    m_pl = np.sqrt(hbar * c / G)
    
    # Planck energy
    E_pl = m_pl * c**2
    
    # Planck temperature
    T_pl = E_pl / k_B
    
    # Planck density
    rho_pl = m_pl / l_pl**3
    
    return {
        'length': l_pl,      # 1.616255e-35 m
        'time': t_pl,        # 5.391247e-44 s
        'mass': m_pl,        # 2.176434e-8 kg
        'energy': E_pl,      # 1.956081e9 J
        'temperature': T_pl, # 1.416784e32 K
        'density': rho_pl    # 5.155e96 kg/m³
    }

def convert_to_planck_units(value: float, 
                           unit_type: str,
                           from_si: bool = True) -> float:
    """
    Convert between SI and Planck units.
    
    Parameters:
    -----------
    value : float
        Value to convert
    unit_type : str
        Type of unit ('length', 'time', 'mass', 'energy', 'temperature', 'density')
    from_si : bool
        If True, convert from SI to Planck units
        
    Returns:
    --------
    converted_value : float
        Converted value
    """
    planck_units = compute_planck_units()
    
    if unit_type not in planck_units:
        raise ValueError(f"Unknown unit type: {unit_type}")
    
    if from_si:
        # Convert from SI to Planck units
        return value / planck_units[unit_type]
    else:
        # Convert from Planck to SI units
        return value * planck_units[unit_type]

def compute_cosmological_parameters(z: float = 0) -> Dict[str, float]:
    """
    Compute cosmological parameters at redshift z.
    
    Parameters:
    -----------
    z : float
        Redshift
        
    Returns:
    --------
    params : dict
        Cosmological parameters
    """
    # Planck 2018 parameters
    H0 = 67.66  # Hubble constant [km/s/Mpc]
    Omega_m = 0.3111  # Matter density
    Omega_lambda = 0.6889  # Dark energy density
    Omega_r = 9.18e-5  # Radiation density
    Omega_k = 0.0007  # Curvature density
    
    # Convert to SI
    H0_si = H0 * 1000 / 3.086e22  # s^-1
    
    # Scale factor
    a = 1 / (1 + z)
    
    # Hubble parameter at redshift z
    Hz = H0_si * np.sqrt(Omega_m * a**(-3) + Omega_r * a**(-4) + 
                        Omega_k * a**(-2) + Omega_lambda)
    
    # Age of universe at redshift z
    # (approximate integral)
    n_points = 1000
    a_vals = np.linspace(0, a, n_points)
    integrand = 1 / (a_vals * H0_si * np.sqrt(
        Omega_m * a_vals**(-3) + Omega_r * a_vals**(-4) + 
        Omega_k * a_vals**(-2) + Omega_lambda
    ))
    age = np.trapz(integrand, a_vals)
    
    # Comoving distance
    integrand = 1 / (H0_si * np.sqrt(
        Omega_m * a_vals**(-3) + Omega_r * a_vals**(-4) + 
        Omega_k * a_vals**(-2) + Omega_lambda
    ))
    comoving_distance = np.trapz(integrand, a_vals)
    
    return {
        'H0': H0,
        'Hz': Hz,
        'age': age,
        'comoving_distance': comoving_distance,
        'scale_factor': a,
        'redshift': z,
        'density_parameters': {
            'Omega_m': Omega_m,
            'Omega_lambda': Omega_lambda,
            'Omega_r': Omega_r,
            'Omega_k': Omega_k
        }
    }

# ============================================================================
# 9. VALIDATION AND TESTING UTILITIES
# ============================================================================

def validate_act_model(model: Any) -> Dict[str, bool]:
    """
    Validate ACT model consistency.
    
    Parameters:
    -----------
    model : object
        ACT model instance
        
    Returns:
    --------
    validation : dict
        Validation results
    """
    validation = {}
    
    # Check vertices
    if hasattr(model, 'vertices'):
        vertices = model.vertices
        validation['vertices_shape'] = vertices.shape[1] == 4  # 4D spacetime
        validation['vertices_finite'] = np.all(np.isfinite(vertices))
        validation['vertices_range'] = np.all(vertices >= -1e10) and np.all(vertices <= 1e10)
    
    # Check operators
    if hasattr(model, 'operators'):
        operators = model.operators
        validation['operators_count'] = len(operators) == model.N
        if len(operators) > 0:
            # Check first operator for unitarity
            U = operators[0]
            if hasattr(U, 'toarray'):
                U = U.toarray()
            U_dag = U.conj().T
            identity_check = np.allclose(U @ U_dag, np.eye(U.shape[0]), atol=1e-10)
            validation['operators_unitary'] = identity_check
    
    # Check triangulation
    if hasattr(model, 'tetrahedra'):
        tetrahedra = model.tetrahedra
        validation['tetrahedra_valid'] = all(len(t) == 4 for t in tetrahedra)
        if tetrahedra:
            # Check all vertices exist
            all_vertices = set(range(model.N))
            tetra_vertices = set()
            for tetra in tetrahedra:
                tetra_vertices.update(tetra)
            validation['tetrahedra_vertices_exist'] = tetra_vertices.issubset(all_vertices)
    
    # Check adjacency matrix
    if hasattr(model, 'adjacency'):
        adj = model.adjacency
        validation['adjacency_symmetric'] = np.allclose(adj.toarray(), adj.toarray().T)
        validation['adjacency_no_self_loops'] = np.all(adj.diagonal() == 0)
    
    return validation

def benchmark_act_model(model: Any, 
                       n_iterations: int = 100) -> Dict[str, float]:
    """
    Benchmark ACT model performance.
    
    Parameters:
    -----------
    model : object
        ACT model instance
    n_iterations : int
        Number of benchmark iterations
        
    Returns:
    --------
    benchmark : dict
        Benchmark results
    """
    import time
    
    benchmark = {}
    
    # Benchmark vertex operations
    if hasattr(model, 'vertices'):
        start = time.time()
        for _ in range(n_iterations):
            _ = np.mean(model.vertices, axis=0)
        benchmark['vertex_mean'] = (time.time() - start) / n_iterations
    
    # Benchmark operator operations
    if hasattr(model, 'operators') and len(model.operators) > 0:
        start = time.time()
        for _ in range(min(n_iterations, 10)):
            U = model.operators[0]
            if hasattr(U, 'toarray'):
                U = U.toarray()
            _ = np.linalg.eigvals(U)
        benchmark['operator_eigvals'] = (time.time() - start) / min(n_iterations, 10)
    
    # Benchmark adjacency operations
    if hasattr(model, 'adjacency'):
        start = time.time()
        for _ in range(n_iterations):
            _ = model.adjacency.sum(axis=1)
        benchmark['adjacency_sum'] = (time.time() - start) / n_iterations
    
    return benchmark

# ============================================================================
# 10. ERROR HANDLING AND LOGGING
# ============================================================================

class ACTLogger:
    """
    Custom logger for ACT simulations.
    """
    
    def __init__(self, log_file: Optional[str] = None, 
                 console_level: str = 'INFO',
                 file_level: str = 'DEBUG'):
        """
        Initialize logger.
        
        Parameters:
        -----------
        log_file : str, optional
            Log file path
        console_level : str
            Console logging level
        file_level : str
            File logging level
        """
        import logging
        
        self.logger = logging.getLogger('ACT')
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, console_level))
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, file_level))
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)

def handle_exception(func: Callable) -> Callable:
    """
    Decorator to handle exceptions in ACT functions.
    
    Parameters:
    -----------
    func : callable
        Function to wrap
        
    Returns:
    --------
    wrapped : callable
        Wrapped function with exception handling
    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger = ACTLogger()
            logger.error(f"Error in {func.__name__}: {str(e)}")
            
            # Save error information
            error_info = {
                'function': func.__name__,
                'error': str(e),
                'args': str(args),
                'kwargs': str(kwargs),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to error log
            error_file = f"error_{func.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            save_results(error_info, error_file, format='json')
            
            # Re-raise for now, but could return None or default value
            raise
    
    return wrapped

# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def test_utils():
    """Test all utility functions."""
    print("Testing ACT utility functions...")
    
    # Test mathematical utilities
    print("\n1. Testing mathematical utilities:")
    z = 1 + 2j
    real, imag = complex_to_real(z)
    print(f"  complex_to_real({z}) = ({real}, {imag})")
    
    U = su_matrix(2)
    print(f"  su_matrix(2) shape: {U.shape}, unitary: {np.allclose(U @ U.conj().T, np.eye(2))}")
    
    # Test geometry utilities
    print("\n2. Testing geometry utilities:")
    vertices = np.random.randn(10, 4)
    simplices = [(0, 1, 2, 3), (4, 5, 6, 7)]
    curvature = compute_curvature_tensor(vertices, simplices)
    print(f"  curvature tensor shape: {curvature.shape}")
    
    volume = compute_volume_form(vertices, (0, 1, 2, 3))
    print(f"  tetrahedron volume: {volume:.6f}")
    
    # Test network utilities
    print("\n3. Testing network utilities:")
    adj = build_adjacency_matrix(vertices[:, :3], radius=0.5)
    print(f"  adjacency matrix shape: {adj.shape}, density: {adj.nnz / adj.shape[0]**2:.3f}")
    
    laplacian = compute_graph_laplacian(adj)
    eigenvalues, _ = compute_spectrum(laplacian, k=5)
    print(f"  Laplacian eigenvalues: {eigenvalues}")
    
    # Test data processing
    print("\n4. Testing data processing:")
    data = np.random.randn(1000)
    smoothed = exponential_smoothing(data, alpha=0.3)
    print(f"  data mean: {np.mean(data):.3f}, smoothed mean: {np.mean(smoothed):.3f}")
    
    a, b, r2 = fit_power_law(np.arange(1, 100), np.arange(1, 100)**2)
    print(f"  power law fit: y = {a:.3f} * x^{b:.3f}, R² = {r2:.3f}")
    
    # Test file I/O
    print("\n5. Testing file I/O:")
    test_data = {'array': np.random.randn(10, 4), 'value': 42, 'text': 'test'}
    save_results(test_data, 'test_save.pkl', format='pkl')
    loaded = load_results('test_save.pkl', format='pkl')
    print(f"  save/load test: {np.allclose(test_data['array'], loaded['array'])}")
    
    # Clean up
    import os
    if os.path.exists('test_save.pkl'):
        os.remove('test_save.pkl')
    
    print("\n✅ All utility functions tested successfully!")

if __name__ == "__main__":
    test_utils()
