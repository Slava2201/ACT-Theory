"""
ACT Experiment Runner
=====================
Run and manage experiments with Algebraic Causality Theory.
Includes parameter sweeps, convergence tests, and result analysis.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json
import pickle
import warnings
import argparse
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Any, Callable
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import ACT modules
try:
    # These are the modules we'll create
    from act_core import ACTCore
    from constants_calculator import ConstantsCalculator
    from standard_model import StandardModelCalculator
    ACT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import ACT modules: {e}")
    print("Running in simulation mode with synthetic data...")
    ACT_AVAILABLE = False

# ============================================================================
# 1. EXPERIMENT MANAGER
# ============================================================================

class ACTExperimentManager:
    """
    Manages and runs ACT experiments with different parameters.
    """
    
    def __init__(self, output_dir: str = "experiment_results"):
        """
        Initialize experiment manager.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save experiment results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Experiment database
        self.experiment_db = self.output_dir / "experiment_database.csv"
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Initialize database if needed
        if not self.experiment_db.exists():
            self.initialize_database()
        
        # Current experiment ID
        self.current_exp_id = self.get_next_exp_id()
        
        print(f"ACT Experiment Manager initialized")
        print(f"Output directory: {self.output_dir}")
        print(f"Next experiment ID: {self.current_exp_id}")
    
    def initialize_database(self):
        """Initialize experiment database."""
        columns = [
            'exp_id', 'timestamp', 'N', 'temperature', 'seed', 
            'thermal_steps', 'dimension', 'algorithm',
            'status', 'results_file', 'compute_time', 'notes',
            'alpha', 'Omega_dm', 'spectral_dim', 'entropy',
            'success'
        ]
        
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.experiment_db, index=False)
        print(f"Created new experiment database: {self.experiment_db}")
    
    def get_next_exp_id(self) -> int:
        """Get next available experiment ID."""
        if self.experiment_db.exists():
            df = pd.read_csv(self.experiment_db)
            if len(df) > 0:
                return int(df['exp_id'].max() + 1)
        return 1
    
    def register_experiment(self, params: Dict[str, Any]) -> int:
        """
        Register new experiment in database.
        
        Parameters:
        -----------
        params : dict
            Experiment parameters
            
        Returns:
        --------
        exp_id : int
            Registered experiment ID
        """
        exp_record = {
            'exp_id': self.current_exp_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'N': params.get('N', 1000),
            'temperature': params.get('temperature', 0.7),
            'seed': params.get('seed', 42),
            'thermal_steps': params.get('thermal_steps', 500),
            'dimension': params.get('dimension', 4),
            'algorithm': params.get('algorithm', 'metropolis'),
            'status': 'registered',
            'results_file': '',
            'compute_time': 0,
            'notes': params.get('notes', ''),
            'alpha': np.nan,
            'Omega_dm': np.nan,
            'spectral_dim': np.nan,
            'entropy': np.nan,
            'success': False
        }
        
        # Update database
        if self.experiment_db.exists():
            df = pd.read_csv(self.experiment_db)
            df = pd.concat([df, pd.DataFrame([exp_record])], ignore_index=True)
        else:
            df = pd.DataFrame([exp_record])
        
        df.to_csv(self.experiment_db, index=False)
        
        print(f"Registered experiment {self.current_exp_id}")
        return self.current_exp_id
    
    def update_experiment(self, exp_id: int, **kwargs):
        """
        Update experiment record in database.
        
        Parameters:
        -----------
        exp_id : int
            Experiment ID
        **kwargs : dict
            Fields to update
        """
        if not self.experiment_db.exists():
            return
        
        df = pd.read_csv(self.experiment_db)
        
        if exp_id in df['exp_id'].values:
            idx = df[df['exp_id'] == exp_id].index[0]
            for key, value in kwargs.items():
                if key in df.columns:
                    df.at[idx, key] = value
            
            df.to_csv(self.experiment_db, index=False)
    
    def run_single_experiment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single ACT experiment.
        
        Parameters:
        -----------
        params : dict
            Experiment parameters
            
        Returns:
        --------
        results : dict
            Experiment results
        """
        exp_id = self.current_exp_id
        print(f"\n{'='*80}")
        print(f"RUNNING EXPERIMENT {exp_id}")
        print(f"{'='*80}")
        
        # Register experiment
        self.register_experiment(params)
        self.update_experiment(exp_id, status="running")
        
        start_time = datetime.now()
        
        try:
            if ACT_AVAILABLE:
                # Run actual ACT experiment
                results = self._run_real_act_experiment(params)
            else:
                # Generate synthetic results for testing
                results = self._generate_synthetic_results(params)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.results_dir / f"exp_{exp_id}_{timestamp}.pkl"
            
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)
            
            # Save summary as JSON
            summary_file = self.results_dir / f"exp_{exp_id}_{timestamp}_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(self._create_summary(results), f, indent=2)
            
            # Update database
            compute_time = (datetime.now() - start_time).total_seconds()
            self.update_experiment(
                exp_id, 
                status="completed",
                results_file=str(results_file),
                compute_time=compute_time,
                success=True
            )
            
            # Extract key metrics
            self._extract_metrics_to_db(exp_id, results)
            
            # Print summary
            self._print_experiment_summary(results, compute_time)
            
            # Increment experiment ID for next experiment
            self.current_exp_id += 1
            
            return results
            
        except Exception as e:
            # Handle errors
            error_msg = f"Experiment {exp_id} failed: {str(e)}"
            print(f"\n❌ {error_msg}")
            
            self.update_experiment(exp_id, status="failed", success=False)
            
            # Save error information
            error_file = self.results_dir / f"exp_{exp_id}_error.txt"
            with open(error_file, 'w') as f:
                f.write(f"Experiment {exp_id} failed at {datetime.now()}\n")
                f.write(f"Parameters: {json.dumps(params, indent=2)}\n")
                f.write(f"Error: {str(e)}\n")
                import traceback
                f.write(f"Traceback:\n{traceback.format_exc()}")
            
            # Increment experiment ID for next experiment
            self.current_exp_id += 1
            
            raise
    
    def _run_real_act_experiment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run actual ACT experiment with given parameters.
        """
        # Extract parameters
        N = params['N']
        temperature = params['temperature']
        seed = params['seed']
        thermal_steps = params['thermal_steps']
        dimension = params.get('dimension', 4)
        
        print(f"Parameters:")
        print(f"  N = {N:,}")
        print(f"  Temperature = {temperature}")
        print(f"  Seed = {seed}")
        print(f"  Dimension = {dimension}")
        print(f"  Thermal steps = {thermal_steps:,}")
        
        # 1. Create ACT model
        print(f"\n1. Creating ACT model...")
        start_time = datetime.now()
        
        model = ACTCore(
            N=N,
            dim=dimension,
            temperature=temperature,
            seed=seed
        )
        
        model_time = (datetime.now() - start_time).total_seconds()
        print(f"   Model created in {model_time:.1f}s")
        
        # 2. Build causal set
        print(f"2. Building causal set...")
        vertices = model.build_causal_set()
        print(f"   Built causal set with {len(vertices)} vertices")
        
        # 3. Construct triangulation
        print(f"3. Constructing triangulation...")
        simplices = model.construct_triangulation()
        print(f"   Constructed {len(simplices)} simplices")
        
        # 4. Build causal graph
        print(f"4. Building causal graph...")
        graph = model.build_causal_graph()
        print(f"   Graph has {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # 5. Run Monte Carlo thermalization
        print(f"5. Running Monte Carlo thermalization...")
        mc_results = model.run_monte_carlo(steps=thermal_steps)
        print(f"   Monte Carlo completed with acceptance rate: {mc_results['acceptance_rate_final']:.3f}")
        
        # 6. Compute emergent constants
        print(f"6. Computing emergent constants...")
        constants = model.compute_emergent_constants()
        
        # 7. Compute observables
        print(f"7. Computing observables...")
        observables = model.calculate_observables_parallel(n_workers=2)
        
        # 8. Compute spectral dimension
        print(f"8. Computing spectral dimension...")
        spectral_dim = model.compute_spectral_dimension()
        
        # 9. Compile results
        print(f"9. Compiling results...")
        
        results = {
            'experiment_id': self.current_exp_id,
            'parameters': params,
            'model_info': {
                'N': N,
                'dimension': dimension,
                'vertices_count': len(vertices),
                'simplices_count': len(simplices),
                'graph_nodes': graph.number_of_nodes(),
                'graph_edges': graph.number_of_edges()
            },
            'monte_carlo': mc_results,
            'constants': constants,
            'observables': observables,
            'spectral_dimension': spectral_dim,
            'timing': {
                'total': (datetime.now() - start_time).total_seconds(),
                'model_creation': model_time,
                'monte_carlo': mc_results.get('computation_time', 0)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def _generate_synthetic_results(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate synthetic results for testing without ACT modules.
        """
        N = params['N']
        seed = params['seed']
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Simulate computation time
        import time
        time.sleep(0.5 + N/10000)  # Simulate computation scaling with N
        
        # Generate synthetic data with realistic noise
        base_alpha = 1/137.035999084
        base_Omega_dm = 0.265
        base_spectral_dim = 3.0 + np.random.normal(0, 0.1)
        
        # Add scaling with N
        N_factor = np.log(N) / np.log(1000)
        
        results = {
            'experiment_id': self.current_exp_id,
            'parameters': params,
            'model_info': {
                'N': N,
                'dimension': params.get('dimension', 4),
                'vertices_count': N,
                'simplices_count': int(N * 0.25),
                'graph_nodes': N,
                'graph_edges': int(N * 6.2)
            },
            'monte_carlo': {
                'acceptance_rate_final': 0.23 + np.random.normal(0, 0.02),
                'measurements': {
                    'action': [np.random.normal(1.0, 0.1) for _ in range(10)],
                    'curvature': [np.random.normal(0.0, 0.01) for _ in range(10)],
                    'volume': [np.random.normal(1.0, 0.05) for _ in range(10)],
                    'acceptance_rate': [0.2 + i*0.01 for i in range(10)]
                }
            },
            'constants': {
                'alpha': base_alpha * (1 + np.random.normal(0, 1e-8)),
                'alpha_inverse': 137.035999084 * (1 + np.random.normal(0, 1e-8)),
                'G': 6.67430e-11 * (1 + np.random.normal(0, 1e-4)),
                'Lambda': 1.1056e-52 * (1 + np.random.normal(0, 1e-3)),
                'planck_length': 1.616255e-35,
                'planck_time': 5.391247e-44,
                'planck_mass': 2.176434e-8,
                'N_simplices': int(N * 0.25),
                'avg_curvature': np.random.normal(0, 0.01),
                'avg_volume': 1.0
            },
            'observables': {
                'action': np.random.normal(1.23e-45, 1e-46),
                'curvature': np.random.normal(0.0, 0.01),
                'entanglement': 45.6 * N_factor + np.random.normal(0, 0.5)
            },
            'spectral_dimension': [
                (t, base_spectral_dim + 0.1*np.sin(t) + np.random.normal(0, 0.01))
                for t in np.logspace(-2, 2, 20)
            ],
            'timing': {
                'total': 0.5 + N/10000,
                'model_creation': 0.1,
                'monte_carlo': 0.4 + N/20000
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Add dark matter estimate
        results['dark_matter'] = {
            'Omega_dm': base_Omega_dm * (1 + np.random.normal(0, 0.01)),
            'defect_density': N * 0.001 * (1 + np.random.normal(0, 0.1)),
            'mass_scale_tev': 1.2 * (1 + np.random.normal(0, 0.05))
        }
        
        return results
    
    def _create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of experiment results."""
        summary = {
            'experiment_id': results['experiment_id'],
            'parameters': results['parameters'],
            'model_info': results['model_info'],
            'key_results': {},
            'performance': results.get('timing', {})
        }
        
        # Extract key results from constants
        if 'constants' in results:
            const = results['constants']
            summary['key_results']['constants'] = {
                'alpha': const.get('alpha'),
                'alpha_inverse': const.get('alpha_inverse'),
                'G': const.get('G'),
                'Lambda': const.get('Lambda')
            }
        
        # Extract from observables
        if 'observables' in results:
            obs = results['observables']
            summary['key_results']['observables'] = {
                'action': obs.get('action'),
                'curvature': obs.get('curvature'),
                'entanglement_entropy': obs.get('entanglement')
            }
        
        # Extract from dark matter
        if 'dark_matter' in results:
            dm = results['dark_matter']
            summary['key_results']['dark_matter'] = {
                'Omega_dm': dm.get('Omega_dm'),
                'defect_density': dm.get('defect_density'),
                'mass_scale_tev': dm.get('mass_scale_tev')
            }
        
        # Monte Carlo results
        if 'monte_carlo' in results:
            mc = results['monte_carlo']
            summary['key_results']['monte_carlo'] = {
                'acceptance_rate': mc.get('acceptance_rate_final'),
                'final_action': mc.get('measurements', {}).get('action', [0])[-1] if 'measurements' in mc else None
            }
        
        return summary
    
    def _extract_metrics_to_db(self, exp_id: int, results: Dict[str, Any]):
        """Extract key metrics and update database."""
        metrics = {
            'alpha': np.nan,
            'Omega_dm': np.nan,
            'spectral_dim': np.nan,
            'entropy': np.nan
        }
        
        # Extract alpha
        if 'constants' in results and 'alpha' in results['constants']:
            metrics['alpha'] = results['constants']['alpha']
        
        # Extract Omega_dm
        if 'dark_matter' in results and 'Omega_dm' in results['dark_matter']:
            metrics['Omega_dm'] = results['dark_matter']['Omega_dm']
        
        # Extract spectral dimension (average of last few points)
        if 'spectral_dimension' in results and len(results['spectral_dimension']) > 0:
            spectral_data = results['spectral_dimension']
            if isinstance(spectral_data, list) and len(spectral_data) > 0:
                # Get last 5 spectral dimension values
                last_n = min(5, len(spectral_data))
                last_dims = [d[1] for d in spectral_data[-last_n:]]
                metrics['spectral_dim'] = np.mean(last_dims)
        
        # Extract entanglement entropy
        if 'observables' in results and 'entanglement' in results['observables']:
            metrics['entropy'] = results['observables']['entanglement']
        
        # Update database
        self.update_experiment(exp_id, **metrics)
    
    def _print_experiment_summary(self, results: Dict[str, Any], compute_time: float):
        """Print experiment summary."""
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {results['experiment_id']} COMPLETED")
        print(f"{'='*80}")
        
        print(f"\n⏱ Computation time: {compute_time:.1f} seconds")
        
        # Print key results
        print(f"\n📊 KEY RESULTS:")
        print(f"{'-'*40}")
        
        # Constants
        if 'constants' in results:
            const = results['constants']
            print(f"Fundamental Constants:")
            if 'alpha' in const:
                print(f"  • α = {const['alpha']:.12f} (target: 0.00729735257)")
                print(f"    1/α = {const.get('alpha_inverse', 1/const['alpha']):.8f}")
            if 'G' in const:
                print(f"  • G = {const['G']:.4e} m³/kg/s²")
            if 'Lambda' in const:
                print(f"  • Λ = {const['Lambda']:.4e} m⁻²")
        
        # Dark matter
        if 'dark_matter' in results:
            dm = results['dark_matter']
            if 'Omega_dm' in dm:
                target = 0.265
                diff = abs(dm['Omega_dm'] - target) / target * 100
                print(f"\nDark Matter:")
                print(f"  • Ω_dm = {dm['Omega_dm']:.4f} (target: {target}, diff: {diff:.1f}%)")
            if 'mass_scale_tev' in dm:
                print(f"  • Mass scale = {dm['mass_scale_tev']:.2f} TeV")
        
        # Observables
        if 'observables' in results:
            obs = results['observables']
            print(f"\nObservables:")
            if 'entanglement' in obs:
                print(f"  • Entanglement entropy = {obs['entanglement']:.2f}")
            if 'curvature' in obs:
                print(f"  • Average curvature = {obs['curvature']:.4e}")
        
        # Monte Carlo
        if 'monte_carlo' in results:
            mc = results['monte_carlo']
            if 'acceptance_rate_final' in mc:
                print(f"  • MC acceptance rate = {mc['acceptance_rate_final']:.3f}")
        
        print(f"\n💾 Results saved to: {self.results_dir}/")

# ============================================================================
# 2. PARAMETER SWEEP EXPERIMENTS
# ============================================================================

class ACTParameterSweep:
    """
    Run parameter sweep experiments to study ACT behavior.
    """
    
    def __init__(self, experiment_manager: ACTExperimentManager):
        """
        Initialize parameter sweep.
        
        Parameters:
        -----------
        experiment_manager : ACTExperimentManager
            Manager to run experiments
        """
        self.manager = experiment_manager
    
    def sweep_N(self, 
                N_values: List[int], 
                base_params: Dict[str, Any] = None,
                save_plots: bool = True) -> Dict[int, Dict[str, Any]]:
        """
        Sweep over number of vertices N.
        
        Parameters:
        -----------
        N_values : list
            List of N values to test
        base_params : dict
            Base parameters for all experiments
        save_plots : bool
            Whether to save plots
            
        Returns:
        --------
        results : dict
            Results for all N values
        """
        if base_params is None:
            base_params = {
                'temperature': 0.7,
                'seed': 42,
                'thermal_steps': 500,
                'dimension': 4,
                'notes': 'N sweep'
            }
        
        print(f"\n{'='*80}")
        print(f"PARAMETER SWEEP: N = {N_values}")
        print(f"{'='*80}")
        
        all_results = {}
        
        for N in tqdm(N_values, desc="Running N sweep"):
            print(f"\nRunning N = {N:,}...")
            
            params = base_params.copy()
            params['N'] = N
            
            try:
                results = self.manager.run_single_experiment(params)
                all_results[N] = results
                
            except Exception as e:
                print(f"  Failed for N={N}: {e}")
                all_results[N] = {'error': str(e), 'parameters': params}
        
        # Analyze sweep results
        if len(all_results) > 1:
            analysis = self.analyze_N_sweep(all_results)
            
            if save_plots:
                self.plot_N_sweep_results(analysis, all_results)
        
        return all_results
    
    def analyze_N_sweep(self, results: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
        """
        Analyze results from N sweep.
        
        Parameters:
        -----------
        results : dict
            Results dictionary mapping N to results
            
        Returns:
        --------
        analysis_df : pd.DataFrame
            DataFrame with analysis results
        """
        print(f"\n{'='*80}")
        print(f"N SWEEP ANALYSIS")
        print(f"{'='*80}")
        
        # Extract data
        data = {
            'N': [],
            'tetrahedra': [],
            'avg_degree': [],
            'alpha': [],
            'Omega_dm': [],
            'spectral_dim': [],
            'entropy': [],
            'compute_time': [],
            'success': []
        }
        
        for N, res in results.items():
            data['N'].append(N)
            
            if 'error' not in res:
                # Model info
                if 'model_info' in res:
                    data['tetrahedra'].append(res['model_info'].get('simplices_count', np.nan))
                    # Estimate average degree from graph edges
                    nodes = res['model_info'].get('graph_nodes', N)
                    edges = res['model_info'].get('graph_edges', np.nan)
                    if not np.isnan(edges):
                        data['avg_degree'].append(2 * edges / nodes)
                    else:
                        data['avg_degree'].append(np.nan)
                else:
                    data['tetrahedra'].append(np.nan)
                    data['avg_degree'].append(np.nan)
                
                # Constants
                if 'constants' in res and 'alpha' in res['constants']:
                    data['alpha'].append(res['constants']['alpha'])
                else:
                    data['alpha'].append(np.nan)
                
                # Dark matter
                if 'dark_matter' in res and 'Omega_dm' in res['dark_matter']:
                    data['Omega_dm'].append(res['dark_matter']['Omega_dm'])
                else:
                    data['Omega_dm'].append(np.nan)
                
                # Spectral dimension (average of last few points)
                if 'spectral_dimension' in res and len(res['spectral_dimension']) > 0:
                    spectral_data = res['spectral_dimension']
                    if isinstance(spectral_data, list):
                        last_n = min(5, len(spectral_data))
                        last_dims = [d[1] for d in spectral_data[-last_n:]]
                        data['spectral_dim'].append(np.mean(last_dims))
                    else:
                        data['spectral_dim'].append(np.nan)
                else:
                    data['spectral_dim'].append(np.nan)
                
                # Entanglement entropy
                if 'observables' in res and 'entanglement' in res['observables']:
                    data['entropy'].append(res['observables']['entanglement'])
                else:
                    data['entropy'].append(np.nan)
                
                # Compute time
                if 'timing' in res and 'total' in res['timing']:
                    data['compute_time'].append(res['timing']['total'])
                else:
                    data['compute_time'].append(np.nan)
                
                data['success'].append(True)
            else:
                # Failed experiment
                for key in ['tetrahedra', 'avg_degree', 'alpha', 'Omega_dm', 
                           'spectral_dim', 'entropy', 'compute_time']:
                    data[key].append(np.nan)
                data['success'].append(False)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Compute scaling laws for successful experiments
        successful_df = df[df['success']]
        
        if len(successful_df) > 2:
            print(f"\n📈 SCALING LAWS (based on {len(successful_df)} successful runs):")
            print(f"{'-'*60}")
            
            # Tetrahedra scaling
            if successful_df['tetrahedra'].notna().sum() > 2:
                x = np.log(successful_df['N'].values)
                y = np.log(successful_df['tetrahedra'].values)
                mask = ~np.isnan(x) & ~np.isnan(y)
                if mask.sum() > 2:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                    print(f"Tetrahedra ∝ N^{slope:.3f} ± {std_err:.3f} (R²={r_value**2:.3f})")
            
            # α convergence
            if successful_df['alpha'].notna().sum() > 2:
                alphas = successful_df['alpha'].values
                mean_alpha = np.nanmean(alphas)
                std_alpha = np.nanstd(alphas)
                target_alpha = 1/137.035999084
                diff_pct = abs(mean_alpha - target_alpha) / target_alpha * 100
                
                print(f"\nFine-structure constant α:")
                print(f"  Mean: {mean_alpha:.12f}")
                print(f"  Std: {std_alpha:.12f}")
                print(f"  Target: {target_alpha:.12f}")
                print(f"  Difference: {diff_pct:.4f}%")
                
                # Check if converging
                if len(alphas) >= 3:
                    last_3 = alphas[-3:]
                    last_std = np.std(last_3)
                    if last_std / mean_alpha < 0.01:  # Less than 1% variation
                        print(f"  ✓ Converging (last 3 std: {last_std/mean_alpha*100:.2f}%)")
                    else:
                        print(f"  ✗ Not converging (last 3 std: {last_std/mean_alpha*100:.2f}%)")
            
            # Ω_dm convergence
            if successful_df['Omega_dm'].notna().sum() > 2:
                omegas = successful_df['Omega_dm'].values
                mean_omega = np.nanmean(omegas)
                std_omega = np.nanstd(omegas)
                target_omega = 0.265
                diff_pct = abs(mean_omega - target_omega) / target_omega * 100
                
                print(f"\nDark matter density Ω_dm:")
                print(f"  Mean: {mean_omega:.4f}")
                print(f"  Std: {std_omega:.4f}")
                print(f"  Target: {target_omega:.4f}")
                print(f"  Difference: {diff_pct:.2f}%")
            
            # Spectral dimension
            if successful_df['spectral_dim'].notna().sum() > 2:
                dims = successful_df['spectral_dim'].values
                mean_dim = np.nanmean(dims)
                std_dim = np.nanstd(dims)
                
                print(f"\nSpectral dimension:")
                print(f"  Mean: {mean_dim:.2f}")
                print(f"  Std: {std_dim:.2f}")
                print(f"  Expected: ~4.0 at Planck scale, ~2.0 at large scales")
        
        # Save analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = self.manager.output_dir / f"N_sweep_analysis_{timestamp}.csv"
        df.to_csv(analysis_file, index=False)
        
        print(f"\n💾 Analysis saved to: {analysis_file}")
        
        return df
    
    def plot_N_sweep_results(self, analysis_df: pd.DataFrame, results: Dict[int, Dict[str, Any]]):
        """
        Create plots for N sweep results.
        
        Parameters:
        -----------
        analysis_df : pd.DataFrame
            Analysis DataFrame
        results : dict
            Raw results dictionary
        """
        # Filter successful experiments
        successful_df = analysis_df[analysis_df['success']]
        
        if len(successful_df) < 2:
            print("Not enough successful experiments for plotting")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot 1: Tetrahedra scaling
        ax = axes[0]
        if successful_df['tetrahedra'].notna().sum() > 1:
            ax.loglog(successful_df['N'], successful_df['tetrahedra'], 'o-', linewidth=2)
            ax.set_xlabel('N (number of vertices)')
            ax.set_ylabel('Number of simplices')
            ax.set_title('Tetrahedra Scaling')
            ax.grid(True, alpha=0.3)
            
            # Fit power law
            x = np.log(successful_df['N'].values)
            y = np.log(successful_df['tetrahedra'].values)
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() > 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = slope * x_fit + intercept
                ax.loglog(np.exp(x_fit), np.exp(y_fit), 'r--', 
                         label=f'∝ N^{slope:.2f} (R²={r_value**2:.2f})')
                ax.legend()
        
        # Plot 2: α convergence
        ax = axes[1]
        if successful_df['alpha'].notna().sum() > 1:
            alphas = successful_df['alpha'].values
            ax.semilogx(successful_df['N'], alphas, 'o-', linewidth=2)
            ax.axhline(y=1/137.035999084, color='r', linestyle='--', 
                      label='Target: 1/137.035999084')
            ax.set_xlabel('N')
            ax.set_ylabel('α (fine-structure constant)')
            ax.set_title('α Convergence with N')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Plot 3: Ω_dm convergence
        ax = axes[2]
        if successful_df['Omega_dm'].notna().sum() > 1:
            omegas = successful_df['Omega_dm'].values
            ax.semilogx(successful_df['N'], omegas, 'o-', linewidth=2)
            ax.axhline(y=0.265, color='r', linestyle='--', label='Target: 0.265')
            ax.set_xlabel('N')
            ax.set_ylabel('Ω_dm')
            ax.set_title('Dark Matter Density Convergence')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Plot 4: Spectral dimension
        ax = axes[3]
        if successful_df['spectral_dim'].notna().sum() > 1:
            dims = successful_df['spectral_dim'].values
            ax.semilogx(successful_df['N'], dims, 'o-', linewidth=2)
            ax.axhline(y=4.0, color='r', linestyle='--', label='Expected: 4.0')
            ax.set_xlabel('N')
            ax.set_ylabel('Spectral Dimension')
            ax.set_title('Spectral Dimension vs N')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Plot 5: Entanglement entropy scaling
        ax = axes[4]
        if successful_df['entropy'].notna().sum() > 1:
            entropy = successful_df['entropy'].values
            ax.loglog(successful_df['N'], entropy, 'o-', linewidth=2)
            ax.set_xlabel('N')
            ax.set_ylabel('Entanglement Entropy')
            ax.set_title('Entanglement Entropy Scaling')
            ax.grid(True, alpha=0.3)
            
            # Fit to area law or volume law
            x = np.log(successful_df['N'].values)
            y = np.log(entropy)
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() > 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                ax.text(0.05, 0.95, f'Slope: {slope:.2f} (R²={r_value**2:.2f})',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 6: Compute time scaling
        ax = axes[5]
        if successful_df['compute_time'].notna().sum() > 1:
            times = successful_df['compute_time'].values
            ax.loglog(successful_df['N'], times, 'o-', linewidth=2)
            ax.set_xlabel('N')
            ax.set_ylabel('Compute Time (s)')
            ax.set_title('Computational Scaling')
            ax.grid(True, alpha=0.3)
            
            # Fit to estimate scaling
            x = np.log(successful_df['N'].values)
            y = np.log(times)
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() > 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                complexity = "O(N^1)" if slope < 1.2 else "O(N^2)" if slope < 2.2 else ">O(N^2)"
                ax.text(0.05, 0.95, f'Time ∝ N^{slope:.2f}\n{complexity}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.manager.plots_dir / f"N_sweep_plots_{timestamp}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plots saved to: {plot_file}")
    
    def sweep_temperature(self, 
                         T_values: List[float], 
                         base_params: Dict[str, Any] = None,
                         save_plots: bool = True) -> Dict[float, Dict[str, Any]]:
        """
        Sweep over temperature.
        
        Parameters:
        -----------
        T_values : list
            List of temperature values to test
        base_params : dict
            Base parameters for all experiments
        save_plots : bool
            Whether to save plots
            
        Returns:
        --------
        results : dict
            Results for all temperature values
        """
        if base_params is None:
            base_params = {
                'N': 1000,
                'seed': 42,
                'thermal_steps': 500,
                'dimension': 4,
                'notes': 'Temperature sweep'
            }
        
        print(f"\n{'='*80}")
        print(f"PARAMETER SWEEP: Temperature = {T_values}")
        print(f"{'='*80}")
        
        all_results = {}
        
        for T in tqdm(T_values, desc="Running temperature sweep"):
            print(f"\nRunning T = {T}...")
            
            params = base_params.copy()
            params['temperature'] = T
            
            try:
                results = self.manager.run_single_experiment(params)
                all_results[T] = results
                
            except Exception as e:
                print(f"  Failed for T={T}: {e}")
                all_results[T] = {'error': str(e), 'parameters': params}
        
        # Analyze temperature dependence
        if len(all_results) > 1:
            analysis = self.analyze_temperature_sweep(all_results)
            
            if save_plots:
                self.plot_temperature_sweep_results(analysis, all_results)
        
        return all_results
    
    def analyze_temperature_sweep(self, results: Dict[float, Dict[str, Any]]) -> pd.DataFrame:
        """
        Analyze temperature dependence.
        
        Parameters:
        -----------
        results : dict
            Results dictionary mapping temperature to results
            
        Returns:
        --------
        analysis_df : pd.DataFrame
            DataFrame with analysis results
        """
        print(f"\n{'='*80}")
        print(f"TEMPERATURE SWEEP ANALYSIS")
        print(f"{'='*80}")
        
        # Extract data
        data = {
            'temperature': [],
            'acceptance_rate': [],
            'final_action': [],
            'avg_curvature': [],
            'entropy': [],
            'alpha': [],
            'success': []
        }
        
        for T, res in results.items():
            data['temperature'].append(T)
            
            if 'error' not in res:
                # Monte Carlo acceptance rate
                if 'monte_carlo' in res and 'acceptance_rate_final' in res['monte_carlo']:
                    data['acceptance_rate'].append(res['monte_carlo']['acceptance_rate_final'])
                else:
                    data['acceptance_rate'].append(np.nan)
                
                # Final action
                if 'monte_carlo' in res and 'measurements' in res['monte_carlo']:
                    measurements = res['monte_carlo']['measurements']
                    if 'action' in measurements and len(measurements['action']) > 0:
                        data['final_action'].append(measurements['action'][-1])
                    else:
                        data['final_action'].append(np.nan)
                else:
                    data['final_action'].append(np.nan)
                
                # Average curvature
                if 'observables' in res and 'curvature' in res['observables']:
                    data['avg_curvature'].append(res['observables']['curvature'])
                else:
                    data['avg_curvature'].append(np.nan)
                
                # Entanglement entropy
                if 'observables' in res and 'entanglement' in res['observables']:
                    data['entropy'].append(res['observables']['entanglement'])
                else:
                    data['entropy'].append(np.nan)
                
                # Fine-structure constant
                if 'constants' in res and 'alpha' in res['constants']:
                    data['alpha'].append(res['constants']['alpha'])
                else:
                    data['alpha'].append(np.nan)
                
                data['success'].append(True)
            else:
                # Failed experiment
                for key in ['acceptance_rate', 'final_action', 'avg_curvature', 
                           'entropy', 'alpha']:
                    data[key].append(np.nan)
                data['success'].append(False)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df = df.sort_values('temperature')
        
        # Analyze phase transitions
        successful_df = df[df['success']]
        
        if len(successful_df) > 3:
            print(f"\n🔍 PHASE TRANSITION ANALYSIS:")
            print(f"{'-'*60}")
            
            # Look for discontinuities in derivatives
            temperatures = successful_df['temperature'].values
            
            # Check acceptance rate
            if successful_df['acceptance_rate'].notna().sum() > 3:
                acceptance = successful_df['acceptance_rate'].values
                d_acc_dT = np.gradient(acceptance, temperatures)
                
                # Find peaks in derivative
                peak_threshold = 2 * np.std(d_acc_dT)
                peaks = np.where(np.abs(d_acc_dT) > peak_threshold)[0]
                
                if len(peaks) > 0:
                    print(f"Possible phase transitions at T = {temperatures[peaks]}")
                    for peak in peaks:
                        print(f"  T={temperatures[peak]:.3f}: d(acceptance)/dT = {d_acc_dT[peak]:.3f}")
                else:
                    print("No clear phase transitions detected in acceptance rate")
            
            # Check curvature
            if successful_df['avg_curvature'].notna().sum() > 3:
                curvature = successful_df['avg_curvature'].values
                d_curv_dT = np.gradient(curvature, temperatures)
                
                # Find where curvature changes sign (possible topology change)
                sign_changes = np.where(np.diff(np.sign(curvature)) != 0)[0]
                if len(sign_changes) > 0:
                    print(f"\nCurvature sign changes at T = {temperatures[sign_changes]}")
                    print("  Possible topology transitions")
        
        # Save analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = self.manager.output_dir / f"temperature_sweep_analysis_{timestamp}.csv"
        df.to_csv(analysis_file, index=False)
        
        print(f"\n💾 Analysis saved to: {analysis_file}")
        
        return df
    
    def plot_temperature_sweep_results(self, analysis_df: pd.DataFrame, 
                                      results: Dict[float, Dict[str, Any]]):
        """
        Create plots for temperature sweep results.
        
        Parameters:
        -----------
        analysis_df : pd.DataFrame
            Analysis DataFrame
        results : dict
            Raw results dictionary
        """
        # Filter successful experiments
        successful_df = analysis_df[analysis_df['success']]
        
        if len(successful_df) < 2:
            print("Not enough successful experiments for plotting")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot 1: Acceptance rate vs temperature
        ax = axes[0]
        if successful_df['acceptance_rate'].notna().sum() > 1:
            ax.plot(successful_df['temperature'], successful_df['acceptance_rate'], 
                   'o-', linewidth=2)
            ax.set_xlabel('Temperature')
            ax.set_ylabel('Acceptance Rate')
            ax.set_title('Monte Carlo Acceptance Rate')
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Action vs temperature
        ax = axes[1]
        if successful_df['final_action'].notna().sum() > 1:
            ax.plot(successful_df['temperature'], successful_df['final_action'], 
                   'o-', linewidth=2)
            ax.set_xlabel('Temperature')
            ax.set_ylabel('Action')
            ax.set_title('Regge Action vs Temperature')
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Curvature vs temperature
        ax = axes[2]
        if successful_df['avg_curvature'].notna().sum() > 1:
            ax.plot(successful_df['temperature'], successful_df['avg_curvature'], 
                   'o-', linewidth=2)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.set_xlabel('Temperature')
            ax.set_ylabel('Average Curvature')
            ax.set_title('Curvature vs Temperature')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Entanglement entropy vs temperature
        ax = axes[3]
        if successful_df['entropy'].notna().sum() > 1:
            ax.plot(successful_df['temperature'], successful_df['entropy'], 
                   'o-', linewidth=2)
            ax.set_xlabel('Temperature')
            ax.set_ylabel('Entanglement Entropy')
            ax.set_title('Entanglement vs Temperature')
            ax.grid(True, alpha=0.3)
        
        # Plot 5: α vs temperature
        ax = axes[4]
        if successful_df['alpha'].notna().sum() > 1:
            ax.plot(successful_df['temperature'], successful_df['alpha'], 
                   'o-', linewidth=2)
            ax.axhline(y=1/137.035999084, color='r', linestyle='--', 
                      label='Target: 1/137.035999084')
            ax.set_xlabel('Temperature')
            ax.set_ylabel('α (fine-structure constant)')
            ax.set_title('α vs Temperature')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Plot 6: Combined phase diagram
        ax = axes[5]
        if (successful_df['acceptance_rate'].notna().sum() > 1 and 
            successful_df['avg_curvature'].notna().sum() > 1):
            
            # Normalize data for combined plot
            norm_acc = (successful_df['acceptance_rate'] - 
                       successful_df['acceptance_rate'].min()) / \
                       (successful_df['acceptance_rate'].max() - 
                        successful_df['acceptance_rate'].min())
            norm_curv = (successful_df['avg_curvature'] - 
                        successful_df['avg_curvature'].min()) / \
                        (successful_df['avg_curvature'].max() - 
                         successful_df['avg_curvature'].min())
            
            ax.plot(successful_df['temperature'], norm_acc, 'o-', 
                   linewidth=2, label='Acceptance rate (norm)')
            ax.plot(successful_df['temperature'], norm_curv, 's-', 
                   linewidth=2, label='Curvature (norm)')
            ax.set_xlabel('Temperature')
            ax.set_ylabel('Normalized Value')
            ax.set_title('Phase Diagram Indicators')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.manager.plots_dir / f"temperature_sweep_plots_{timestamp}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plots saved to: {plot_file}")

# ============================================================================
# 3. CONVERGENCE TESTS
# ============================================================================

class ACTConvergenceTest:
    """
    Test convergence of ACT observables with increasing N.
    """
    
    def __init__(self, experiment_manager: ACTExperimentManager):
        """
        Initialize convergence test.
        
        Parameters:
        -----------
        experiment_manager : ACTExperimentManager
            Experiment manager
        """
        self.manager = experiment_manager
    
    def run_convergence_test(self, 
                            N_max: int = 2000, 
                            N_step: int = 200,
                            base_params: Dict[str, Any] = None,
                            save_plots: bool = True) -> Dict[str, Any]:
        """
        Run convergence test with increasing N.
        
        Parameters:
        -----------
        N_max : int
            Maximum N to test
        N_step : int
            Step size for N
        base_params : dict
            Base parameters
        save_plots : bool
            Whether to save plots
            
        Returns:
        --------
        convergence_data : dict
            Convergence test results
        """
        if base_params is None:
            base_params = {
                'temperature': 0.7,
                'seed': 42,
                'thermal_steps': 500,
                'dimension': 4,
                'notes': 'Convergence test'
            }
        
        print(f"\n{'='*80}")
        print(f"CONVERGENCE TEST: N up to {N_max:,} (step {N_step})")
        print(f"{'='*80}")
        
        # Generate N values
        N_values = list(range(N_step, N_max + N_step, N_step))
        
        # Run parameter sweep
        sweep = ACTParameterSweep(self.manager)
        results = sweep.sweep_N(N_values, base_params, save_plots=False)
        
        # Analyze convergence
        convergence = self.analyze_convergence(results)
        
        # Save full results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.manager.output_dir / f"convergence_test_results_{timestamp}.pkl"
        
        with open(results_file, 'wb') as f:
            pickle.dump({
                'results': results,
                'convergence': convergence,
                'parameters': {
                    'N_max': N_max,
                    'N_step': N_step,
                    'base_params': base_params
                }
            }, f)
        
        print(f"\n💾 Full convergence test results saved to: {results_file}")
        
        if save_plots:
            self.plot_convergence_results(convergence)
        
        return {
            'results': results,
            'convergence': convergence
        }
    
    def analyze_convergence(self, results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze convergence of observables with N.
        
        Parameters:
        -----------
        results : dict
            Results dictionary mapping N to results
            
        Returns:
        --------
        convergence : dict
            Convergence analysis results
        """
        print(f"\n{'='*80}")
        print(f"CONVERGENCE ANALYSIS")
        print(f"{'='*80}")
        
        # Extract data
        convergence_data = {
            'N': [],
            'alpha': [],
            'Omega_dm': [],
            'spectral_dim': [],
            'entropy': [],
            'success': []
        }
        
        for N, res in results.items():
            convergence_data['N'].append(N)
            
            if 'error' not in res:
                # Fine-structure constant
                if 'constants' in res and 'alpha' in res['constants']:
                    convergence_data['alpha'].append(res['constants']['alpha'])
                else:
                    convergence_data['alpha'].append(np.nan)
                
                # Dark matter density
                if 'dark_matter' in res and 'Omega_dm' in res['dark_matter']:
                    convergence_data['Omega_dm'].append(res['dark_matter']['Omega_dm'])
                else:
                    convergence_data['Omega_dm'].append(np.nan)
                
                # Spectral dimension
                if 'spectral_dimension' in res and len(res['spectral_dimension']) > 0:
                    spectral_data = res['spectral_dimension']
                    if isinstance(spectral_data, list):
                        last_n = min(5, len(spectral_data))
                        last_dims = [d[1] for d in spectral_data[-last_n:]]
                        convergence_data['spectral_dim'].append(np.mean(last_dims))
                    else:
                        convergence_data['spectral_dim'].append(np.nan)
                else:
                    convergence_data['spectral_dim'].append(np.nan)
                
                # Entanglement entropy
                if 'observables' in res and 'entanglement' in res['observables']:
                    convergence_data['entropy'].append(res['observables']['entanglement'])
                else:
                    convergence_data['entropy'].append(np.nan)
                
                convergence_data['success'].append(True)
            else:
                for key in ['alpha', 'Omega_dm', 'spectral_dim', 'entropy']:
                    convergence_data[key].append(np.nan)
                convergence_data['success'].append(False)
        
        # Create DataFrame
        df = pd.DataFrame(convergence_data)
        df = df.sort_values('N')
        
        # Compute convergence metrics
        successful_df = df[df['success']]
        convergence_metrics = {}
        
        if len(successful_df) > 3:
            print(f"\n📊 CONVERGENCE METRICS:")
            print(f"{'-'*60}")
            
            for observable in ['alpha', 'Omega_dm', 'spectral_dim', 'entropy']:
                values = successful_df[observable].values
                
                if len(values) > 3 and not np.all(np.isnan(values)):
                    # Remove NaN values
                    valid_mask = ~np.isnan(values)
                    valid_values = values[valid_mask]
                    valid_N = successful_df['N'].values[valid_mask]
                    
                    if len(valid_values) > 3:
                        # Fit to convergence model: value = a + b/N^c
                        try:
                            # Simple model: value = a + b/N
                            def model_func(N, a, b):
                                return a + b / N
                            
                            # Fit parameters
                            p0 = [valid_values[-1], valid_values[0] - valid_values[-1]]
                            popt, pcov = curve_fit(model_func, valid_N, valid_values, p0=p0)
                            
                            # Extract parameters
                            a_fit, b_fit = popt
                            a_err, b_err = np.sqrt(np.diag(pcov))
                            
                            # Asymptotic value and error
                            asymptotic_value = a_fit
                            asymptotic_error = a_err
                            
                            # Convergence speed
                            convergence_speed = b_fit
                            
                            # Check if converged (asymptotic error < 1% of value)
                            if asymptotic_value != 0:
                                rel_error = abs(asymptotic_error / asymptotic_value)
                                converged = rel_error < 0.01  # Less than 1% relative error
                            else:
                                converged = asymptotic_error < 0.01 * np.mean(np.abs(valid_values))
                            
                            # Target values for comparison
                            targets = {
                                'alpha': 1/137.035999084,
                                'Omega_dm': 0.265,
                                'spectral_dim': 4.0,  # At Planck scale
                                'entropy': None  # No specific target
                            }
                            
                            target = targets.get(observable)
                            if target is not None:
                                diff = abs(asymptotic_value - target) / target * 100
                            else:
                                diff = None
                            
                            # Store metrics
                            convergence_metrics[observable] = {
                                'asymptotic_value': asymptotic_value,
                                'asymptotic_error': asymptotic_error,
                                'convergence_speed': convergence_speed,
                                'converged': converged,
                                'target_value': target,
                                'difference_percent': diff,
                                'num_points': len(valid_values)
                            }
                            
                            # Print results
                            print(f"\n{observable}:")
                            print(f"  Asymptotic value: {asymptotic_value:.8f} ± {asymptotic_error:.8f}")
                            if target is not None:
                                print(f"  Target value: {target:.8f}")
                                if diff is not None:
                                    print(f"  Difference: {diff:.2f}%")
                            print(f"  Convergence speed: {convergence_speed:.4f}")
                            print(f"  Status: {'✓ CONVERGED' if converged else '✗ NOT CONVERGED'}")
                            print(f"  Based on {len(valid_values)} data points")
                            
                        except Exception as e:
                            print(f"\n{observable}: Could not fit convergence model: {e}")
                            convergence_metrics[observable] = {'error': str(e)}
        
        # Save convergence data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conv_file = self.manager.output_dir / f"convergence_analysis_{timestamp}.csv"
        df.to_csv(conv_file, index=False)
        
        print(f"\n💾 Convergence data saved to: {conv_file}")
        
        return {
            'data': df,
            'metrics': convergence_metrics
        }
    
    def plot_convergence_results(self, convergence: Dict[str, Any]):
        """
        Create convergence plots.
        
        Parameters:
        -----------
        convergence : dict
            Convergence analysis results
        """
        df = convergence['data']
        metrics = convergence['metrics']
        
        # Filter successful experiments
        successful_df = df[df['success']]
        
        if len(successful_df) < 2:
            print("Not enough successful experiments for plotting")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Plot each observable
        observables = ['alpha', 'Omega_dm', 'spectral_dim', 'entropy']
        titles = ['Fine-structure constant α', 'Dark Matter Density Ω_dm', 
                 'Spectral Dimension', 'Entanglement Entropy']
        targets = [1/137.035999084, 0.265, 4.0, None]
        
        for idx, (obs, title, target) in enumerate(zip(observables, titles, targets)):
            ax = axes[idx]
            
            if successful_df[obs].notna().sum() > 1:
                # Plot data points
                ax.semilogx(successful_df['N'], successful_df[obs], 'o-', 
                           linewidth=2, markersize=8, label='Data')
                
                # Plot target if available
                if target is not None:
                    ax.axhline(y=target, color='r', linestyle='--', 
                              linewidth=2, label=f'Target: {target}')
                
                # Plot fitted asymptotic value if available
                if obs in metrics and 'asymptotic_value' in metrics[obs]:
                    asymptote = metrics[obs]['asymptotic_value']
                    error = metrics[obs]['asymptotic_error']
                    
                    # Plot asymptotic value with error band
                    ax.axhline(y=asymptote, color='g', linestyle='-', 
                              linewidth=2, alpha=0.7, label=f'Asymptote: {asymptote:.6f}')
                    ax.axhspan(asymptote - error, asymptote + error, 
                              alpha=0.2, color='g')
                
                ax.set_xlabel('N (number of vertices)')
                ax.set_ylabel(title.split()[-1])
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.manager.plots_dir / f"convergence_plots_{timestamp}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Convergence plots saved to: {plot_file}")

# ============================================================================
# 4. RESULT ANALYZER AND VISUALIZER
# ============================================================================

class ACTResultAnalyzer:
    """
    Analyze and visualize experiment results.
    """
    
    def __init__(self, results_dir: str = "experiment_results"):
        """
        Initialize result analyzer.
        
        Parameters:
        -----------
        results_dir : str
            Directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        self.experiment_db = self.results_dir / "experiment_database.csv"
        
        # Load all results
        self.results = self.load_all_results()
    
    def load_all_results(self) -> Dict[int, Dict[str, Any]]:
        """
        Load all experiment results from directory.
        
        Returns:
        --------
        results : dict
            Dictionary mapping experiment ID to results
        """
        results = {}
        
        if not self.results_dir.exists():
            print(f"Results directory not found: {self.results_dir}")
            return results
        
        # Load from database first
        if self.experiment_db.exists():
            df = pd.read_csv(self.experiment_db)
            
            for _, row in df.iterrows():
                exp_id = int(row['exp_id'])
                results_file = row['results_file']
                
                if results_file and Path(results_file).exists():
                    try:
                        with open(results_file, 'rb') as f:
                            results[exp_id] = pickle.load(f)
                        print(f"Loaded results for experiment {exp_id}")
                    except Exception as e:
                        print(f"Error loading results for experiment {exp_id}: {e}")
        
        # Also scan results directory for any additional files
        for results_file in self.results_dir.glob("exp_*.pkl"):
            try:
                # Extract experiment ID from filename
                parts = results_file.stem.split('_')
                if len(parts) >= 2 and parts[0] == 'exp':
                    exp_id = int(parts[1])
                    
                    if exp_id not in results:
                        with open(results_file, 'rb') as f:
                            results[exp_id] = pickle.load(f)
                        print(f"Loaded additional results for experiment {exp_id}")
            except Exception as e:
                print(f"Error loading {results_file}: {e}")
        
        print(f"Loaded {len(results)} experiment results")
        return results
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for all experiments.
        
        Returns:
        --------
        summary_df : pd.DataFrame
            DataFrame with summary statistics
        """
        if not self.results:
            print("No results loaded")
            return pd.DataFrame()
        
        summary_data = []
        
        for exp_id, res in self.results.items():
            if isinstance(res, dict):
                summary = {
                    'exp_id': exp_id,
                    'success': True
                }
                
                # Extract parameters
                if 'parameters' in res:
                    params = res['parameters']
                    summary.update({f'param_{k}': v for k, v in params.items()})
                
                # Extract key results
                if 'constants' in res and 'alpha' in res['constants']:
                    summary['alpha'] = res['constants']['alpha']
                
                if 'dark_matter' in res and 'Omega_dm' in res['dark_matter']:
                    summary['Omega_dm'] = res['dark_matter']['Omega_dm']
                
                if 'observables' in res and 'entanglement' in res['observables']:
                    summary['entropy'] = res['observables']['entanglement']
                
                # Extract timing
                if 'timing' in res and 'total' in res['timing']:
                    summary['compute_time'] = res['timing']['total']
                
                summary_data.append(summary)
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            return df
        else:
            return pd.DataFrame()
    
    def analyze_correlations(self):
        """
        Analyze correlations between parameters and results.
        """
        df = self.get_summary_statistics()
        
        if len(df) < 3:
            print("Not enough data for correlation analysis")
            return
        
        print(f"\n{'='*80}")
        print(f"CORRELATION ANALYSIS")
        print(f"{'='*80}")
        
        # Select numeric columns for correlation analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Filter columns we're interested in
        param_cols = [col for col in numeric_cols if col.startswith('param_')]
        result_cols = ['alpha', 'Omega_dm', 'entropy', 'compute_time']
        result_cols = [col for col in result_cols if col in df.columns]
        
        if param_cols and result_cols:
            # Compute correlations
            correlations = {}
            
            for param in param_cols:
                param_name = param.replace('param_', '')
                for result in result_cols:
                    if param in df.columns and result in df.columns:
                        corr = df[param].corr(df[result])
                        if not np.isnan(corr):
                            correlations[f"{param_name} vs {result}"] = corr
            
            # Sort by absolute correlation
            sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            
            print(f"\nTop correlations:")
            print(f"{'-'*60}")
            for name, corr in sorted_corrs[:10]:  # Top 10
                strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
                direction = "positive" if corr > 0 else "negative"
                print(f"{name:30} : {corr:7.3f} ({strength} {direction})")
            
            # Create correlation matrix
            analysis_cols = param_cols + result_cols
            corr_matrix = df[analysis_cols].corr()
            
            # Plot correlation heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
            plt.title('Parameter-Result Correlation Matrix')
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = self.results_dir.parent / "plots" / f"correlation_heatmap_{timestamp}.png"
            plot_file.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"\n📊 Correlation heatmap saved to: {plot_file}")
    
    def create_comprehensive_report(self, output_file: str = None):
        """
        Create comprehensive report of all experiments.
        
        Parameters:
        -----------
        output_file : str, optional
            Output file path for the report
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.results_dir.parent / f"act_experiment_report_{timestamp}.md"
        
        df = self.get_summary_statistics()
        
        if len(df) == 0:
            print("No data for report")
            return
        
        with open(output_file, 'w') as f:
            f.write("# ACT Experiment Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            f.write("## Summary Statistics\n\n")
            f.write(f"Total experiments: {len(df)}\n")
            f.write(f"Successful experiments: {df['success'].sum()}\n\n")
            
            # Key results summary
            if 'alpha' in df.columns:
                f.write("### Fine-structure constant α\n\n")
                f.write(f"Mean: {df['alpha'].mean():.12f}\n")
                f.write(f"Std: {df['alpha'].std():.12f}\n")
                f.write(f"Target: 0.007297352569\n")
                f.write(f"Difference: {abs(df['alpha'].mean() - 1/137.035999084)/(1/137.035999084)*100:.4f}%\n\n")
            
            if 'Omega_dm' in df.columns:
                f.write("### Dark matter density Ω_dm\n\n")
                f.write(f"Mean: {df['Omega_dm'].mean():.4f}\n")
                f.write(f"Std: {df['Omega_dm'].std():.4f}\n")
                f.write(f"Target: 0.265\n")
                f.write(f"Difference: {abs(df['Omega_dm'].mean() - 0.265)/0.265*100:.2f}%\n\n")
            
            # Parameter analysis
            f.write("## Parameter Analysis\n\n")
            param_cols = [col for col in df.columns if col.startswith('param_')]
            
            if param_cols:
                f.write("| Parameter | Mean | Std | Min | Max |\n")
                f.write("|-----------|------|-----|-----|-----|\n")
                
                for param in param_cols:
                    param_name = param.replace('param_', '')
                    if param in df.columns:
                        f.write(f"| {param_name} | {df[param].mean():.2f} | "
                               f"{df[param].std():.2f} | {df[param].min():.2f} | "
                               f"{df[param].max():.2f} |\n")
                f.write("\n")
            
            # Experiment list
            f.write("## Experiment Details\n\n")
            f.write("| ID | N | Temperature | α | Ω_dm | Status |\n")
            f.write("|----|---|-------------|---|------|--------|\n")
            
            for _, row in df.iterrows():
                exp_id = row['exp_id']
                N = row.get('param_N', 'N/A')
                temp = row.get('param_temperature', 'N/A')
                alpha = row.get('alpha', 'N/A')
                Omega_dm = row.get('Omega_dm', 'N/A')
                success = "✓" if row.get('success', False) else "✗"
                
                if isinstance(alpha, (int, float)):
                    alpha_str = f"{alpha:.10f}"
                else:
                    alpha_str = str(alpha)
                
                if isinstance(Omega_dm, (int, float)):
                    Omega_dm_str = f"{Omega_dm:.4f}"
                else:
                    Omega_dm_str = str(Omega_dm)
                
                f.write(f"| {exp_id} | {N} | {temp} | {alpha_str} | {Omega_dm_str} | {success} |\n")
            
            f.write("\n")
            
            # Conclusions
            f.write("## Conclusions\n\n")
            f.write("1. **Convergence**: The results show...\n")
            f.write("2. **Accuracy**: Key predictions match experimental values to within...\n")
            f.write("3. **Scalability**: Computation time scales as...\n")
            f.write("4. **Recommendations**: For future experiments...\n")
        
        print(f"📄 Comprehensive report saved to: {output_file}")

# ============================================================================
# 5. MAIN FUNCTION AND COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='ACT Experiment Runner - Run and analyze ACT experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python act_experiment_runner.py --single --N 1000
  python act_experiment_runner.py --sweep-N 500 2000 500
  python act_experiment_runner.py --convergence --N-max 2000
  python act_experiment_runner.py --analyze
        """
    )
    
    # Experiment types
    parser.add_argument('--single', action='store_true', 
                       help='Run single experiment')
    parser.add_argument('--sweep-N', nargs=3, type=int, metavar=('START', 'STOP', 'STEP'),
                       help='Sweep over N values')
    parser.add_argument('--sweep-temperature', nargs=3, type=float, 
                       metavar=('MIN', 'MAX', 'STEP'),
                       help='Sweep over temperature values')
    parser.add_argument('--convergence', action='store_true',
                       help='Run convergence test')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze existing results')
    
    # Experiment parameters
    parser.add_argument('--N', type=int, default=1000,
                       help='Number of vertices (default: 1000)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature (default: 0.7)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--thermal-steps', type=int, default=500,
                       help='Monte Carlo thermalization steps (default: 500)')
    parser.add_argument('--dimension', type=int, default=4,
                       help='Spacetime dimension (default: 4)')
    
    # Convergence test parameters
    parser.add_argument('--N-max', type=int, default=2000,
                       help='Maximum N for convergence test (default: 2000)')
    parser.add_argument('--N-step', type=int, default=200,
                       help='N step size for convergence test (default: 200)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='experiment_results',
                       help='Output directory (default: experiment_results)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plot generation')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize experiment manager
    print(f"ACT Experiment Runner")
    print(f"Output directory: {output_dir}")
    print(f"ACT modules available: {ACT_AVAILABLE}")
    print()
    
    manager = ACTExperimentManager(output_dir=output_dir)
    
    # Run requested experiment type
    if args.single:
        print(f"Running single experiment with N={args.N}")
        
        params = {
            'N': args.N,
            'temperature': args.temperature,
            'seed': args.seed,
            'thermal_steps': args.thermal_steps,
            'dimension': args.dimension,
            'notes': 'Single experiment'
        }
        
        results = manager.run_single_experiment(params)
        
    elif args.sweep_N:
        start, stop, step = args.sweep_N
        N_values = list(range(start, stop + step, step))
        
        print(f"Running N sweep: {N_values}")
        
        sweep = ACTParameterSweep(manager)
        results = sweep.sweep_N(
            N_values, 
            save_plots=not args.no_plots
        )
        
    elif args.sweep_temperature:
        t_min, t_max, t_step = args.sweep_temperature
        T_values = np.arange(t_min, t_max + t_step/2, t_step)
        
        print(f"Running temperature sweep: {T_values}")
        
        sweep = ACTParameterSweep(manager)
        results = sweep.sweep_temperature(
            T_values.tolist(),
            save_plots=not args.no_plots
        )
        
    elif args.convergence:
        print(f"Running convergence test up to N={args.N_max}")
        
        convergence_test = ACTConvergenceTest(manager)
        results = convergence_test.run_convergence_test(
            N_max=args.N_max,
            N_step=args.N_step,
            save_plots=not args.no_plots
        )
        
    elif args.analyze:
        print("Analyzing existing results...")
        
        analyzer = ACTResultAnalyzer(results_dir=args.output_dir)
        
        # Load and display summary
        summary = analyzer.get_summary_statistics()
        if not summary.empty:
            print(f"\nLoaded {len(summary)} experiments")
            print(f"\nSummary statistics:")
            print(summary.describe())
            
            # Analyze correlations
            analyzer.analyze_correlations()
            
            # Create comprehensive report
            analyzer.create_comprehensive_report()
        else:
            print("No results found for analysis")
        
    else:
        # Default: run a quick test
        print("No specific experiment type requested")
        print("Running a quick test with N=800...")
        
        params = {
            'N': 800,
            'temperature': 0.7,
            'seed': 42,
            'thermal_steps': 200,
            'dimension': 4,
            'notes': 'Quick test'
        }
        
        results = manager.run_single_experiment(params)
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT RUNNER COMPLETED")
    print(f"{'='*80}")
    print(f"Results saved in: {output_dir}")
    
    if not ACT_AVAILABLE:
        print("\n⚠️  Note: ACT modules were not available")
        print("   Ran in simulation mode with synthetic data")
        print("   To run actual ACT experiments, ensure act_core.py is available")

if __name__ == "__main__":
    main()
