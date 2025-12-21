```python
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
from typing import Dict, List, Tuple, Optional, Any
import sys
import os
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import ACT modules
try:
    from act_model import ACTModel, run_act_experiment
    from act_dark_matter import analyze_act_dark_matter
    from act_cosmology import run_act_cosmology_simulation
    ACT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import ACT modules: {e}")
    print("Running in limited mode with mock data...")
    ACT_AVAILABLE = False

# ============================================================================
# 1. EXPERIMENT MANAGER
# ============================================================================

class ACTExperimentManager:
    """
    Manages and runs ACT experiments with different parameters.
    """
    
    def __init__(self, output_dir="experiment_results"):
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
            'include_dm', 'thermal_steps', 'status',
            'results_file', 'compute_time', 'notes'
        ]
        
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.experiment_db, index=False)
        print(f"Created new experiment database: {self.experiment_db}")
    
    def get_next_exp_id(self):
        """Get next available experiment ID."""
        if self.experiment_db.exists():
            df = pd.read_csv(self.experiment_db)
            if len(df) > 0:
                return df['exp_id'].max() + 1
        return 1
    
    def register_experiment(self, params: Dict[str, Any]):
        """
        Register new experiment in database.
        
        Parameters:
        -----------
        params : dict
            Experiment parameters
        """
        exp_record = {
            'exp_id': self.current_exp_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'N': params.get('N', 1000),
            'temperature': params.get('temperature', 0.7),
            'seed': params.get('seed', 42),
            'include_dm': params.get('include_dm', True),
            'thermal_steps': params.get('thermal_steps', 500),
            'status': 'registered',
            'results_file': '',
            'compute_time': 0,
            'notes': params.get('notes', '')
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
    
    def update_experiment_status(self, exp_id: int, status: str, 
                                 results_file: str = "", compute_time: float = 0):
        """
        Update experiment status in database.
        
        Parameters:
        -----------
        exp_id : int
            Experiment ID
        status : str
            New status (running, completed, failed)
        results_file : str
            Path to results file
        compute_time : float
            Computation time in seconds
        """
        if not self.experiment_db.exists():
            return
        
        df = pd.read_csv(self.experiment_db)
        
        if exp_id in df['exp_id'].values:
            idx = df[df['exp_id'] == exp_id].index[0]
            df.at[idx, 'status'] = status
            df.at[idx, 'results_file'] = results_file
            df.at[idx, 'compute_time'] = compute_time
            df.to_csv(self.experiment_db, index=False)
            
            print(f"Updated experiment {exp_id}: {status}")
    
    def run_single_experiment(self, params: Dict[str, Any]):
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
        self.update_experiment_status(exp_id, "running")
        
        start_time = datetime.now()
        
        try:
            if ACT_AVAILABLE:
                # Run actual ACT experiment
                results = self._run_act_experiment(params)
            else:
                # Generate mock results for testing
                results = self._generate_mock_results(params)
            
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
            self.update_experiment_status(
                exp_id, "completed", str(results_file), compute_time
            )
            
            # Print summary
            self._print_experiment_summary(results, compute_time)
            
            return results
            
        except Exception as e:
            # Handle errors
            error_msg = f"Experiment {exp_id} failed: {str(e)}"
            print(f"\n❌ {error_msg}")
            
            self.update_experiment_status(exp_id, "failed")
            
            # Save error information
            error_file = self.results_dir / f"exp_{exp_id}_error.txt"
            with open(error_file, 'w') as f:
                f.write(f"Experiment {exp_id} failed at {datetime.now()}\n")
                f.write(f"Parameters: {json.dumps(params, indent=2)}\n")
                f.write(f"Error: {str(e)}\n")
            
            # Increment experiment ID for next experiment
            self.current_exp_id += 1
            
            raise
    
    def _run_act_experiment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run actual ACT experiment with given parameters.
        """
        # Extract parameters
        N = params['N']
        temperature = params['temperature']
        seed = params['seed']
        include_dm = params['include_dm']
        thermal_steps = params['thermal_steps']
        
        print(f"Parameters:")
        print(f"  N = {N:,}")
        print(f"  Temperature = {temperature}")
        print(f"  Seed = {seed}")
        print(f"  Include DM = {include_dm}")
        print(f"  Thermal steps = {thermal_steps:,}")
        
        # 1. Create and thermalize ACT model
        print(f"\n1. Creating ACT model...")
        model = ACTModel(
            N=N,
            temperature=temperature,
            seed=seed,
            include_dark_matter=include_dm
        )
        
        print(f"2. Thermalizing model...")
        model.thermalize(n_steps=thermal_steps, batch_size=min(100, N//10))
        
        # 2. Run dark matter analysis
        print(f"3. Analyzing dark matter...")
        dm_results = analyze_act_dark_matter(model, visualize=False, save_results=False)
        
        # 3. Run cosmology simulation
        print(f"4. Running cosmology simulation...")
        cosmo_results = run_act_cosmology_simulation(
            model, save_results=False, visualize=False
        )
        
        # 4. Compute observables
        print(f"5. Computing observables...")
        observables = model.calculate_observables()
        
        # 5. Compile results
        results = {
            'experiment_id': self.current_exp_id,
            'parameters': params,
            'model_info': {
                'N': N,
                'tetrahedra_count': len(model.tetrahedra),
                'average_degree': model.adjacency.sum() / N,
                'causal_density': model.causal_matrix.sum() / (N * (N - 1))
            },
            'dark_matter': dm_results,
            'cosmology': cosmo_results,
            'observables': observables,
            'fundamental_constants': observables.get('fundamental_constants', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def _generate_mock_results(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate mock results for testing without ACT modules.
        """
        N = params['N']
        
        # Simulate computation time
        import time
        time.sleep(2)  # Simulate computation
        
        results = {
            'experiment_id': self.current_exp_id,
            'parameters': params,
            'model_info': {
                'N': N,
                'tetrahedra_count': N // 4,
                'average_degree': 6.2,
                'causal_density': 0.1
            },
            'dark_matter': {
                'summary': {
                    'n_defects': N // 10,
                    'Omega_dm_predicted': 0.265,
                    'mean_defect_mass_tev': 1.2,
                    'best_fit_profile': 'Cored'
                }
            },
            'cosmology': {
                'summary': {
                    'parameters': {
                        'H0': '70.2 km/s/Mpc',
                        'Ω_Λ': '0.685',
                        'n_s': '0.965',
                        'σ_8': '0.811'
                    },
                    'consistency_score': 0.92
                }
            },
            'observables': {
                'action': {'S_total': 1.23e-45},
                'spectral_dimension': 3.2,
                'hausdorff_dimension': 3.1,
                'entanglement': {'entropy': 45.6}
            },
            'fundamental_constants': {
                'alpha': 1/137.035999084,
                'G': 6.67430e-11,
                'c': 299792458,
                'Lambda': 1.1056e-52
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def _create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of experiment results."""
        summary = {
            'experiment_id': results['experiment_id'],
            'parameters': results['parameters'],
            'model_info': results['model_info'],
            'key_results': {}
        }
        
        # Extract key results
        if 'dark_matter' in results:
            dm = results['dark_matter']
            if 'summary' in dm:
                summary['key_results']['dark_matter'] = dm['summary']
        
        if 'cosmology' in results:
            cosmo = results['cosmology']
            if 'summary' in cosmo:
                summary['key_results']['cosmology'] = cosmo['summary']
        
        if 'fundamental_constants' in results:
            summary['key_results']['constants'] = results['fundamental_constants']
        
        if 'observables' in results:
            obs = results['observables']
            summary['key_results']['observables'] = {
                'action': obs.get('action', {}).get('S_total', 'N/A'),
                'spectral_dimension': obs.get('spectral_dimension', 'N/A'),
                'entanglement_entropy': obs.get('entanglement', {}).get('entropy', 'N/A')
            }
        
        return summary
    
    def _print_experiment_summary(self, results: Dict[str, Any], compute_time: float):
        """Print experiment summary."""
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {results['experiment_id']} COMPLETED")
        print(f"{'='*80}")
        
        print(f"\n⏱ Computation time: {compute_time:.1f} seconds")
        
        # Print key results
        print(f"\n📊 KEY RESULTS:")
        print(f"{'-'*40}")
        
        # Dark matter
        if 'dark_matter' in results and 'summary' in results['dark_matter']:
            dm = results['dark_matter']['summary']
            print(f"Dark Matter:")
            print(f"  • Defects: {dm.get('n_defects', 'N/A'):,}")
            print(f"  • Ω_dm: {dm.get('Omega_dm_predicted', 'N/A'):.4f} (obs: 0.265)")
            print(f"  • Mass scale: {dm.get('mean_defect_mass_tev', 'N/A'):.1f} TeV")
        
        # Cosmology
        if 'cosmology' in results and 'summary' in results['cosmology']:
            cosmo = results['cosmology']['summary']
            if 'parameters' in cosmo:
                params = cosmo['parameters']
                print(f"\nCosmology:")
                print(f"  • H₀: {params.get('H0', 'N/A')}")
                print(f"  • Ω_Λ: {params.get('Ω_Λ', 'N/A')}")
                print(f"  • n_s: {params.get('n_s', 'N/A')}")
                print(f"  • σ₈: {params.get('σ_8', 'N/A')}")
            
            if 'consistency_score' in cosmo:
                score = cosmo['consistency_score']
                print(f"  • Consistency: {score*100:.1f}%")
        
        # Fundamental constants
        if 'fundamental_constants' in results:
            const = results['fundamental_constants']
            print(f"\nFundamental Constants:")
            print(f"  • α: {const.get('alpha', 'N/A'):.10f}")
            print(f"  • G: {const.get('G', 'N/A'):.4e} m³/kg/s²")
            print(f"  • Λ: {const.get('Lambda', 'N/A'):.4e} m⁻²")
        
        print(f"\n💾 Results saved to: {self.results_dir}/")
        
        # Increment experiment ID for next experiment
        self.current_exp_id += 1
        print(f"\nNext experiment ID: {self.current_exp_id}")

# ============================================================================
# 2. PARAMETER SWEEP EXPERIMENTS
# ============================================================================

class ACTParameterSweep:
    """
    Run parameter sweep experiments to study ACT behavior.
    """
    
    def __init__(self, experiment_manager):
        """
        Initialize parameter sweep.
        
        Parameters:
        -----------
        experiment_manager : ACTExperimentManager
            Manager to run experiments
        """
        self.manager = experiment_manager
    
    def sweep_N(self, N_values: List[int], base_params: Dict[str, Any] = None):
        """
        Sweep over number of vertices N.
        
        Parameters:
        -----------
        N_values : list
            List of N values to test
        base_params : dict
            Base parameters for all experiments
            
        Returns:
        --------
        results : dict
            Results for all N values
        """
        if base_params is None:
            base_params = {
                'temperature': 0.7,
                'seed': 42,
                'include_dm': True,
                'thermal_steps': 500,
                'notes': 'N sweep'
            }
        
        print(f"\n{'='*80}")
        print(f"PARAMETER SWEEP: N = {N_values}")
        print(f"{'='*80}")
        
        all_results = {}
        
        for N in N_values:
            print(f"\nRunning N = {N:,}...")
            
            params = base_params.copy()
            params['N'] = N
            
            try:
                results = self.manager.run_single_experiment(params)
                all_results[N] = results
                
            except Exception as e:
                print(f"  Failed for N={N}: {e}")
                all_results[N] = {'error': str(e)}
        
        # Analyze sweep results
        self.analyze_N_sweep(all_results)
        
        return all_results
    
    def analyze_N_sweep(self, results: Dict[int, Dict[str, Any]]):
        """
        Analyze results from N sweep.
        """
        print(f"\n{'='*80}")
        print(f"N SWEEP ANALYSIS")
        print(f"{'='*80}")
        
        # Extract data
        N_list = []
        tetrahedra_list = []
        degree_list = []
        Omega_dm_list = []
        alpha_list = []
        
        for N, res in results.items():
            if 'error' not in res:
                N_list.append(N)
                
                # Model info
                if 'model_info' in res:
                    tetrahedra_list.append(res['model_info']['tetrahedra_count'])
                    degree_list.append(res['model_info']['average_degree'])
                
                # Dark matter
                if 'dark_matter' in res and 'summary' in res['dark_matter']:
                    dm = res['dark_matter']['summary']
                    Omega_dm_list.append(dm.get('Omega_dm_predicted', np.nan))
                
                # Fundamental constants
                if 'fundamental_constants' in res:
                    const = res['fundamental_constants']
                    alpha_list.append(const.get('alpha', np.nan))
        
        if len(N_list) < 2:
            print("Not enough successful runs for analysis")
            return
        
        # Create analysis dataframe
        analysis_data = {
            'N': N_list,
            'tetrahedra': tetrahedra_list,
            'avg_degree': degree_list,
            'Omega_dm': Omega_dm_list,
            'alpha': alpha_list
        }
        
        df = pd.DataFrame(analysis_data)
        
        # Compute scaling laws
        print(f"\n📈 SCALING LAWS:")
        print(f"{'-'*40}")
        
        # Tetrahedra vs N
        if len(df) > 1:
            coeffs = np.polyfit(np.log(df['N']), np.log(df['tetrahedra']), 1)
            print(f"Tetrahedra ∝ N^{coeffs[0]:.3f}")
        
        # Ω_dm scaling
        if 'Omega_dm' in df.columns and df['Omega_dm'].notna().sum() > 1:
            Omega_dm_vals = df['Omega_dm'].values
            if not np.all(np.isnan(Omega_dm_vals)):
                mean_Omega = np.nanmean(Omega_dm_vals)
                std_Omega = np.nanstd(Omega_dm_vals)
                print(f"Ω_dm = {mean_Omega:.4f} ± {std_Omega:.4f}")
                print(f"  Target: 0.265")
        
        # α scaling
        if 'alpha' in df.columns and df['alpha'].notna().sum() > 1:
            alpha_vals = df['alpha'].values
            if not np.all(np.isnan(alpha_vals)):
                mean_alpha = np.nanmean(alpha_vals)
                std_alpha = np.nanstd(alpha_vals)
                print(f"α = {mean_alpha:.10f} ± {std_alpha:.10f}")
                print(f"  Target: 1/137.035999084 = 0.00729735257")
        
        # Save analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = self.manager.output_dir / f"N_sweep_analysis_{timestamp}.csv"
        df.to_csv(analysis_file, index=False)
        
        print(f"\n💾 Analysis saved to: {analysis_file}")
        
        return df
    
    def sweep_temperature(self, T_values: List[float], base_params: Dict[str, Any] = None):
        """
        Sweep over temperature.
        
        Parameters:
        -----------
        T_values : list
            List of temperature values to test
        base_params : dict
            Base parameters for all experiments
        """
        if base_params is None:
            base_params = {
                'N': 1000,
                'seed': 42,
                'include_dm': True,
                'thermal_steps': 500,
                'notes': 'Temperature sweep'
            }
        
        print(f"\n{'='*80}")
        print(f"PARAMETER SWEEP: Temperature = {T_values}")
        print(f"{'='*80}")
        
        all_results = {}
        
        for T in T_values:
            print(f"\nRunning T = {T}...")
            
            params = base_params.copy()
            params['temperature'] = T
            
            try:
                results = self.manager.run_single_experiment(params)
                all_results[T] = results
                
            except Exception as e:
                print(f"  Failed for T={T}: {e}")
                all_results[T] = {'error': str(e)}
        
        # Analyze temperature dependence
        self.analyze_temperature_sweep(all_results)
        
        return all_results
    
    def analyze_temperature_sweep(self, results: Dict[float, Dict[str, Any]]):
        """
        Analyze temperature dependence.
        """
        print(f"\n{'='*80}")
        print(f"TEMPERATURE SWEEP ANALYSIS")
        print(f"{'='*80}")
        
        # Extract data
        T_list = []
        action_list = []
        entropy_list = []
        dimension_list = []
        
        for T, res in results.items():
            if 'error' not in res:
                T_list.append(T)
                
                # Action
                if 'observables' in res and 'action' in res['observables']:
                    action = res['observables']['action']
                    if isinstance(action, dict) and 'S_total' in action:
                        action_list.append(action['S_total'])
                    else:
                        action_list.append(np.nan)
                else:
                    action_list.append(np.nan)
                
                # Entanglement entropy
                if 'observables' in res and 'entanglement' in res['observables']:
                    ent = res['observables']['entanglement']
                    if isinstance(ent, dict) and 'entropy' in ent:
                        entropy_list.append(ent['entropy'])
                    else:
                        entropy_list.append(np.nan)
                else:
                    entropy_list.append(np.nan)
                
                # Spectral dimension
                if 'observables' in res and 'spectral_dimension' in res['observables']:
                    dim = res['observables']['spectral_dimension']
                    if isinstance(dim, (int, float)):
                        dimension_list.append(dim)
                    else:
                        dimension_list.append(np.nan)
                else:
                    dimension_list.append(np.nan)
        
        if len(T_list) < 2:
            print("Not enough successful runs for analysis")
            return
        
        # Create analysis dataframe
        analysis_data = {
            'temperature': T_list,
            'action': action_list,
            'entanglement_entropy': entropy_list,
            'spectral_dimension': dimension_list
        }
        
        df = pd.DataFrame(analysis_data)
        
        # Look for phase transitions
        print(f"\n🔍 PHASE TRANSITION ANALYSIS:")
        print(f"{'-'*40}")
        
        # Check for discontinuities in derivatives
        if len(df) > 3:
            # Sort by temperature
            df = df.sort_values('temperature')
            
            # Compute derivatives
            dS_dT = np.gradient(df['entanglement_entropy'].values, df['temperature'].values)
            dDim_dT = np.gradient(df['spectral_dimension'].values, df['temperature'].values)
            
            # Look for peaks in derivatives (possible phase transitions)
            dS_peaks = np.where(np.abs(dS_dT) > 2 * np.std(dS_dT))[0]
            dDim_peaks = np.where(np.abs(dDim_dT) > 2 * np.std(dDim_dT))[0]
            
            if len(dS_peaks) > 0:
                print(f"Possible phase transitions at T = {df['temperature'].iloc[dS_peaks].values}")
            else:
                print("No clear phase transitions detected")
        
        # Save analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = self.manager.output_dir / f"temperature_sweep_analysis_{timestamp}.csv"
        df.to_csv(analysis_file, index=False)
        
        print(f"\n💾 Analysis saved to: {analysis_file}")
        
        return df

# ============================================================================
# 3. CONVERGENCE TESTS
# ============================================================================

class ACTConvergenceTest:
    """
    Test convergence of ACT observables with increasing N.
    """
    
    def __init__(self, experiment_manager):
        self.manager = experiment_manager
    
    def run_convergence_test(self, N_max: int = 2000, N_step: int = 200, 
                           base_params: Dict[str, Any] = None):
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
            
        Returns:
        --------
        convergence_data : dict
            Convergence test results
        """
        if base_params is None:
            base_params = {
                'temperature': 0.7,
                'seed': 42,
                'include_dm': True,
                'thermal_steps': 500,
                'notes': 'Convergence test'
            }
        
        print(f"\n{'='*80}")
        print(f"CONVERGENCE TEST: N up to {N_max:,} (step {N_step})")
        print(f"{'='*80}")
        
        N_values = list(range(N_step, N_max + N_step, N_step))
        
        # Run parameter sweep
        sweep = ACTParameterSweep(self.manager)
        results = sweep.sweep_N(N_values, base_params)
        
        # Analyze convergence
        convergence = self.analyze_convergence(results)
        
        return {
            'results': results,
            'convergence': convergence
        }
    
    def analyze_convergence(self, results: Dict[int, Dict[str, Any]]):
        """
        Analyze convergence of observables with N.
        """
        print(f"\n{'='*80}")
        print(f"CONVERGENCE ANALYSIS")
        print(f"{'='*80}")
        
        # Extract convergence data
        convergence_data = {
            'N': [],
            'alpha': [],
            'Omega_dm': [],
            'spectral_dim': [],
            'entanglement': []
        }
        
        for N, res in results.items():
            if 'error' not in res:
                convergence_data['N'].append(N)
                
                # Fundamental constants
                if 'fundamental_constants' in res:
                    const = res['fundamental_constants']
                    convergence_data['alpha'].append(const.get('alpha', np.nan))
                else:
                    convergence_data['alpha'].append(np.nan)
                
                # Dark matter
                if 'dark_matter' in res and 'summary' in res['dark_matter']:
                    dm = res['dark_matter']['summary']
                    convergence_data['Omega_dm'].append(dm.get('Omega_dm_predicted', np.nan))
                else:
                    convergence_data['Omega_dm'].append(np.nan)
                
                # Observables
                if 'observables' in res:
                    obs = res['observables']
                    convergence_data['spectral_dim'].append(obs.get('spectral_dimension', np.nan))
                    if 'entanglement' in obs:
                        convergence_data['entanglement'].append(obs['entanglement'].get('entropy', np.nan))
                    else:
                        convergence_data['entanglement'].append(np.nan)
                else:
                    convergence_data['spectral_dim'].append(np.nan)
                    convergence_data['entanglement'].append(np.nan)
        
        # Compute convergence metrics
        df = pd.DataFrame(convergence_data)
        df = df.sort_values('N')
        
        convergence_metrics = {}
        
        for observable in ['alpha', 'Omega_dm', 'spectral_dim', 'entanglement']:
            values = df[observable].values
            
            if len(values) > 1 and not np.all(np.isnan(values)):
                # Fit to convergence model: value = a + b/N^c
                valid_idx = ~np.isnan(values)
                N_valid = df['N'].values[valid_idx]
                vals_valid = values[valid_idx]
                
                if len(vals_valid) > 3:
                    # Simple convergence: check standard deviation of last few points
                    last_n = min(5, len(vals_valid))
                    last_values = vals_valid[-last_n:]
                    
                    mean_last = np.mean(last_values)
                    std_last = np.std(last_values)
                    rel_std = std_last / np.abs(mean_last) if mean_last != 0 else np.inf
                    
                    convergence_metrics[observable] = {
                        'mean_last': mean_last,
                        'std_last': std_last,
                        'relative_std': rel_std,
                        'converged': rel_std < 0.1  # Less than 10% variation
                    }
        
        # Print convergence results
        print(f"\n📊 CONVERGENCE METRICS:")
        print(f"{'-'*40}")
        
        for obs, metrics in convergence_metrics.items():
            status = "✓ CONVERGED" if metrics['converged'] else "✗ NOT CONVERGED"
            print(f"{obs}:")
            print(f"  Mean (last points): {metrics['mean_last']:.6f}")
            print(f"  Std dev: {metrics['std_last']:.6f}")
            print(f"  Relative std: {metrics['relative_std']:.3f}")
            print(f"  Status: {status}")
        
        # Save convergence data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conv_file = self.manager.output_dir / f"convergence_analysis_{timestamp}.csv"
        df.to_csv(conv_file, index=False)
        
        print(f"\n💾 Convergence data saved to: {conv_file}")
        
        return {
            'data': df,
            'metrics': convergence_metrics
        }

# ============================================================================
# 4. PARALLEL EXPERIMENT RUNNER
# ============================================================================

class ACTParallelRunner:
    """
    Run multiple experiments in parallel.
    """
    
    def __init__(self, experiment_manager, n_workers: int = None):
        """
        Initialize parallel runner.
        
        Parameters:
        -----------
        experiment_manager : ACTExperimentManager
            Experiment manager
        n_workers : int
            Number of parallel workers (default: CPU count)
        """
        self.manager = experiment_manager
        
        if n_workers is None:
            n_workers = mp.cpu_count()
        self.n_workers = min(n_workers, mp.cpu_count())
        
        print(f"Parallel runner initialized with {self.n_workers} workers")
    
    def run_parallel_experiments(self, experiment_list: List[Dict[str, Any]]):
        """
        Run multiple experiments in parallel.
        
        Parameters:
        -----------
        experiment_list : list
            List of experiment parameter dictionaries
            
        Returns:
        --------
        all_results : list
            List of results from all experiments
        """
        print(f"\n{'='*80}")
        print(f"PARALLEL EXPERIMENTS: {len(experiment_list)} experiments")
        print(f"{'='*80}")
        
        # Prepare experiment functions
        def run_experiment_wrapper(params):
            """Wrapper to run single experiment."""
            try:
                # We need to create a new manager for each process
                # because database access isn't thread-safe
                temp_manager = ACTExperimentManager(
                    output_dir=str(self.manager.output_dir / "parallel")
                )
                temp_manager.current_exp_id = params.get('exp_id', 1)
                
                results = temp_manager.run_single_experiment(params)
                return results
            except Exception as e:
                return {'error': str(e), 'params': params}
        
        # Run in parallel
        print(f"Starting {len(experiment_list)} experiments with {self.n_workers} workers...")
        
        with mp.Pool(processes=self.n_workers) as pool:
            results = pool.map(run_experiment_wrapper, experiment_list)
        
        # Collect results
        all_results = []
        successful = 0
        failed = 0
        
        for i, res in enumerate(results):
            if 'error' in res:
                failed += 1
                print(f"Experiment {i+1} failed: {res['error']}")
            else:
                successful += 1
            
            all_results.append(res)
        
        print(f"\n📊 PARALLEL RUN SUMMARY:")
        print(f"{'-'*40}")
        print(f"Successful: {successful}/{len(experiment_list)}")
        print(f"Failed: {failed}/{len(experiment_list)}")
        
        # Save combined results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_file = self.manager.output_dir / f"parallel_results_{timestamp}.pkl"
        
        with open(combined_file, 'wb') as f:
            pickle.dump(all_results, f)
        
        print(f"\n💾 Combined results saved to: {combined_file}")
        
        return all_results
    
    def generate_grid_search(self, param_grid: Dict[str, List]):
        """
        Generate parameter grid for grid search.
        
        Parameters:
        -----------
        param_grid : dict
            Dictionary of parameter names and value lists
            
        Returns:
        --------
        experiment_list : list
            List of all parameter combinations
        """
        from itertools import product
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        combinations = list(product(*param_values))
        
        experiment_list = []
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            params['exp_id'] = i + 1
            experiment_list.append(params)
        
        print(f"Generated {len(experiment_list)} parameter combinations")
        
        return experiment_list

# ============================================================================
# 5. RESULT ANALYZER AND VISUALIZER
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
