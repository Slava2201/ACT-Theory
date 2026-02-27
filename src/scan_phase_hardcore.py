import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
# Импортируем наш основной движок из файла (укажи имя своего файла с классом ACT_ProductionEngine)
# from act_production_v7 import ACT_ProductionEngine 

def run_phase_scan(start_phi=0.0, end_phi=2.0, steps=50):
    results = []
    print(f"🚀 Starting Phase Scan: {steps} steps from {start_phi}π to {end_phi}π")
    
    for p in tqdm(np.linspace(start_phi, end_phi, steps)):
        # Запускаем движок на минималках (L=8 или 10) для скорости сканирования
        engine = ACT_ProductionEngine(L=8, k=100, stable_phase=p)
        
        # Решаем спектры
        spectra = engine.solve_all_octants()
        if len(spectra) < 8: continue
        
        # Считаем космологию
        dm_res = engine.compute_dark_matter_calibrated(spectra)
        
        results.append({
            'phase': p,
            'dm_percent': dm_res['dm_raw'],
            'de_percent': dm_res['de_raw'],
            'baryon_percent': dm_res['b_raw']
        })
    
    return pd.DataFrame(results)

def plot_phase_analysis(df):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("ACT Phase Stability Analysis", fontsize=20, y=0.95)

    # 1. DM vs Phase
    axes[0,0].scatter(df['phase'], df['dm_percent'], color='cyan', alpha=0.6)
    axes[0,0].axhline(26.0, color='red', linestyle='--', label='Target 26%')
    axes[0,0].set_title("Dark Matter vs Phase")
    axes[0,0].set_ylabel("DM %")

    # 2. DE vs Phase
    axes[0,1].scatter(df['phase'], df['de_percent'], color='magenta', alpha=0.6)
    axes[0,1].axhline(68.0, color='red', linestyle='--', label='Target 68%')
    axes[0,1].set_title("Dark Energy vs Phase")

    # 3. Baryons vs Phase
    axes[1,0].scatter(df['phase'], df['baryon_percent'], color='lime', alpha=0.6)
    axes[1,0].axhline(5.0, color='red', linestyle='--', label='Target 5%')
    axes[1,0].set_title("Baryons vs Phase")
    axes[1,0].set_xlabel("Phase (π)")

    # 4. Heatmap (Stability Density)
    sns.kdeplot(x=df['phase'], y=df['dm_percent'], cmap="rocket", fill=True, ax=axes[1,1])
    axes[1,1].axhline(26.0, color='white', linestyle='--', alpha=0.5)
    axes[1,1].set_title("Phase Stability Heatmap")
    axes[1,1].set_xlabel("Phase (π)")

    for ax in axes.flat: ax.grid(True, alpha=0.2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("phase_stability_scan.png", dpi=150)
    plt.show()

# Запуск
df_results = run_phase_scan(steps=40) # 40 шагов хватит для красивой карты
plot_phase_analysis(df_results)

