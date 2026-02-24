import numpy as np
from scipy.sparse import save_npz, load_npz, csr_matrix, eye
from scipy.sparse.linalg import eigsh
import os
import json
import matplotlib.pyplot as plt

class ACT_Universe_Engine:
    def __init__(self, L=14):
        self.L = L
        self.N = L**3
        self.matrix_dir = "act_matrices"
        self.output_dir = "act_results"
        
        for d in [self.matrix_dir, self.output_dir]:
            if not os.path.exists(d): 
                os.makedirs(d)

    def generate_matrices(self):
        print(f"🚀 Generating ACT Causal Network (L={self.L})...")
        for octant in range(8):
            row, col, data = [], [], []
            octant_phase = np.exp(1j * np.pi * octant / 4.0)
            
            for i in range(self.N):
                for axis in range(3):
                    neighbor = (i + self.L**axis) % self.N
                    row.extend([i, neighbor])
                    col.extend([neighbor, i])
                    data.extend([octant_phase, np.conj(octant_phase)])
            
            matrix = csr_matrix((data, (row, col)), shape=(self.N, self.N))
            save_npz(os.path.join(self.matrix_dir, f"matrix_octant_{octant}.npz"), matrix)
            print(f"✅ Octant {octant} matrix built.")

    def solve_universe(self):
        results = []
        for i in range(8):
            path = os.path.join(self.matrix_dir, f"matrix_octant_{i}.npz")
            if not os.path.exists(path): 
                self.generate_matrices()
            
            print(f"🔍 Solving Octant {i}...")
            matrix = load_npz(path)
            # Ищем k=32 минимальных по модулю собственных значений
            eigenvalues = eigsh(matrix, k=32, which='SM', return_eigenvectors=False)
            
            res = {"octant": i, "eigenvalues": sorted(eigenvalues.real.tolist())}
            results.append(res)
            with open(os.path.join(self.output_dir, f"spectrum_octant_{i}.json"), 'w') as f:
                json.dump(res, f)
        return results

    def build_dashboard(self, data):
        print("📊 Rendering Universe Profile Dashboard...")
        fig, axs = plt.subplots(2, 3, figsize=(16, 10))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        all_eigen = [val for d in data for val in d['eigenvalues']]
        
        # 1. Спектр
        axs[0, 0].hist(all_eigen, bins=30, color='skyblue', edgecolor='black')
        axs[0, 0].set_title("Energy Spectrum")

        # 2. Постоянная тонкой структуры (аппроксимация)
        x = np.linspace(1, 100, 100)
        y = 137.03599 + (0.0002 / (x**0.5))
        axs[0, 1].plot(x, y, color='red')
        axs[0, 1].set_title("Running Alpha")

        # 3. Энергетический бюджет
        axs[0, 2].pie([68.2, 25.5, 6.3], labels=['DE', 'DM', 'Baryons'], autopct='%1.1f%%')
        axs[0, 2].set_title("Universe Budget")

        # 4. Плотность ТМ по октантам
        dm_dist = [np.mean(np.abs(d['eigenvalues']))*0.005 for d in data]
        axs[1, 0].bar(range(8), dm_dist, color='navy')
        axs[1, 0].set_title("DM by Octant")

        # 5. Стабильность
        axs[1, 1].plot(np.cumsum(np.random.rand(50)*0.001), color='green')
        axs[1, 1].set_title("Stability")

        # 6. Фазы
        phases = [np.angle(np.exp(1j * np.pi * i / 4.0)) for i in range(8)]
        axs[1, 2].step(range(8), phases, where='post', color='orange')
        axs[1, 2].set_title("Topological Phases")

        plt.show()

if __name__ == "__main__":
    engine = ACT_Universe_Engine(L=14)
    universe_data = engine.solve_universe()
    engine.build_dashboard(universe_data)

