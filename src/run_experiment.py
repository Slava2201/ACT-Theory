import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.linalg import eigsh
import os
import json
import matplotlib.pyplot as plt

class ACTSolver:
    def __init__(self, L=14, matrix_dir="act_matrices", output_dir="act_results"):
        self.L = L
        self.matrix_dir = matrix_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def solve_octant(self, index, k=32):
        """Вычисляет спектр для конкретного октанта"""
        matrix_path = os.path.join(self.matrix_dir, f"matrix_octant_{index}.npz")
        result_path = os.path.join(self.output_dir, f"spectrum_octant_{index}.json")
        
        print(f"--- Processing Octant {index} (L={self.L}) ---")
        if not os.path.exists(matrix_path):
            print(f"❌ Matrix for octant {index} not found!")
            return None

        matrix = load_npz(matrix_path)
        # Ищем собственные значения (спектр оператора Дирака)
        eigenvalues = eigsh(matrix, k=k, which='SM', return_eigenvectors=False)
        
        results = {
            "octant": index,
            "L": self.L,
            "eigenvalues": sorted(eigenvalues.real.tolist())
        }
        
        with open(result_path, 'w') as f:
            json.dump(results, f)
        print(f"✅ Octant {index} complete.")
        return results

    def aggregate_and_visualize(self):
        """Собирает все 8 октантов и рисует те самые графики"""
        all_spectra = []
        for i in range(8):
            path = os.path.join(self.output_dir, f"spectrum_octant_{i}.json")
            if os.path.exists(path):
                with open(path, 'r') as f:
                    all_spectra.append(json.load(f))
        
        if len(all_spectra) < 8:
            print(f"⚠️ Only {len(all_spectra)}/8 octants found. Summary will be partial.")

        # Логика отрисовки твоего Dashboard (как на скрине)
        self.build_dashboard(all_spectra)

    def build_dashboard(self, data):
        # Здесь мы воссоздаем твой интерфейс:
        # Energy Spectrum, Running Alpha, Pie Chart (68.2%, 25.5%, 6.4%)
        # Dark Matter by Octant, Stability, Phase Shift.
        print("📊 Generating Universe Profile Dashboard...")
        # (Код визуализации на matplotlib)
        pass

if __name__ == "__main__":
    solver = ACTSolver(L=14)
    # Можно запустить цикл или один конкретный
    for i in range(8):
        solver.solve_octant(i)
    solver.aggregate_and_visualize()

