"""Simple XRD Pattern Calculator - Rust vs Python Comparison (Parallel)"""

from pathlib import Path
import time
import csv
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from xrd_rust_calculator import XRDCalculatorRust
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from threading import Lock

WAVELENGTH = "MoKa"
TWO_THETA_MIN = 2
TWO_THETA_MAX = 60
NUM_WORKERS = 14  # Number of parallel processes

# Lock for thread-safe CSV writing
csv_lock = Lock()

def process_single_file(filepath, structures_dir):
    """Process a single structure file and return results"""
    relative_path = filepath.relative_to(structures_dir)
    
    if len(relative_path.parts) > 1:
        subfolder = str(Path(*relative_path.parts[:-1]))
    else:
        subfolder = "root"
    
    try:
        structure = Structure.from_file(filepath)
        formula = structure.composition.reduced_formula
        
        # Rust calculation
        calc_rust = XRDCalculatorRust(wavelength=WAVELENGTH)
        start = time.perf_counter()
        pattern_rust = calc_rust.get_pattern(structure, two_theta_range=(TWO_THETA_MIN, TWO_THETA_MAX))
        time_rust = time.perf_counter() - start
        
        # Python calculation
        calc_python = XRDCalculator(wavelength=WAVELENGTH)
        start = time.perf_counter()
        pattern_python = calc_python.get_pattern(structure, two_theta_range=(TWO_THETA_MIN, TWO_THETA_MAX))
        time_python = time.perf_counter() - start
        
        row_data = {
            'file': filepath.name,
            'subfolder': subfolder,
            'formula': formula,
            'atoms': structure.num_sites,
            'peaks_rust': len(pattern_rust.x),
            'peaks_python': len(pattern_python.x),
            'time_rust_sec': f"{time_rust:.4f}",
            'time_python_sec': f"{time_python:.4f}",
            'speedup': f"{time_python / time_rust:.2f}"
        }
        
        return ('success', filepath, subfolder, row_data)
        
    except Exception as e:
        error_row = {
            'file': filepath.name,
            'subfolder': subfolder,
            'formula': 'ERROR',
            'atoms': 0,
            'peaks_rust': 0,
            'peaks_python': 0,
            'time_rust_sec': 0,
            'time_python_sec': 0,
            'speedup': 0
        }
        return ('error', filepath, subfolder, error_row, str(e))


if __name__ == '__main__':
    structures_dir = Path(".")
    output_dir = Path("./xrd_results")
    output_dir.mkdir(exist_ok=True)

    summary_file = output_dir / "benchmark_summary.csv"
    csv_headers = ['file', 'subfolder', 'formula', 'atoms', 'peaks_rust', 'peaks_python', 
                   'time_rust_sec', 'time_python_sec', 'speedup']

    with open(summary_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

    print("Searching for structure files recursively...")
    structure_files = []

    structure_files.extend(structures_dir.rglob("*.cif"))
    structure_files.extend(structures_dir.rglob("*.CIF"))
    structure_files.extend(structures_dir.rglob("POSCAR*"))
    structure_files.extend(structures_dir.rglob("CONTCAR*"))

    structure_files = sorted(set(structure_files))

    print(f"\n{'='*70}")
    print(f"Found {len(structure_files)} structure files in all subdirectories")
    print(f"Wavelength: {WAVELENGTH}")
    print(f"2θ range: {TWO_THETA_MIN}° - {TWO_THETA_MAX}°")
    print(f"Parallel workers: {NUM_WORKERS}")
    print(f"Saving real-time results to: {summary_file}")
    print(f"{'='*70}\n")

    subdirs = set()
    for f in structure_files:
        relative_path = f.relative_to(structures_dir)
        if len(relative_path.parts) > 1:
            subdir = relative_path.parts[0]
            subdirs.add(subdir)

    if subdirs:
        print(f"Subdirectories found: {', '.join(sorted(subdirs))}\n")

    results = []
    completed = 0
    total = len(structure_files)
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all jobs
        future_to_file = {
            executor.submit(process_single_file, filepath, structures_dir): filepath 
            for filepath in structure_files
        }
        
        for future in as_completed(future_to_file):
            completed += 1
            result = future.result()
            
            if result[0] == 'success':
                _, filepath, subfolder, row_data = result
                
                # Thread-safe CSV writing
                with csv_lock:
                    with open(summary_file, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=csv_headers)
                        writer.writerow(row_data)
                
                results.append(row_data)
                
                print(f"[{completed}/{total}] {subfolder}/{filepath.name}")
                print(f"  ✓ Rust: {row_data['time_rust_sec']}s ({row_data['peaks_rust']} peaks)")
                print(f"  ✓ Python: {row_data['time_python_sec']}s ({row_data['peaks_python']} peaks)")
                print(f"  → Speedup: {row_data['speedup']}x\n")
                
            else: 
                _, filepath, subfolder, error_row, error_msg = result
                
                print(f"[{completed}/{total}] {subfolder}/{filepath.name}")
                print(f"  ✗ ERROR: {error_msg}\n")
                
                with csv_lock:
                    with open(summary_file, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=csv_headers)
                        writer.writerow(error_row)

    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"\nTotal files processed: {len(results)}")

    if results:
        total_time_rust = sum(float(r['time_rust_sec']) for r in results)
        total_time_python = sum(float(r['time_python_sec']) for r in results)
        avg_speedup = sum(float(r['speedup']) for r in results) / len(results)
        
        print(f"\nTotal calculation time:")
        print(f"  Rust:   {total_time_rust:.2f}s")
        print(f"  Python: {total_time_python:.2f}s")
        print(f"  Average speedup: {avg_speedup:.1f}x")
        
        sorted_by_speedup = sorted(results, key=lambda x: float(x['speedup']), reverse=True)
        
        print(f"\nTop 5 speedups:")
        for i, r in enumerate(sorted_by_speedup[:5], 1):
            print(f"  {i}. {r['file']} ({r['subfolder']}): {r['speedup']}x")
        
        from collections import defaultdict
        by_subfolder = defaultdict(list)
        for r in results:
            by_subfolder[r['subfolder']].append(r)
        
        print(f"\nResults by subfolder:")
        for subfolder, items in sorted(by_subfolder.items()):
            count = len(items)
            avg_speedup_sub = sum(float(r['speedup']) for r in items) / count
            print(f"  {subfolder}: {count} files, avg speedup: {avg_speedup_sub:.1f}x")

    print(f"\n{'='*70}")
    print(f"Results saved to: {summary_file}")
    print(f"{'='*70}\n")
