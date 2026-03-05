#!/usr/bin/env python3
"""
============================================================
C Benchmark Framework with Absolute Paths
============================================================

HOW TO USE:

1. Make sure your C programs:
      - Accept n as command line argument:
            int n = atoi(argv[1]);
      - Print execution time like:
            Execution Time: X.XXXX seconds

2. Edit ONLY the CONFIGURATION SECTION below.

3. Run:
      python3 benchmark.py

Results will be written to results.csv
============================================================
"""

# ==========================================================
# ================== CONFIGURATION =========================
# ==========================================================

C_FILES = [
    "matrixmult.c",
    "matrixmult_opt.c",
]

N_VALUES = [1000, 2000, 3000]

RUNS_PER_N = 1

OUTPUT_CSV = "results.csv"

COMPILER = "gcc"
COMPILER_FLAGS = ["-O3", "-lm"]

# ==========================================================
# ============== DO NOT MODIFY BELOW ======================
# ==========================================================

import subprocess
import csv
import statistics
from pathlib import Path
import sys
import os


class _Compiler:
    """Handles compilation of C source files."""

    @staticmethod
    def compile(source_file: Path) -> Path:
        binary = source_file.with_suffix("").resolve()  # absolute path to binary
        print(f"[INFO] Compiling '{source_file}' -> '{binary}' ...")

        cmd = [COMPILER, str(source_file), "-o", str(binary), "-O3", "-lm"]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"[OK] Compilation succeeded: {binary}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Compilation failed for {source_file}")
            print("[COMPILER OUTPUT]")
            print(e.stdout)
            print(e.stderr)
            sys.exit(1)

        return binary


class _Executor:
    """Runs compiled binaries and extracts execution time."""

    @staticmethod
    def run(binary: Path, n: int) -> float:
        binary_path = str(binary.resolve())  # absolute path
        print(f"[INFO] Running '{binary_path}' with n={n} ...")
        try:
            result = subprocess.run(
                [binary_path, str(n)],
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError:
            print(f"[ERROR] Execution failed for {binary_path} with n={n}")
            print("Check your C program for runtime errors.")
            sys.exit(1)

        for line in result.stdout.splitlines():
            if "Execution Time" in line:
                print(f"[INFO] Output received: {line.strip()}")
                return float(line.split()[-2])

        print(f"[ERROR] Execution time not found in output of {binary_path}")
        print("Make sure your C program prints 'Execution Time: X.XXXX seconds'")
        sys.exit(1)


class _BenchmarkEngine:
    """Core benchmarking logic (hidden from user)."""

    def __init__(self, files, n_values, runs):
        self.files = files
        self.n_values = n_values
        self.runs = runs

    def run(self):
        results = {}

        for file in self.files:
            source_path = Path(file).resolve()  # absolute path
            if not source_path.exists():
                print(f"[ERROR] File not found: {file}")
                sys.exit(1)

            print(f"\n[INFO] Benchmarking '{source_path}' ...")

            binary = _Compiler.compile(source_path)
            results[file] = self._benchmark_file(binary)

        return results

    def _benchmark_file(self, binary):
        file_results = {}

        for n in self.n_values:
            print(f"[INFO] Starting {self.runs} run(s) for n={n} ...")
            timings = []

            for run_idx in range(1, self.runs + 1):
                print(f"[INFO] Run {run_idx}/{self.runs} ...")
                t = _Executor.run(binary, n)
                timings.append(t)

            avg_time = statistics.mean(timings)
            file_results[n] = avg_time

            print(f"[OK] n={n:<6} avg_time={avg_time:.4f}s")

        return file_results


class _CSVWriter:
    """Handles writing benchmark results to CSV."""

    @staticmethod
    def write(filename, data, n_values):
        print(f"[INFO] Writing results to '{filename}' ...")
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(["file"] + n_values)

            for file, results in data.items():
                row = [file] + [results[n] for n in n_values]
                writer.writerow(row)

        print(f"[OK] CSV file saved: '{filename}'")


# ==========================================================
# ====================== MAIN ==============================
# ==========================================================

def main():
    print("[START] Benchmark process initiated...\n")

    engine = _BenchmarkEngine(C_FILES, N_VALUES, RUNS_PER_N)
    results = engine.run()

    _CSVWriter.write(OUTPUT_CSV, results, N_VALUES)

    print(f"\n[FINISH] Benchmark complete. Results saved to '{OUTPUT_CSV}'")


if __name__ == "__main__":
    main()