#!/usr/bin/env python3
"""
============================================================
C Benchmark Framework with Absolute Paths & Dynamic CSV
============================================================

HOW TO USE:

1. Make sure your C programs:
   - Accept n as command line argument:
     int n = atoi(argv[1]);

   - Print execution time like:
     Execution Time: X.XXXX seconds

2. Place 'build_matrixmult.sh' in the same folder as
   'matrixmult_library.c'.

3. Edit ONLY the CONFIGURATION SECTION below.

4. Run:
   python3 benchmark.py

Results will be written to a CSV whose name reflects the config.
============================================================
"""

# ==========================================================
# ================== CONFIGURATION =========================
# ==========================================================

import re
from pathlib import Path

C_FILES = [
    #"matrixmult.c",
    "matrixmult_opt.c",
    "matrixmult_opt|NOALIGN.c",
    "matrixmult_library.c"
    # Add your OpenBLAS version here
]

N_VALUES = [1000, 2000, 3000, 5000, 10000]

RUNS_PER_N = 1

COMPILER = "gcc"

COMPILER_FLAGS = [
    "-O3",
    "-lm",
    "-march=native"
]

# generate a safe, descriptive CSV filename
file_stems = "_".join(Path(f).stem for f in C_FILES)
n_values_str = "_".join(str(n) for n in N_VALUES)
flags_str = "_".join(f.replace("-", "") for f in COMPILER_FLAGS)
safe_flags_str = re.sub(r"[^\w]+", "_", flags_str)

OUTPUT_CSV = f"results|{COMPILER}|{file_stems}|{n_values_str}|{safe_flags_str}.csv"

# ==========================================================
# ============== DO NOT MODIFY BELOW ======================
# ==========================================================

import subprocess
import csv
import statistics
import sys
import os


class _Compiler:
    """Handles compilation of C source files."""

    @staticmethod
    def compile(source_file: Path):

        source_file = source_file.resolve()

        # Special case: use build script for matrixmult_library.c
        if source_file.name == "matrixmult_library.c":

            build_script = source_file.parent / "build_matrixmult.sh"

            if not build_script.exists():
                print(f"[ERROR] Build script not found: {build_script}")
                sys.exit(1)

            print(f"[INFO] Running build script for {source_file} ...")

            try:
                subprocess.run([str(build_script)], check=True)
            except subprocess.CalledProcessError:
                print(f"[ERROR] Build script failed for {source_file}")
                sys.exit(1)

            binary = source_file.parent / "matrixmult"

            if not binary.exists():
                print(f"[ERROR] Expected executable not found: {binary}")
                sys.exit(1)

            print(f"[OK] Build script completed: {binary}")
            return binary

        binary = source_file.with_suffix("").resolve()

        print(f"[INFO] Compiling '{source_file}' -> '{binary}' ...")

        cmd = [COMPILER, str(source_file), "-o", str(binary)] + COMPILER_FLAGS

        try:

            # If using icx, source oneAPI environment first
            if COMPILER == "icx":

                compile_command = " ".join(cmd)

                subprocess.run(
                    [
                        "bash",
                        "-c",
                        f"source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 && {compile_command}",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )

            else:

                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                )

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

        binary_path = str(binary.resolve())

        print(f"[INFO] Running '{binary_path}' with n={n} ...")

        try:

            result = subprocess.run(
                [binary_path, str(n)],
                capture_output=True,
                text=True,
                check=True,
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
    """Core benchmarking logic."""

    def __init__(self, files, n_values, runs):
        self.files = files
        self.n_values = n_values
        self.runs = runs

    def run(self):

        results = {}

        for file in self.files:

            source_path = Path(file).resolve()

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