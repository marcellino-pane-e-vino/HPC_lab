#!/usr/bin/env python3
"""
============================================================
C Benchmark Framework: Compiler vs Flags
============================================================

Benchmark a single C program across:
- different compilers (rows)
- different compiler flag sets (columns)

Fixed program and fixed dimension(s).

HOW TO USE:

1. Ensure your C program:
      - Accepts n as command line argument:
            int n = atoi(argv[1]);
      - Prints execution time like:
            Execution Time: X.XXXX seconds

2. Place any required build scripts (e.g., for matrixmult_library.c) in the program folder.

3. Edit ONLY THE CONFIGURATION SECTION below.

4. Run:
      python3 benchmark_compiler_flags.py

Results will be written to a CSV named <program>_compiler_flags.csv
============================================================
"""

import subprocess
import statistics
import csv
import sys
from pathlib import Path

# ==========================================================
# ================== CONFIGURATION =========================
# ==========================================================

# Fixed program
C_FILE = "matrixmult_library.c"

# Fixed dimension(s)
N_VALUE = 2000

# Number of runs to average
RUNS_PER_N = 1

# Rows: compilers to test
COMPILERS = ["gcc", "icc", "icx"]

# Columns: sets of compiler flags to test
COMPILER_FLAGS_LIST = [
    ["-O2", "-lm"],
    ["-O3", "-lm"],
    ["-O3", "-lm", "-march=native"]
]

# ==========================================================
# ============== DO NOT MODIFY BELOW ======================
# ==========================================================

class Compiler:
    """Handles compilation of C source file with given compiler and flags."""

    @staticmethod
    def compile(source_file: Path, compiler: str, flags: list) -> Path:
        source_file = source_file.resolve()

        # Special case: build script for matrixmult_library.c
        if source_file.name == "matrixmult_library.c":
            build_script = source_file.parent / "build_matrixmult.sh"
            if build_script.exists():
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
                return binary

        # Normal compilation
        binary_name = f"{source_file.stem}_{compiler}_" + "_".join(f.replace("-", "") for f in flags)
        binary = source_file.parent / binary_name
        print(f"[INFO] Compiling '{source_file}' with {compiler} flags {flags} -> '{binary}'")
        cmd = [compiler, str(source_file), "-o", str(binary)] + flags
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Compilation failed for {source_file} with {compiler}")
            print(e.stdout)
            print(e.stderr)
            sys.exit(1)
        return binary


class Executor:
    """Runs compiled binary and extracts execution time."""

    @staticmethod
    def run(binary: Path, n: int) -> float:
        print(f"[INFO] Running '{binary}' with n={n}")
        try:
            result = subprocess.run([str(binary), str(n)],
                                    capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            print(f"[ERROR] Execution failed for {binary}")
            sys.exit(1)

        for line in result.stdout.splitlines():
            if "Execution Time" in line:
                return float(line.split()[-2])

        print(f"[ERROR] Execution time not found in output of {binary}")
        sys.exit(1)


class Benchmark:
    """Benchmark fixed program across compilers and flags."""

    def __init__(self, c_file, n_value, runs, compilers, flags_list):
        self.c_file = Path(c_file)
        self.n_value = n_value
        self.runs = runs
        self.compilers = compilers
        self.flags_list = flags_list

    def run(self):
        results = {}  # {compiler: {flags_str: avg_time}}

        for compiler in self.compilers:
            print(f"\n[INFO] Benchmarking with compiler: {compiler}")
            results[compiler] = {}

            for flags in self.flags_list:
                flags_str = "_".join(flags).replace("-", "")
                timings = []

                binary = Compiler.compile(self.c_file, compiler, flags)

                for _ in range(self.runs):
                    t = Executor.run(binary, self.n_value)
                    timings.append(t)

                avg_time = statistics.mean(timings)
                results[compiler][flags_str] = avg_time
                print(f"[OK] Compiler={compiler}, Flags={flags} -> Avg Time={avg_time:.4f}s")

        return results


class CSVWriter:
    """Writes results to CSV with compilers as rows, flags as columns."""

    @staticmethod
    def write(filename, data, flags_list):
        headers = ["compiler"] + ["_".join(f).replace("-", "") for f in flags_list]

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for compiler, times in data.items():
                row = [compiler] + [times.get("_".join(f).replace("-", ""), "") for f in flags_list]
                writer.writerow(row)


# ==========================================================
# ====================== MAIN ==============================
# ==========================================================

def main():
    print("[START] Benchmark process initiated...")
    benchmark = Benchmark(C_FILE, N_VALUE, RUNS_PER_N, COMPILERS, COMPILER_FLAGS_LIST)
    results = benchmark.run()

    csv_file = f"{Path(C_FILE).stem}_compiler_flags.csv"
    CSVWriter.write(csv_file, results, COMPILER_FLAGS_LIST)
    print(f"[FINISH] Benchmark complete. Results saved to '{csv_file}'")


if __name__ == "__main__":
    main()