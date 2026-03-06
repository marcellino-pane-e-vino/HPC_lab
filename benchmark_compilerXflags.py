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

Results will be written to a CSV with descriptive name including compiler sets, flags, and dimension.
============================================================
"""

import subprocess
import statistics
import csv
import sys
import re
from pathlib import Path

# ==========================================================
# ================== CONFIGURATION =========================
# ==========================================================

# Fixed program
C_FILE = "matrixmult_opt.c"

# Fixed dimension(s)
N_VALUE = 8000

# Number of runs to average
RUNS_PER_N = 1

# Rows: compilers to test
COMPILERS = ["gcc", "icc", "icx"]

# Columns: sets of compiler flags to test
COMPILER_FLAGS_LIST = [
    ["-lm", "-march=native"],
    ["-O1", "-lm", "-march=native"],
    ["-O2", "-lm", "-march=native"],
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
        safe_flags = "_".join(f.replace("-", "") for f in flags)
        binary_name = f"{source_file.stem}_{compiler}_{safe_flags}"
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
    def write(program_file, n_value, compilers, flags_list, data):
        # generate descriptive filename
        program_stem = Path(program_file).stem
        compilers_str = "_".join(compilers)
        flags_str = "_".join("_".join(f).replace("-", "") for f in flags_list)
        safe_flags_str = re.sub(r"[^\w]+", "_", flags_str)
        output_csv = f"results|{program_stem}|n{n_value}|{compilers_str}|{safe_flags_str}.csv"

        headers = ["compiler"] + ["_".join(f).replace("-", "") for f in flags_list]

        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for compiler, times in data.items():
                row = [compiler] + [times.get("_".join(f).replace("-", ""), "") for f in flags_list]
                writer.writerow(row)

        print(f"[INFO] CSV results written to '{output_csv}'")


# ==========================================================
# ====================== MAIN ==============================
# ==========================================================

def main():
    print("[START] Benchmark process initiated...")
    benchmark = Benchmark(C_FILE, N_VALUE, RUNS_PER_N, COMPILERS, COMPILER_FLAGS_LIST)
    results = benchmark.run()
    CSVWriter.write(C_FILE, N_VALUE, COMPILERS, COMPILER_FLAGS_LIST, results)
    print("[FINISH] Benchmark complete.")


if __name__ == "__main__":
    main()