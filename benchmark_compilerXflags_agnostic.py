#!/usr/bin/env python3
"""
============================================================
C Benchmark Framework: Compiler vs Optimization Profiles
============================================================

Benchmark a single C program across:
- different compilers (rows)
- different optimization profiles (columns)

Profiles represent compiler-independent optimization strategies.
Each compiler maps them to equivalent flags.

Advantages:
- Avoids unsupported flags
- Produces compact CSV tables
- Enables fair cross-compiler comparisons
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

C_FILE = "matrixmult_opt.c"
N_VALUE = 300
RUNS_PER_N = 1

COMPILERS = ["gcc", "icc", "icx"]

# ----------------------------------------------------------
# Flag aliases -> compiler specific flags
# ----------------------------------------------------------

FLAG_ALIASES = {
    "OPT_O3": {"gcc": ["-O3"], "icc": ["-O3"], "icx": ["-O3"]},
    "MATH_LIB": {"gcc": ["-lm"], "icc": ["-lm"], "icx": ["-lm"]},
    "CPU_NATIVE": {"gcc": ["-march=native"], "icc": ["-xHost"], "icx": ["-xHost"]},
    "FAST": {"gcc": ["-Ofast"], "icc": ["-Ofast"], "icx": ["-fast"]},
    "MEMORY_ALIGNMENT": {"gcc": ["-DALIGNED"], "icc": ["-DALIGNED"], "icx": ["-DALIGNED"]},
    "INLINE": {"gcc": ["-DNOFUNCCALL"], "icc": ["-DNOFUNCCALL"], "icx": ["-DNOFUNCCALL"]},
    "LINKING": {"gcc": ["-flto"], "icc": ["-ipo"], "icx": ["-ipo"]}
}

# ----------------------------------------------------------
# Optimization configurations (CSV columns)
# ----------------------------------------------------------

OPT_CONFIGS = [
    ["OPT_O3", "MATH_LIB"],
    ["OPT_O3", "MATH_LIB", "CPU_NATIVE"],
    ["OPT_O3", "MATH_LIB", "CPU_NATIVE", "FAST"],
    ["OPT_O3", "MATH_LIB", "CPU_NATIVE", "FAST", "MEMORY_ALIGNMENT"],
    ["OPT_O3", "MATH_LIB", "CPU_NATIVE", "FAST", "MEMORY_ALIGNMENT", "INLINE"],
    ["OPT_O3", "MATH_LIB", "CPU_NATIVE", "FAST", "MEMORY_ALIGNMENT", "INLINE", "LINKING"]
]

# Output folder
OUTPUT_FOLDER = Path(__file__).parent / "benchmarks"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ==========================================================
# ============== FUNCTIONS / CLASSES ======================
# ==========================================================

def expand_flags(compiler, config):
    """Translate flag aliases to compiler flags."""
    flags = []
    for alias in config:
        flags.extend(FLAG_ALIASES.get(alias, {}).get(compiler, []))
    return flags

class Compiler:
    """Handles compilation of C source file with given compiler and flags."""

    @staticmethod
    def compile(source_file: Path, compiler: str, flags: list) -> Path:
        source_file = source_file.resolve()
        safe_flags = "_".join(f.replace("-", "") for f in flags)
        if safe_flags == "":
            safe_flags = "default"
        binary = OUTPUT_FOLDER / f"{source_file.stem}_{compiler}_{safe_flags}"

        print(f"[INFO] Compiling '{source_file}' with {compiler} flags {flags}")

        # Separate -lm to put at the end for ICC/ICX
        normal_flags = [f for f in flags if f != "-lm"]
        lm_flags = ["-lm"] if "-lm" in flags else []

        cmd = [compiler, str(source_file), "-o", str(binary)] + normal_flags + lm_flags

        try:
            if compiler in ["icc", "icx"]:
                compile_command = " ".join(cmd)
                subprocess.run(
                    ["bash", "-c", f"source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 && {compile_command}"],
                    check=True, capture_output=True, text=True
                )
            else:
                subprocess.run(cmd, check=True, capture_output=True, text=True)

            print(f"[OK] Compilation succeeded: {binary}")

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Compilation failed for {compiler}")
            print(e.stdout)
            print(e.stderr)
            sys.exit(1)

        return binary

class Executor:
    """Runs compiled binary and extracts execution time."""

    @staticmethod
    def run(binary: Path, n: int) -> float:
        binary_path = str(binary.resolve())
        print(f"[INFO] Running '{binary_path}' with n={n}")
        try:
            result = subprocess.run([binary_path, str(n)],
                                    capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            print(f"[ERROR] Execution failed for {binary_path}")
            sys.exit(1)

        for line in result.stdout.splitlines():
            if "Execution Time" in line:
                print(f"[INFO] Output received: {line.strip()}")
                return float(line.split()[-2])

        print("[ERROR] Execution time not found in program output")
        sys.exit(1)

class Benchmark:
    """Benchmark fixed program across compilers and optimization configs."""

    def __init__(self, c_file, n_value, runs, compilers, opt_configs):
        self.c_file = Path(c_file)
        self.n_value = n_value
        self.runs = runs
        self.compilers = compilers
        self.opt_configs = opt_configs

    def run(self):
        results = {}
        for compiler in self.compilers:
            print(f"\n[INFO] Benchmarking compiler: {compiler}")
            results[compiler] = {}
            for config in self.opt_configs:
                config_name = "_".join(config)
                flags = expand_flags(compiler, config)
                binary = Compiler.compile(self.c_file, compiler, flags)
                timings = [Executor.run(binary, self.n_value) for _ in range(self.runs)]
                avg_time = statistics.mean(timings)
                results[compiler][config_name] = avg_time
                print(f"[OK] Compiler={compiler}, Config={config_name} -> Avg={avg_time:.4f}s")
        return results

class CSVWriter:
    """Writes results to CSV with compilers as rows."""

    @staticmethod
    def write(program_file, n_value, compilers, configs, data):
        program_stem = Path(program_file).stem
        config_names = ["_".join(c) for c in configs]
        compilers_str = "_".join(compilers)
        safe_config_str = re.sub(r"[^\w]+", "_", "_".join(config_names))
        output_csv = OUTPUT_FOLDER / f"results|{program_stem}|n{n_value}|{compilers_str}|{safe_config_str}.csv"

        headers = ["compiler"] + config_names
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for compiler, times in data.items():
                row = [compiler] + [times.get(cfg, "") for cfg in config_names]
                writer.writerow(row)

        print(f"[INFO] CSV results written to '{output_csv}'")

# ==========================================================
# ====================== MAIN ==============================
# ==========================================================

def main():
    print("[START] Benchmark process initiated...")
    benchmark = Benchmark(C_FILE, N_VALUE, RUNS_PER_N, COMPILERS, OPT_CONFIGS)
    results = benchmark.run()
    CSVWriter.write(C_FILE, N_VALUE, COMPILERS, OPT_CONFIGS, results)
    print("[FINISH] Benchmark complete.")

if __name__ == "__main__":
    main()