#!/usr/bin/env python3
"""
============================================================
C Benchmark Framework: Compiler vs Optimization Profiles
============================================================

Benchmark a single C program across:
- different compilers
- different optimization profiles

CSV Output:
    Rows    -> optimization configurations
    Columns -> compilers
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

C_FILE = "matrixmult_opt.c"

N_VALUE = 300
RUNS_PER_N = 1
N_THREADS = 24

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
    "LINKING": {"gcc": ["-flto"], "icc": ["-ipo"], "icx": ["-ipo"]},
    "OPENMP": {"gcc": ["-fopenmp"], "icc": ["-qopenmp"], "icx": ["-fopenmp"]},
}

# ----------------------------------------------------------
# Optimization configurations
# ----------------------------------------------------------

OPT_CONFIGS = [
    #["OPT_O3", "MATH_LIB"],
    #["OPT_O3", "MATH_LIB", "CPU_NATIVE"],
    #["OPT_O3", "MATH_LIB", "CPU_NATIVE", "FAST"],
    #["OPT_O3", "MATH_LIB", "CPU_NATIVE", "FAST", "MEMORY_ALIGNMENT"],
    ["OPT_O3", "MATH_LIB", "CPU_NATIVE", "FAST", "MEMORY_ALIGNMENT", "INLINE"],
    ["OPT_O3", "MATH_LIB", "CPU_NATIVE", "FAST", "MEMORY_ALIGNMENT", "INLINE", "LINKING"],
    
    #["OPT_O3", "OPENMP", "MATH_LIB", "CPU_NATIVE", "FAST", "MEMORY_ALIGNMENT"],
    #["OPT_O3", "OPENMP", "MATH_LIB", "CPU_NATIVE", "FAST", "MEMORY_ALIGNMENT", "LINKING"]
]

OUTPUT_FOLDER = Path(__file__).parent / "benchmarks"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ==========================================================
# ================== UTILITY FUNCTIONS =====================
# ==========================================================

def expand_flags(compiler, config):
    flags = []
    for alias in config:
        flags.extend(FLAG_ALIASES.get(alias, {}).get(compiler, []))
    return flags


# ==========================================================
# ===================== COMPILER ===========================
# ==========================================================

class Compiler:

    @staticmethod
    def compile(source_file: Path, compiler: str, flags: list) -> Path:

        source_file = source_file.resolve()
        safe_flags = "_".join(f.replace("-", "") for f in flags) or "default"
        binary = OUTPUT_FOLDER / f"{source_file.stem}_{compiler}_{safe_flags}"

        print(f"\n[COMPILING] {compiler}")
        print(f"  Source file : {source_file}")
        print(f"  Flags       : {' '.join(flags)}")

        normal_flags = [f for f in flags if f != "-lm"]
        lm_flags = ["-lm"] if "-lm" in flags else []

        cmd = [compiler, str(source_file), "-o", str(binary)] + normal_flags + lm_flags

        try:

            if compiler in ["icc", "icx"]:

                compile_command = " ".join(cmd)

                subprocess.run(
                    ["bash", "-c",
                     f"source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 && {compile_command}"],
                    check=True,
                    capture_output=True,
                    text=True
                )

            else:

                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )

            print(f"  ✔ Compilation successful → {binary}")

        except subprocess.CalledProcessError as e:

            stderr = e.stderr or ""
            stdout = e.stdout or ""

            print(f"\n[ERROR] Compilation failed with compiler: {compiler}")

            if "omp_" in stderr or "omp_" in stdout:

                print("\n[DIAGNOSTIC] OpenMP symbols detected but OpenMP is not enabled.")
                print("Add the OPENMP alias to your configuration.\n")

                print("Example:")
                print('    ["OPT_O3", "OPENMP", "MATH_LIB"]')

                print("\nOPENMP mappings:")
                print("  gcc  -> -fopenmp")
                print("  icc  -> -qopenmp")
                print("  icx  -> -fopenmp")

            else:

                print(stdout)
                print(stderr)

            sys.exit(1)

        return binary


# ==========================================================
# ===================== EXECUTOR ===========================
# ==========================================================

class Executor:

    @staticmethod
    def run(binary: Path, n: int, config: list) -> float:

        binary_path = str(binary.resolve())

        print(f"  Running benchmark (n={n})...")

        use_openmp = "OPENMP" in config

        if use_openmp:
            cmd = [binary_path, str(n), str(N_THREADS)]
        else:
            cmd = [binary_path, str(n)]

        try:

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

        except subprocess.CalledProcessError as e:

            stdout = e.stdout or ""
            stderr = e.stderr or ""

            print(f"\n[ERROR] Execution failed for {binary_path}")

            if use_openmp:

                print("\n[DIAGNOSTIC] Possible argument overfeeding detected.")
                print("The benchmark passed two arguments:")
                print(f"    N={n}, N_THREADS={N_THREADS}")

                print("\nBut the program may expect only one argument.")

                print("\nSolutions:")
                print("  • Modify the C program to accept NUM_THREADS")
                print("  • Remove OPENMP from this configuration")

            print("\nProgram output:")
            print(stdout)
            print(stderr)

            sys.exit(1)

        for line in result.stdout.splitlines():

            if "Execution Time" in line or "Tempo di esecuzione" in line:

                time = float(line.split()[-2])

                print(f"  ✔ Execution time: {time:.6f} s")

                return time

        print("[ERROR] Execution time not found in program output")

        sys.exit(1)


# ==========================================================
# ===================== BENCHMARK ==========================
# ==========================================================

class Benchmark:

    def __init__(self, c_file, n_value, runs, compilers, opt_configs):

        self.c_file = Path(c_file)
        self.n_value = n_value
        self.runs = runs
        self.compilers = compilers
        self.opt_configs = opt_configs

    def run(self):

        print("\n================================================")
        print("Benchmark started")
        print("Program:", self.c_file)
        print("Matrix size (n):", self.n_value)
        print("Compilers:", ", ".join(self.compilers))
        print("Configurations:", len(self.opt_configs))
        print("================================================")

        results = {}

        for config in self.opt_configs:

            config_name = "_".join(config)

            print(f"\n[CONFIGURATION] {config_name}")

            results[config_name] = {}

            for compiler in self.compilers:

                flags = expand_flags(compiler, config)

                binary = Compiler.compile(self.c_file, compiler, flags)

                timings = [
                    Executor.run(binary, self.n_value, config)
                    for _ in range(self.runs)
                ]

                avg_time = statistics.mean(timings)

                results[config_name][compiler] = avg_time

                print(f"  → Average time with {compiler}: {avg_time:.6f} s")

        print("\nBenchmark finished successfully.")

        return results


# ==========================================================
# ====================== CSV WRITER ========================
# ==========================================================

class CSVWriter:

    @staticmethod
    def write(program_file, n_value, compilers, configs, data):

        program_stem = Path(program_file).stem
        compilers_str = "_".join(compilers)

        unique_flags = sorted({flag for config in configs for flag in config})
        flags_str = "+".join(unique_flags)

        output_csv = OUTPUT_FOLDER / (
            f"results|{program_stem}|n{n_value}|{compilers_str}|{flags_str}.csv"
        )

        headers = ["configuration"] + compilers

        with open(output_csv, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow(headers)

            for config in configs:

                config_name = "_".join(config)

                row = [config_name]

                for compiler in compilers:

                    row.append(data.get(config_name, {}).get(compiler, ""))

                writer.writerow(row)

        print("\n[CSV OUTPUT]")
        print(f"Results written to: {output_csv}")


# ==========================================================
# ========================= MAIN ===========================
# ==========================================================

def main():

    print("\n[START] Launching benchmark framework")

    benchmark = Benchmark(
        C_FILE,
        N_VALUE,
        RUNS_PER_N,
        COMPILERS,
        OPT_CONFIGS
    )

    results = benchmark.run()

    CSVWriter.write(
        C_FILE,
        N_VALUE,
        COMPILERS,
        OPT_CONFIGS,
        results
    )

    print("\n[FINISH] Benchmark process completed successfully.\n")


if __name__ == "__main__":
    main()