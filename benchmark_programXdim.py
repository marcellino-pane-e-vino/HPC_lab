#!/usr/bin/env python3
"""
============================================================
C Benchmark Framework
Program vs Problem Size
============================================================

Benchmark multiple implementations across different
problem sizes with automatic compiler/runtime handling.

Features
--------
• Program type abstraction (sequential, OpenMP, MPI, CUDA)
• Layered flag system
• Automatic compiler selection
• Clean execution modes
• CSV output for benchmarking data
============================================================
"""

import subprocess
import statistics
import csv
import sys
from pathlib import Path

# ==========================================================
# ====================== CONFIGURATION =====================
# ==========================================================

# ----------------------------------------------------------
# Programs to benchmark
# ----------------------------------------------------------

PROGRAMS = [

    {
        "file": "matrixmult_opt.c",
        "type": "sequential"
    },

    {
        "file": "omp_matrixmult.c",
        "type": "openmp"
    }

]

# ----------------------------------------------------------
# Benchmark parameters
# ----------------------------------------------------------

N_VALUES = [1000, 2000, 3000]

RUNS_PER_N = 1

N_THREADS = 24

# ----------------------------------------------------------
# Default compiler
# ----------------------------------------------------------

DEFAULT_COMPILER = "gcc"

# ----------------------------------------------------------
# Global optimization flags
# ----------------------------------------------------------

GLOBAL_FLAGS = [
    "OPT_O3",
    "MATH_LIB",
    "CPU_NATIVE"
]

# ----------------------------------------------------------
# Program type configuration
# ----------------------------------------------------------

PROGRAM_TYPES = {

    "sequential": {

        "compiler": DEFAULT_COMPILER,

        "compile_prefix": [],
        "compile_suffix": [],

        "run_mode": "normal"
    },

    "openmp": {

        "compiler": DEFAULT_COMPILER,

        "compile_prefix": ["OPENMP"],
        "compile_suffix": [],

        "run_mode": "openmp"
    },

    "mpi": {

        "compiler": "mpicc",

        "compile_prefix": [],
        "compile_suffix": [],

        "run_mode": "mpi"
    },

    "cuda": {

        "compiler": "nvcc",

        "compile_prefix": [],
        "compile_suffix": [],

        "run_mode": "cuda"
    }
}

# ----------------------------------------------------------
# Flag aliases
# ----------------------------------------------------------

FLAG_ALIASES = {

    "OPT_O3": {
        "gcc": ["-O3"],
        "icc": ["-O3"],
        "icx": ["-O3"],
        "nvcc": ["-O3"]
    },

    "MATH_LIB": {
        "gcc": ["-lm"],
        "icc": ["-lm"],
        "icx": ["-lm"]
    },

    "CPU_NATIVE": {
        "gcc": ["-march=native"],
        "icc": ["-xHost"],
        "icx": ["-xHost"]
    },

    "FAST": {
        "gcc": ["-Ofast"],
        "icc": ["-Ofast"],
        "icx": ["-fast"]
    },

    "OPENMP": {
        "gcc": ["-fopenmp"],
        "icc": ["-qopenmp"],
        "icx": ["-fopenmp"]
    }

}

# ==========================================================
# ===================== OUTPUT SETUP =======================
# ==========================================================

OUTPUT_FOLDER = Path(__file__).parent / "benchmarks"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

program_names = "_".join(Path(p["file"]).stem for p in PROGRAMS)
n_values_str = "_".join(str(n) for n in N_VALUES)

OUTPUT_CSV = OUTPUT_FOLDER / f"results_{program_names}_{n_values_str}.csv"

# ==========================================================
# ===================== FLAG UTILITIES =====================
# ==========================================================

def expand_flags(compiler, config):

    flags = []

    for alias in config:

        flags.extend(
            FLAG_ALIASES.get(alias, {}).get(compiler, [])
        )

    return flags


def build_flags(program):

    ptype = program["type"]

    type_info = PROGRAM_TYPES[ptype]

    flags = []

    flags += type_info.get("compile_prefix", [])
    flags += GLOBAL_FLAGS
    flags += program.get("flags", [])
    flags += type_info.get("compile_suffix", [])

    compiler = type_info.get("compiler", DEFAULT_COMPILER)

    return expand_flags(compiler, flags)


# ==========================================================
# ======================== COMPILER ========================
# ==========================================================

class Compiler:

    @staticmethod
    def compile(program):

        source_file = Path(program["file"]).resolve()

        if not source_file.exists():

            print(f"[ERROR] Source file not found: {source_file}")
            sys.exit(1)

        ptype = program["type"]

        type_info = PROGRAM_TYPES[ptype]

        compiler = type_info.get("compiler", DEFAULT_COMPILER)

        flags = build_flags(program)

        safe_flags = "_".join(f.replace("-", "") for f in flags)

        binary = OUTPUT_FOLDER / f"{source_file.stem}_{safe_flags}"

        print("\n[COMPILING]")
        print(f"  Program  : {source_file.name}")
        print(f"  Type     : {ptype}")
        print(f"  Compiler : {compiler}")
        print(f"  Flags    : {' '.join(flags)}")

        normal_flags = [f for f in flags if f != "-lm"]
        lm_flags = ["-lm"] if "-lm" in flags else []

        cmd = [compiler, str(source_file), "-o", str(binary)] + normal_flags + lm_flags

        try:

            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            print(f"  ✔ Compilation successful → {binary}")

        except subprocess.CalledProcessError as e:

            print("\n[ERROR] Compilation failed\n")

            print(e.stdout)
            print(e.stderr)

            sys.exit(1)

        return binary


# ==========================================================
# ======================== EXECUTOR ========================
# ==========================================================

class Executor:

    @staticmethod
    def run(program, binary, n):

        ptype = program["type"]

        run_mode = PROGRAM_TYPES[ptype]["run_mode"]

        print(f"  Running benchmark (n={n})...")

        if run_mode == "openmp":

            cmd = [str(binary), str(n), str(N_THREADS)]

        elif run_mode == "mpi":

            cmd = ["mpirun", "-np", str(N_THREADS), str(binary), str(n)]

        else:

            cmd = [str(binary), str(n)]

        try:

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

        except subprocess.CalledProcessError as e:

            print("\n[ERROR] Execution failed\n")

            print(e.stdout)
            print(e.stderr)

            sys.exit(1)

        for line in result.stdout.splitlines():

            if "Execution Time" in line or "Tempo di esecuzione" in line:

                t = float(line.split()[-2])

                print(f"  ✔ Execution time: {t:.6f} s")

                return t

        print("[ERROR] Execution time not found in output")

        sys.exit(1)


# ==========================================================
# ======================= BENCHMARK ========================
# ==========================================================

class Benchmark:

    def __init__(self, programs, n_values, runs):

        self.programs = programs
        self.n_values = n_values
        self.runs = runs

    def run(self):

        print("\n================================================")
        print("Benchmark started")
        print("Programs:", ", ".join(p["file"] for p in self.programs))
        print("Matrix sizes:", ", ".join(map(str, self.n_values)))
        print("================================================")

        results = {}

        for program in self.programs:

            file = program["file"]

            print(f"\n[PROGRAM] {file}")

            binary = Compiler.compile(program)

            program_results = {}

            for n in self.n_values:

                timings = [

                    Executor.run(program, binary, n)

                    for _ in range(self.runs)

                ]

                avg = statistics.mean(timings)

                program_results[n] = avg

                print(f"  → Average time: {avg:.6f} s")

            results[file] = program_results

        print("\nBenchmark finished successfully.")

        return results


# ==========================================================
# ======================== CSV WRITER ======================
# ==========================================================

class CSVWriter:

    @staticmethod
    def write(filename, data, n_values):

        print("\n[CSV OUTPUT]")

        with open(filename, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow(["program"] + n_values)

            for file, results in data.items():

                row = [file] + [results[n] for n in n_values]

                writer.writerow(row)

        print(f"Results written to: {filename}")


# ==========================================================
# =========================== MAIN =========================
# ==========================================================

def main():

    print("\n[START] Launching benchmark framework")

    benchmark = Benchmark(

        PROGRAMS,
        N_VALUES,
        RUNS_PER_N

    )

    results = benchmark.run()

    CSVWriter.write(

        OUTPUT_CSV,
        results,
        N_VALUES

    )

    print("\n[FINISH] Benchmark process completed successfully.\n")


if __name__ == "__main__":

    main()