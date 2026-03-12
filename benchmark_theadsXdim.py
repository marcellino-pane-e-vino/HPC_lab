#!/usr/bin/env python3

import subprocess
import statistics
import csv
import sys
from pathlib import Path

# ==========================================================
# ====================== CONFIGURATION =====================
# ==========================================================

PROGRAM = {
    "file": "omp_matrixmult.c",
    "type": "openmp"
}

MATRIX_SIZES = [1000, 2000, 3000]

THREAD_VALUES = [8, 16, 24]

RUNS_PER_TEST = 1

DEFAULT_COMPILER = "gcc"

GLOBAL_FLAGS = [
    "OPT_O3",
    "MATH_LIB",
    "CPU_NATIVE"
]

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
    }
}

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
# OUTPUT
# ==========================================================

OUTPUT_FOLDER = Path(__file__).parent / "benchmarks"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

sizes_str = "_".join(str(n) for n in MATRIX_SIZES)

OUTPUT_CSV = OUTPUT_FOLDER / f"thread_scaling_{sizes_str}.csv"

# ==========================================================
# FLAG UTILITIES
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
# COMPILER
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

        binary = OUTPUT_FOLDER / f"{source_file.stem}_scaling"

        print("\n[COMPILING]")
        print(f"Program  : {source_file.name}")
        print(f"Compiler : {compiler}")
        print(f"Flags    : {' '.join(flags)}")

        normal_flags = [f for f in flags if f != "-lm"]
        lm_flags = ["-lm"] if "-lm" in flags else []

        cmd = [compiler, str(source_file), "-o", str(binary)] + normal_flags + lm_flags

        subprocess.run(cmd, check=True)

        return binary

# ==========================================================
# EXECUTOR
# ==========================================================

class Executor:

    @staticmethod
    def run(program, binary, n, threads):

        ptype = program["type"]
        run_mode = PROGRAM_TYPES[ptype]["run_mode"]

        if run_mode == "openmp":
            cmd = [str(binary), str(n), str(threads)]
        else:
            cmd = [str(binary), str(n)]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        for line in result.stdout.splitlines():

            if "Execution Time" in line or "Tempo di esecuzione" in line:

                return float(line.split()[-2])

        print("[ERROR] Execution time not found")
        sys.exit(1)

# ==========================================================
# BENCHMARK
# ==========================================================

class Benchmark:

    def run(self):

        binary = Compiler.compile(PROGRAM)

        results = {}

        for threads in THREAD_VALUES:

            print(f"\n[THREADS] {threads}")

            row = {}

            for n in MATRIX_SIZES:

                timings = [

                    Executor.run(PROGRAM, binary, n, threads)

                    for _ in range(RUNS_PER_TEST)
                ]

                avg = statistics.mean(timings)

                row[n] = avg

                print(f"  n={n} → {avg:.6f}s")

            results[threads] = row

        return results

# ==========================================================
# CSV
# ==========================================================

class CSVWriter:

    @staticmethod
    def write(filename, data):

        with open(filename, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow(["threads"] + MATRIX_SIZES)

            for t in THREAD_VALUES:

                writer.writerow(
                    [t] + [data[t][n] for n in MATRIX_SIZES]
                )

# ==========================================================
# MAIN
# ==========================================================

def main():

    benchmark = Benchmark()

    results = benchmark.run()

    CSVWriter.write(OUTPUT_CSV, results)

    print(f"\nResults written to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()