#!/usr/bin/env python3
"""
============================================================
C Benchmark Framework: Program vs Problem Size
============================================================

Benchmark multiple C programs across different matrix sizes.

CSV Output:
    Rows    -> programs
    Columns -> problem sizes (N)

Quality-of-life features:
    • Compiler flag aliases
    • Optional OpenMP support
    • Clear terminal diagnostics
============================================================
"""

import subprocess
import statistics
import csv
import sys
from pathlib import Path
import re

# ==========================================================
# ================== CONFIGURATION =========================
# ==========================================================

C_FILES = [
    #"matrixmult.c",
    "matrixmult_opt.c",
    #"matrixmult_opt_NOALIGN.c",
    #"matrixmult_library.c"
    "omp_matrixmult.c"
]

N_VALUES = [1000, 2000, 3000]

RUNS_PER_N = 1
N_THREADS = 24

COMPILER = "gcc"

# ----------------------------------------------------------
# Flag aliases
# ----------------------------------------------------------

FLAG_ALIASES = {

    "OPT_O3": {
        "gcc": ["-O3"],
        "icc": ["-O3"],
        "icx": ["-O3"]
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

# ----------------------------------------------------------
# Optimization profile
# ----------------------------------------------------------

OPT_CONFIG = [
    "OPT_O3",
    "MATH_LIB",
    "CPU_NATIVE"
]

# ==========================================================
# ================= OUTPUT CONFIGURATION ===================
# ==========================================================

OUTPUT_FOLDER = Path(__file__).parent / "benchmarks"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

file_stems = "_".join(Path(f).stem for f in C_FILES)
n_values_str = "_".join(str(n) for n in N_VALUES)
flags_str = "+".join(OPT_CONFIG)

OUTPUT_CSV = OUTPUT_FOLDER / (
    f"results|{COMPILER}|{file_stems}|{n_values_str}|{flags_str}.csv"
)

# ==========================================================
# ================== UTILITY FUNCTIONS =====================
# ==========================================================

def expand_flags(compiler, config):

    flags = []

    for alias in config:

        flags.extend(
            FLAG_ALIASES.get(alias, {}).get(compiler, [])
        )

    return flags


# ==========================================================
# ===================== COMPILER ===========================
# ==========================================================

class Compiler:

    @staticmethod
    def compile(source_file: Path):

        source_file = source_file.resolve()

        flags = expand_flags(COMPILER, OPT_CONFIG)

        safe_flags = "_".join(f.replace("-", "") for f in flags)

        binary = OUTPUT_FOLDER / f"{source_file.stem}_{safe_flags}"

        print(f"\n[COMPILING]")
        print(f"  Source file : {source_file}")
        print(f"  Compiler    : {COMPILER}")
        print(f"  Flags       : {' '.join(flags)}")

        # special case for library version
        if source_file.name == "matrixmult_library.c":

            build_script = source_file.parent / "build_matrixmult.sh"

            if not build_script.exists():

                print(f"[ERROR] Build script not found: {build_script}")
                sys.exit(1)

            print(f"  Running build script...")

            try:

                subprocess.run(
                    [str(build_script)],
                    check=True
                )

            except subprocess.CalledProcessError:

                print("[ERROR] Build script failed.")
                sys.exit(1)

            binary = source_file.parent / "matrixmult"

            if not binary.exists():

                print("[ERROR] Expected executable not found.")
                sys.exit(1)

            print(f"  ✔ Build successful → {binary}")

            return binary

        normal_flags = [f for f in flags if f != "-lm"]
        lm_flags = ["-lm"] if "-lm" in flags else []

        cmd = [COMPILER, str(source_file), "-o", str(binary)] + normal_flags + lm_flags

        try:

            if COMPILER in ["icc", "icx"]:

                compile_command = " ".join(cmd)

                subprocess.run(
                    [
                        "bash",
                        "-c",
                        f"source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 && {compile_command}"
                    ],
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

            print(f"\n[ERROR] Compilation failed.")

            stdout = e.stdout or ""
            stderr = e.stderr or ""

            print(stdout)
            print(stderr)

            if "omp_" in stderr:

                print("\n[DIAGNOSTIC]")
                print("OpenMP symbols detected but OpenMP not enabled.")
                print("Add 'OPENMP' to OPT_CONFIG.")

            sys.exit(1)

        return binary


# ==========================================================
# ===================== EXECUTOR ===========================
# ==========================================================

class Executor:

    @staticmethod
    def run(binary: Path, n: int):

        print(f"  Running benchmark (n={n})...")

        use_openmp = "OPENMP" in OPT_CONFIG

        if use_openmp:

            cmd = [str(binary), str(n), str(N_THREADS)]

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

            print(f"\n[ERROR] Execution failed.")

            stdout = e.stdout or ""
            stderr = e.stderr or ""

            print(stdout)
            print(stderr)

            if use_openmp:

                print("\n[DIAGNOSTIC]")
                print("Possible argument mismatch with OpenMP.")
                print("Program may not accept NUM_THREADS argument.")

            sys.exit(1)

        for line in result.stdout.splitlines():

            if "Execution Time" in line or "Tempo di esecuzione" in line:

                t = float(line.split()[-2])

                print(f"  ✔ Execution time: {t:.6f} s")

                return t

        print("[ERROR] Execution time not found in program output")

        sys.exit(1)


# ==========================================================
# ===================== BENCHMARK ==========================
# ==========================================================

class Benchmark:

    def __init__(self, files, n_values, runs):

        self.files = files
        self.n_values = n_values
        self.runs = runs

    def run(self):

        print("\n================================================")
        print("Benchmark started")
        print("Compiler:", COMPILER)
        print("Programs:", ", ".join(self.files))
        print("Matrix sizes:", ", ".join(map(str, self.n_values)))
        print("================================================")

        results = {}

        for file in self.files:

            source = Path(file)

            if not source.exists():

                print(f"[ERROR] File not found: {file}")
                sys.exit(1)

            print(f"\n[PROGRAM] {file}")

            binary = Compiler.compile(source)

            program_results = {}

            for n in self.n_values:

                timings = [
                    Executor.run(binary, n)
                    for _ in range(self.runs)
                ]

                avg = statistics.mean(timings)

                program_results[n] = avg

                print(f"  → Average time: {avg:.6f} s")

            results[file] = program_results

        print("\nBenchmark finished successfully.")

        return results


# ==========================================================
# ====================== CSV WRITER ========================
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
# ========================= MAIN ===========================
# ==========================================================

def main():

    print("\n[START] Launching benchmark framework")

    benchmark = Benchmark(
        C_FILES,
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