#!/usr/bin/env python3
"""
============================================================
Universal C Benchmark Framework
============================================================

Supports experiments:

1) threads × matrix size
2) program × matrix size
3) compiler × optimization flags

Features
--------
• Program type abstraction
• Flag alias system
• Compile caching
• CSV output
• LaTeX table output
• Descriptive output filenames
• Robust validation and clear terminal messages
============================================================
"""

import subprocess
import statistics
import csv
import sys
from pathlib import Path
import os

# ==========================================================
# ====================== TERMINAL UTILS ===================
# ==========================================================

def info(msg):
    print(f"[INFO] {msg}")

def success(msg):
    print(f"[OK]   {msg}")

def warn(msg):
    print(f"[WARN] {msg}")

def error(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)

# ==========================================================
# ====================== EXPERIMENT MODE ===================
# ==========================================================

MODE = "program_vs_size"
# OPTIONS: "threads_vs_size", "program_vs_size", "compiler_vs_flags"

# ==========================================================
# ====================== PROGRAM CONFIG ====================
# ==========================================================

PROGRAMS = [
    {"file": "matrixmult_library.c", "type": "sequential"},
    {"file": "omp_matrixmult.c", "type": "openmp"}
]

SINGLE_PROGRAM = {"file": "matrixmult_opt.c", "type": "sequential"}

# ==========================================================
# ====================== SINGLE CONFIG =====================
# ==========================================================
# This will be used in "threads_vs_size" and "program_vs_size" modes.
SINGLE_CONFIG = ["OPT_O3", "MATH_LIB", "CPU_NATIVE", "FAST"]

# ==========================================================
# ====================== PARAMETERS ========================
# ==========================================================

MATRIX_SIZES = [1000, 2000, 3000]
THREAD_VALUES = [8, 16, 24]
COMPILERS = ["gcc", "icc", "icx"]
RUNS_PER_TEST = 1
N_THREADS = 24
N_VALUE = 300

# ==========================================================
# ====================== FLAG SYSTEM =======================
# ==========================================================

OPT_CONFIGS = [
    ["OPT_O3", "MATH_LIB", "CPU_NATIVE", "FAST", "MEMORY_ALIGNMENT", "INLINE"],
    ["OPT_O3", "MATH_LIB", "CPU_NATIVE", "FAST", "MEMORY_ALIGNMENT", "INLINE", "LINKING"]
]

FLAG_ALIASES = {
    "OPT_O3": {"gcc": ["-O3"], "icc": ["-O3"], "icx": ["-O3"]},
    "MATH_LIB": {"gcc": ["-lm"], "icc": ["-lm"], "icx": ["-lm"]},
    "CPU_NATIVE": {"gcc": ["-march=native"], "icc": ["-xHost"], "icx": ["-xHost"]},
    "FAST": {"gcc": ["-Ofast"], "icc": ["-Ofast"], "icx": ["-fast"]},
    "MEMORY_ALIGNMENT": {"gcc": ["-DALIGNED"], "icc": ["-DALIGNED"], "icx": ["-DALIGNED"]},
    "INLINE": {"gcc": ["-DNOFUNCCALL"], "icc": ["-DNOFUNCCALL"], "icx": ["-DNOFUNCCALL"]},
    "LINKING": {"gcc": ["-flto"], "icc": ["-ipo"], "icx": ["-ipo"]},
    "OPENMP": {"gcc": ["-fopenmp"], "icc": ["-qopenmp"], "icx": ["-fopenmp"]}
}

# ==========================================================
# ================= PROGRAM TYPE SYSTEM ====================
# ==========================================================

DEFAULT_COMPILER = "icx"

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
    }
}

# ==========================================================
# ======================= OUTPUT SETUP =====================
# ==========================================================

OUTPUT_FOLDER = Path(__file__).parent / "benchmarks"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ==========================================================
# ===================== VALIDATION =========================
# ==========================================================

class Validator:

    @staticmethod
    def validate():
        info("Validating configuration...")

        valid_modes = ["threads_vs_size", "program_vs_size", "compiler_vs_flags"]
        if MODE not in valid_modes:
            error(f"Invalid MODE '{MODE}'. Choose from: {valid_modes}")

        for p in PROGRAMS + [SINGLE_PROGRAM]:
            path = Path(p["file"])
            if not path.exists():
                error(f"Source file not found: {path}")

        for c in COMPILERS:
            if subprocess.call(["which", c], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
                warn(f"Compiler '{c}' not found in PATH")

        success("Configuration validated")

# ==========================================================
# ===================== FLAG UTILITIES =====================
# ==========================================================

def expand_flags(compiler, aliases):
    flags = []
    for alias in aliases:
        compiler_flags = FLAG_ALIASES.get(alias, {}).get(compiler)
        if compiler_flags is None:
            warn(f"No flags found for alias '{alias}' and compiler '{compiler}'")
        else:
            flags.extend(compiler_flags)
    return flags

def build_flags(program, compiler, config=None):
    ptype = program["type"]
    type_info = PROGRAM_TYPES[ptype]

    aliases = []
    aliases += type_info.get("compile_prefix", [])
    if config:
        aliases += config
    aliases += program.get("flags", [])
    aliases += type_info.get("compile_suffix", [])

    return expand_flags(compiler, aliases)

# ==========================================================
# ========================= COMPILER =======================
# ==========================================================

class Compiler:

    cache = {}

    @staticmethod
    def compile(program, compiler, flags):
        key = (program["file"], compiler, tuple(flags))
        if key in Compiler.cache:
            return Compiler.cache[key]

        source = Path(program["file"]).resolve()
        safe_flags = "_".join(f.replace("-", "") for f in flags)
        binary = OUTPUT_FOLDER / f"{source.stem}_{compiler}_{safe_flags}"

        info(f"Compiling {source.name} with {compiler}")
        info(f"Flags: {' '.join(flags)}")

        # Check if the program is matrixmult_library.c and handle OpenBLAS
        if source.name == "matrixmult_library.c":
            build_script = source.parent / "build_matrixmult.sh"
            if not build_script.exists():
                error(f"Build script not found: {build_script}")

            # Ensure the build script is executable
            if not os.access(build_script, os.X_OK):
                info(f"Build script not executable. Fixing permissions...")
                try:
                    os.chmod(build_script, 0o755)
                except Exception as e:
                    error(f"Failed to set execute permission: {e}")

            info(f"Running build script for {source} ...")
            try:
                subprocess.run([str(build_script)], check=True)
            except subprocess.CalledProcessError:
                error(f"Build script failed for {source}")
            
            binary = source.parent / "matrixmult"
            if not binary.exists():
                error(f"Expected executable not found: {binary}")

            success(f"Build script completed: {binary}")
            Compiler.cache[key] = binary
            return binary

        # For other files, continue with normal compilation
        normal_flags = [f for f in flags if f != "-lm"]
        lm_flags = ["-lm"] if "-lm" in flags else []

        cmd = [compiler, str(source), "-o", str(binary)] + normal_flags + lm_flags

        try:
            if compiler in ["icc", "icx"]:
                compile_command = " ".join(cmd)
                subprocess.run(
                    ["bash", "-c",
                     f"source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 && {compile_command}"],
                    check=True
                )
            else:
                subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            error(f"Compilation failed for {source.name}. Command: {' '.join(cmd)}")

        Compiler.cache[key] = binary
        success(f"Compilation finished: {binary}")
        return binary

# ==========================================================
# ========================= EXECUTOR =======================
# ==========================================================

class Executor:

    @staticmethod
    def run(binary, n, threads=None):
        cmd = [str(binary), str(n)]
        if threads:
            cmd.append(str(threads))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            error(f"Execution failed: {binary}\n{e.stderr}")

        for line in result.stdout.splitlines():
            if "Execution Time" in line or "Tempo di esecuzione" in line:
                try:
                    return float(line.split()[-2])
                except ValueError:
                    error(f"Could not parse execution time from output: {line}")

        error("Execution time not found in program output")

# ==========================================================
# ======================= CSV WRITER =======================
# ==========================================================

class CSVWriter:

    @staticmethod
    def write(filename, headers, rows):
        info(f"Writing CSV results → {filename}")
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for r in rows:
                writer.writerow(r)
        success("CSV export complete")

# ==========================================================
# ====================== LATEX WRITER ======================
# ==========================================================

class LaTeXWriter:

    @staticmethod
    def write(filename, headers, rows):
        info(f"Writing LaTeX table → {filename}")
        with open(filename, "w") as f:
            cols = "c|" + "c" * (len(headers)-1)
            f.write(f"\\begin{{tabular}}{{{cols}}}\n")
            # Convert all items in headers to strings before joining
            f.write(" & ".join(str(header) for header in headers) + " \\\\\n")
            f.write("\\hline\n")
            for r in rows:
                formatted = [f"{x:.3f}" if isinstance(x, float) else str(x) for x in r]
                f.write(" & ".join(formatted) + " \\\\\n")
            f.write("\\end{tabular}\n")
        success("LaTeX export complete")

# ==========================================================
# ====================== BENCHMARKS ========================
# ==========================================================

class Benchmarks:

    @staticmethod
    def threads_vs_size():
        program = SINGLE_PROGRAM
        compiler = PROGRAM_TYPES[program["type"]]["compiler"]
        flags = build_flags(program, compiler, SINGLE_CONFIG)
        binary = Compiler.compile(program, compiler, flags)

        headers = ["threads"] + MATRIX_SIZES
        rows = []

        for t in THREAD_VALUES:
            row = [t]
            for n in MATRIX_SIZES:
                times = [Executor.run(binary, n, t) for _ in range(RUNS_PER_TEST)]
                row.append(statistics.mean(times))
            rows.append(row)
        return headers, rows

    @staticmethod
    def program_vs_size():
        headers = ["program"] + MATRIX_SIZES
        rows = []

        for program in PROGRAMS:
            compiler = PROGRAM_TYPES[program["type"]]["compiler"]
            flags = build_flags(program, compiler, SINGLE_CONFIG)
            binary = Compiler.compile(program, compiler, flags)
            row = [program["file"]]
            for n in MATRIX_SIZES:
                times = [Executor.run(binary, n, N_THREADS) for _ in range(RUNS_PER_TEST)]
                row.append(statistics.mean(times))
            rows.append(row)
        return headers, rows

    @staticmethod
    def compiler_vs_flags():
        program = SINGLE_PROGRAM
        headers = ["config"] + COMPILERS
        rows = []

        for config in OPT_CONFIGS:
            row = ["_".join(config)]
            for compiler in COMPILERS:
                flags = expand_flags(compiler, config)
                binary = Compiler.compile(program, compiler, flags)
                times = [Executor.run(binary, N_VALUE, N_THREADS) for _ in range(RUNS_PER_TEST)]
                row.append(statistics.mean(times))
            rows.append(row)
        return headers, rows

# ==========================================================
# =========================== MAIN =========================
# ==========================================================

def print_experiment_summary():
    print("\n================================================")
    print("Benchmark Configuration")
    print("------------------------------------------------")
    print(f"Mode           : {MODE}")
    print(f"Matrix sizes   : {MATRIX_SIZES}")
    print(f"Threads        : {THREAD_VALUES}")
    print(f"Compilers      : {COMPILERS}")
    print(f"Runs per test  : {RUNS_PER_TEST}")
    print("================================================\n")

def main():
    info("Starting Universal Benchmark Framework")
    Validator.validate()
    print_experiment_summary()

    if MODE == "threads_vs_size":
        headers, rows = Benchmarks.threads_vs_size()
    elif MODE == "program_vs_size":
        headers, rows = Benchmarks.program_vs_size()
    elif MODE == "compiler_vs_flags":
        headers, rows = Benchmarks.compiler_vs_flags()
    else:
        error("Unsupported benchmark mode")

    csv_file = OUTPUT_FOLDER / f"{MODE}.csv"
    tex_file = OUTPUT_FOLDER / f"{MODE}.tex"

    CSVWriter.write(csv_file, headers, rows)
    LaTeXWriter.write(tex_file, headers, rows)

    success("Benchmark completed successfully")

if __name__ == "__main__":
    main()