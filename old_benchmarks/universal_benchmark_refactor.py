#!/usr/bin/env python3

import subprocess
import csv
import sys
import time
import os
import hashlib
from pathlib import Path
from itertools import product


# ==========================================================
# ======================== CONFIG ==========================
# ==========================================================

OUTPUT_FOLDER = Path("benchmarks")

PROGRAMS = [
    ("matrixmult_library.c", "sequential"),
    ("matrixmult_opt.c", "sequential"),
    #("omp_matrixmult.c", "openmp"),
    #("omp_matrixmult_tiling.c", "openmp")
]

PARAMETERS = {
    "program": PROGRAMS,
    "size": [1000,2000,3000,5000,8000,10000,15000],
    "threads": [1],
    "compiler": ["icx"],
    "flagset": [
        ["OPT_O3","CPU_NATIVE"],
        ["OPT_O3","CPU_NATIVE","LINKING"],
        ["FAST"]
    ]
}


# ==========================================================
# ====================== TERMINAL UTILS ====================
# ==========================================================

class Logger:

    @staticmethod
    def info(msg): print(f"[INFO] {msg}")

    @staticmethod
    def success(msg): print(f"[OK]   {msg}")

    @staticmethod
    def warn(msg): print(f"[WARN] {msg}")

    @staticmethod
    def error(msg):
        print(f"[ERROR] {msg}")
        sys.exit(1)


# ==========================================================
# ======================== PROGRAM =========================
# ==========================================================

class Program:

    def __init__(self, file, ptype):
        self.file = file
        self.type = ptype

    @property
    def path(self):
        return Path(self.file).resolve()

    @property
    def name(self):
        return Path(self.file).name


# ==========================================================
# ===================== SYSTEM DEFINITIONS =================
# ==========================================================

class SystemDefs:

    FLAG_ALIASES = {
        "OPT_O3": {"gcc": ["-O3"], "icc": ["-O3"], "icx": ["-O3"]},
        "MATH_LIB": {"gcc": ["-lm"], "icc": ["-lm"], "icx": ["-lm"]},
        "CPU_NATIVE": {"gcc": ["-march=native"],"icc": ["-xHost"],"icx": ["-xHost"]},
        "FAST": {"gcc": ["-Ofast"], "icc": ["-Ofast"], "icx": ["-fast"]},
        "LINKING": {"gcc": ["-flto"], "icc": ["-ipo"], "icx": ["-ipo"]},

        "OPENMP": {"gcc": ["-fopenmp"], "icc": ["-qopenmp"],"icx": ["-fopenmp"]}
    }

    PROGRAM_TYPES = {
        "sequential": {"compile_prefix": []},
        "openmp": {"compile_prefix": ["OPENMP"]}
    }


# ==========================================================
# ====================== FLAG MANAGER ======================
# ==========================================================

class FlagManager:

    @staticmethod
    def build_flags(program, compiler, user_flags):

        aliases = []

        aliases += SystemDefs.PROGRAM_TYPES[program.type]["compile_prefix"]
        aliases += user_flags

        flags = []

        for alias in aliases:

            compiler_flags = SystemDefs.FLAG_ALIASES.get(alias, {}).get(compiler)

            if compiler_flags:
                flags.extend(compiler_flags)

        return flags


# ==========================================================
# ======================== COMPILER ========================
# ==========================================================

class Compiler:

    cache = {}

    @staticmethod
    def compile(program, compiler, flags, output_folder):

        key = (program.file, compiler, tuple(flags))

        if key in Compiler.cache:
            return Compiler.cache[key]

        source = program.path

        flag_hash = Compiler.flags_hash(flags)
        binary = output_folder / f"{source.stem}_{compiler}_{flag_hash}"

        if binary.exists():
            Compiler.cache[key] = binary
            return binary

        Logger.info(f"Compiling {source.name} with {compiler}")

        if source.name == "matrixmult_library.c":

            build_script = source.parent / "build_matrixmult.sh"

            if not build_script.exists():
                Logger.error(f"Build script not found: {build_script}")

            if not os.access(build_script, os.X_OK):
                os.chmod(build_script, 0o755)

            subprocess.run([str(build_script)], check=True)

            binary = source.parent / "matrixmult"

            Compiler.cache[key] = binary
            return binary

        cmd = [compiler] + flags + [str(source), "-o", str(binary)]

        if compiler in ["icc","icx"]:

            subprocess.run(
                ["bash","-c",
                f"source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 && {' '.join(cmd)}"],
                check=True
            )

        else:
            subprocess.run(cmd, check=True)

        Compiler.cache[key] = binary

        return binary

    @staticmethod
    def flags_hash(flags):
        s = " ".join(flags)
        return hashlib.md5(s.encode()).hexdigest()[:8]


# ==========================================================
# ========================= EXECUTOR =======================
# ==========================================================

class Executor:

    @staticmethod
    def run(binary, size=None, threads=None):

        cmd = [str(binary)]

        if size:
            cmd.append(str(size))

        if threads:
            cmd.append(str(threads))

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            Logger.error(f"Program failed:\n{result.stderr}")

        for line in result.stdout.splitlines():

            if "Execution Time" in line or "Tempo di esecuzione" in line:
                return float(line.split()[-2])

        Logger.error("Execution time not found")


# ==========================================================
# ================= PARAMETER SWEEP ENGINE =================
# ==========================================================

class ParameterSweep:

    def __init__(self, parameters):

        self.keys = list(parameters.keys())
        self.values = list(parameters.values())

    def __iter__(self):

        for combo in product(*self.values):
            yield dict(zip(self.keys, combo))




def print_experiment_recap(parameters):
    Logger.info("Experiment Recap:")
    for k, v in parameters.items():
        if k == "program":
            v = [p[0] if isinstance(p, tuple) else p.name for p in v]
        Logger.info(f"  {k}: {v}")
    Logger.info(f"Total combinations: {len(list(ParameterSweep(parameters)))}")

def print_progress_bar(iteration, total, prefix='', length=40):
    """
    Print a progress bar in the terminal.

    iteration : int : current iteration
    total     : int : total iterations
    prefix    : str : optional prefix string
    length    : int : character length of the bar
    """
    percent = iteration / total
    filled_length = int(length * percent)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {iteration}/{total}')
    sys.stdout.flush()
    if iteration == total:
        print()

def get_csv_path(parameters):
    headers = list(parameters.keys()) + ["runtime"]
    name = "results|" + "|".join(f"{k}={len(v)}" for k, v in PARAMETERS.items())
    csv_file = OUTPUT_FOLDER / f"{name}.csv"
    return csv_file, headers


def load_completed(csv_file):
    completed = set()

    if not csv_file.exists():
        return completed

    with open(csv_file) as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            completed.add(tuple(row[:-1]))

    return completed


def open_csv_writer(csv_file, headers):
    write_header = not csv_file.exists()

    f = open(csv_file, "a", newline="")
    writer = csv.writer(f)

    if write_header:
        writer.writerow(headers)

    return f, writer


def build_key(p, parameters):
    return tuple(
        tuple(p.get(k)) if isinstance(p.get(k), list) else (p.get(k).file if isinstance(p.get(k), Program) else p.get(k))
        for k in parameters.keys()
    )


def run_experiment(p, parameters):
    program = p.get("program")
    compiler = p.get("compiler", "gcc")
    flagset = p.get("flagset", [])
    size = p.get("size")
    threads = p.get("threads")

    flags = FlagManager.build_flags(program, compiler, flagset)
    binary = Compiler.compile(program, compiler, flags, OUTPUT_FOLDER)

    runtime = Executor.run(binary, size, threads)

    key = build_key(p, parameters)

    return key, runtime

# ==========================================================
# ============================ MAIN ========================
# ==========================================================

def main():
    # Ensure output folder exists
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    # Prepare parameters: convert programs to Program objects
    parameters = PARAMETERS.copy()
    if "program" in parameters:
        parameters["program"] = [Program(p[0], p[1]) for p in parameters["program"]]

    # Print experiment recap
    print_experiment_recap(parameters)

    # Generate all combinations
    sweep = list(ParameterSweep(parameters))
    total_runs = len(sweep)

    # CSV file path and headers
    csv_file, headers = get_csv_path(parameters)

    # Load already completed runs to skip them
    completed = load_completed(csv_file)

    # Run experiments
    for i, p in enumerate(sweep, 1):
        key = build_key(p, parameters)

        # Skip if already completed
        if key in completed:
            print_progress_bar(i, total_runs, prefix='Running experiments')
            continue

        # Run experiment
        key, runtime = run_experiment(p, parameters)

        # Write result immediately to CSV
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            # Write header if file is empty
            if f.tell() == 0:
                writer.writerow(headers)
            writer.writerow(list(key) + [runtime])

        # Update progress bar
        print_progress_bar(i, total_runs, prefix='Running experiments')

    Logger.success(f"Benchmark complete → {csv_file}")

if __name__ == "__main__":
    main()