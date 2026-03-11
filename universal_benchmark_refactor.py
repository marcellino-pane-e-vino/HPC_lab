#!/usr/bin/env python3

import subprocess
import csv
import sys
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
    ("omp_matrixmult.c", "openmp")
]

PARAMETERS = {
    "program": PROGRAMS,
    "size": [1000,2000,3000],
    "threads": [1,2,4,8,12,16,24],
    "compiler": ["gcc","icx"],
    "flagset": [
        ["OPT_O3","MATH_LIB","CPU_NATIVE","FAST"]
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

        binary = output_folder / f"{source.stem}_{compiler}"

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

        cmd = [compiler, str(source), "-o", str(binary)] + flags

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

        result = subprocess.run(cmd,capture_output=True,text=True)

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


# ==========================================================
# ============================ MAIN ========================
# ==========================================================

def main():

    OUTPUT_FOLDER.mkdir(exist_ok=True)

    parameters = PARAMETERS.copy()

    if "program" in parameters:
        parameters["program"] = [Program(p[0],p[1]) for p in parameters["program"]]

    sweep = ParameterSweep(parameters)

    rows = []

    for p in sweep:
        program = p.get("program")
        compiler = p.get("compiler","gcc")
        flagset = p.get("flagset",[])
        size = p.get("size")
        threads = p.get("threads")

        flags = FlagManager.build_flags(program,compiler,flagset)
        binary = Compiler.compile(program,compiler,flags,OUTPUT_FOLDER)
        runtime = Executor.run(binary,size,threads)
        row = [
            (p.get(k).file if isinstance(p.get(k), Program) else p.get(k))
            for k in parameters.keys()
        ] + [runtime]
        rows.append(row)

    headers = list(parameters.keys()) + ["runtime"]
    name = "results|" + "|".join(f"{k}={len(v)}" for k,v in PARAMETERS.items())
    csv_file = OUTPUT_FOLDER / f"{name}.csv"

    with open(csv_file,"w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    Logger.success(f"Benchmark complete → {csv_file}")


if __name__ == "__main__":
    main()