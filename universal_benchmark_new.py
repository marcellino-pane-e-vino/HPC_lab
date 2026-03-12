#!/usr/bin/env python3

import subprocess
import csv
import sys
import time
import os
import hashlib
from pathlib import Path
from itertools import product
import datetime

# ==========================================================
# ======================== CONFIG ==========================
# ==========================================================

OUTPUT_FOLDER = Path("benchmarks")

# Concept: Allow 'None' in threads so sequential programs only get 1 argument (size)
PARAMETERS = {
    "program": [
        ("matrixmult_library.c", "sequential"),
        ("matrixmult_opt.c", "sequential"),
        # ("omp_matrixmult.c", "openmp"),
    ],
    "size": [1000, 2000, 3000],
    "threads": [None],  # Use None for sequential, use integers [1, 2, 4] for OpenMP
    "compiler": ["icx"],
    "flagset": [
        #["FAST"], 
        ["OPT_O3", "CPU_NATIVE"]
    ]
}

# ==========================================================
# ====================== UTILITIES =========================
# ==========================================================

class Logger:
    @staticmethod
    def _ts(): return datetime.datetime.now().strftime("%H:%M:%S")
    @staticmethod
    def info(msg): print(f"[{Logger._ts()}] [INFO]  {msg}")
    @staticmethod
    def success(msg): print(f"[{Logger._ts()}] [OK]    {msg}")
    @staticmethod
    def error(msg):
        print(f"[{Logger._ts()}] [ERROR] {msg}")
        sys.exit(1)

class StepTimer:
    """Context manager for accurate duration tracking."""
    def __init__(self, description):
        self.desc = description

    def __enter__(self):
        Logger.info(f"STARTING : {self.desc}")
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, *args):
        elapsed = time.perf_counter() - self.start
        if exc_type is None:
            Logger.success(f"FINISHED : {self.desc} ({elapsed:.2f}s)")
        else:
            Logger.error(f"FAILED   : {self.desc} ({elapsed:.2f}s)")

def load_intel_environment():
    """
    Concept: Run setvars.sh once and absorb the environment into Python.
    This entirely removes the need for messy bash wrappers during compilation.
    """
    if "CMPLR_ROOT" in os.environ:
        return # Already loaded

    Logger.info("Loading Intel oneAPI environment...")
    cmd = ['bash', '-c', 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 && env']
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        for line in proc.stdout.splitlines():
            if '=' in line:
                key, val = line.split('=', 1)
                os.environ[key] = val
        Logger.success("Intel environment loaded natively.")
    except Exception as e:
        Logger.error("Failed to load Intel environment.")

# ==========================================================
# ===================== CORE CLASSES =======================
# ==========================================================

class SystemDefs:
    ALIASES = {
        "OPT_O3": {"gcc": ["-O3"], "icc": ["-O3"], "icx": ["-O3"]},
        "MATH_LIB": {"gcc": ["-lm"], "icc": ["-lm"], "icx": ["-lm"]},
        "CPU_NATIVE": {"gcc": ["-march=native"], "icc": ["-xHost"], "icx": ["-xHost"]},
        "FAST": {"gcc": ["-Ofast"], "icc": ["-Ofast"], "icx": ["-fast"]},
        "LINKING": {"gcc": ["-flto"], "icc": ["-ipo"], "icx": ["-ipo"]},
        "OPENMP": {"gcc": ["-fopenmp"], "icc": ["-qopenmp"], "icx": ["-qopenmp"]}
    }
    TYPES = {
        "sequential": [],
        "openmp": ["OPENMP"]
    }

class Compiler:
    def __init__(self, output_dir):
        self.out_dir = Path(output_dir)
        self.out_dir.mkdir(exist_ok=True)
        self.cache = {}

    def _get_flags(self, ptype, compiler_name, user_flags):
        flag_list = SystemDefs.TYPES[ptype] + user_flags
        resolved = []
        for alias in flag_list:
            if alias in SystemDefs.ALIASES and compiler_name in SystemDefs.ALIASES[alias]:
                resolved.extend(SystemDefs.ALIASES[alias][compiler_name])
        return resolved

    def compile(self, file_path, ptype, compiler_name, user_flags):
        path = Path(file_path).resolve()
        flags = self._get_flags(ptype, compiler_name, user_flags)
        
        # Concept: Hash the exact flags to guarantee unique binaries
        flag_hash = hashlib.md5(" ".join(flags).encode()).hexdigest()[:8]
        binary = self.out_dir / f"{path.stem}_{compiler_name}_{flag_hash}"

        cache_key = (str(path), compiler_name, tuple(flags))
        
        # FIX 1: Check memory cache FIRST. This preserves special paths.
        if cache_key in self.cache:
            cached_bin = self.cache[cache_key]
            Logger.info(f"CACHE HIT (Memory): {cached_bin.name}")
            return cached_bin
            
        # FIX 2: Check disk separately for standard binaries
        if binary.exists():
            Logger.info(f"CACHE HIT (Disk): {binary.name}")
            self.cache[cache_key] = binary
            return binary

        with StepTimer(f"Compiling {path.name} with {compiler_name} [{flag_hash}]"):
            # Special case for the library build script
            if path.name == "matrixmult_library.c":
                build_script = path.parent / "build_matrixmult.sh"
                if not os.access(build_script, os.X_OK): os.chmod(build_script, 0o755)
                subprocess.run([str(build_script)], check=True)
                special_bin = path.parent / "matrixmult"
                
                # Save the special path to memory cache
                self.cache[cache_key] = special_bin 
                return special_bin

            # Standard compilation (Intel env is already loaded globally)
            cmd = [compiler_name] + flags + [str(path), "-o", str(binary)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            
            if res.returncode != 0:
                Logger.error(f"Compile failed:\n{res.stderr}")

        self.cache[cache_key] = binary
        return binary

        with StepTimer(f"Compiling {path.name} with {compiler_name} [{flag_hash}]"):
            # Special case for the library build script
            if path.name == "matrixmult_library.c":
                build_script = path.parent / "build_matrixmult.sh"
                if not os.access(build_script, os.X_OK): os.chmod(build_script, 0o755)
                subprocess.run([str(build_script)], check=True)
                special_bin = path.parent / "matrixmult"
                self.cache[cache_key] = special_bin
                return special_bin

            # Standard compilation (Intel env is already loaded globally)
            cmd = [compiler_name] + flags + [str(path), "-o", str(binary)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            
            if res.returncode != 0:
                Logger.error(f"Compile failed:\n{res.stderr}")

        self.cache[cache_key] = binary
        return binary

class Executor:
    @staticmethod
    def run(binary, size, threads):
        cmd = [str(binary), str(size)]
        if threads is not None:
            cmd.append(str(threads))

        with StepTimer(f"Execution: {binary.name} (Size: {size}, Threads: {threads})"):
            res = subprocess.run(cmd, capture_output=True, text=True)
            
            if res.returncode != 0:
                Logger.error(f"Execution crashed:\n{res.stderr}")

            for line in res.stdout.splitlines():
                if "Execution Time" in line or "Tempo di esecuzione" in line:
                    runtime = float(line.split()[-2])
                    Logger.info(f"Internal Runtime: {runtime}s")
                    return runtime
            
            Logger.error(f"No execution time found in output:\n{res.stdout}")

# ==========================================================
# ======================= PIPELINE =========================
# ==========================================================

class BenchmarkPipeline:
    def __init__(self, parameters, output_folder):
        self.params = parameters
        self.compiler = Compiler(output_folder)
        self.csv_path = output_folder / "benchmark_results.csv"
        self.completed_runs = self._load_cache()

    def _load_cache(self):
        completed = set()
        if self.csv_path.exists():
            with open(self.csv_path, 'r') as f:
                next(csv.reader(f), None) # Skip header
                for row in csv.reader(f):
                    if row: completed.add(tuple(row[:-1]))
        return completed

    def _build_csv_key(self, combo):
        # Concept: Ensure every element is a string to match CSV readouts perfectly
        return tuple(str(combo[k]) for k in self.params.keys())

    def run(self):
        keys, values = list(self.params.keys()), list(self.params.values())
        combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        Logger.info(f"Starting pipeline: {len(combinations)} total configurations.")

        # Open file once, append mode
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(keys + ["runtime"])
                f.flush()

            for i, combo in enumerate(combinations, 1):
                csv_key = self._build_csv_key(combo)
                
                if csv_key in self.completed_runs:
                    Logger.info(f"--- Skipping {i}/{len(combinations)}: Already in CSV ---")
                    continue

                Logger.info(f"--- Running {i}/{len(combinations)} ---")
                
                prog_file, prog_type = combo["program"]
                binary = self.compiler.compile(
                    prog_file, prog_type, combo["compiler"], combo["flagset"]
                )
                
                runtime = Executor.run(binary, combo["size"], combo["threads"])
                
                writer.writerow(list(csv_key) + [runtime])
                f.flush() # Force save to disk immediately

        Logger.success(f"Pipeline complete. Results saved to {self.csv_path}")


if __name__ == "__main__":
    load_intel_environment()
    pipeline = BenchmarkPipeline(PARAMETERS, OUTPUT_FOLDER)
    pipeline.run()