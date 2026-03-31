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
OUTPUT_FOLDER = Path("test_di_oggi")

REPETITIONS = 3  

PARAMETERS = {
    "program": [
    # ---- sequential zone ----
        #("matrixmult_library.c", "sequential"),
        #("matrixmult_opt.c", "sequential"),
    
    # ---- OpenMP zone ----
        #("omp_matrixmult_naive.c", "openmp"),
        #("omp_matrixmult_library.c", "openmp"),
        ("omp_matrixmult_opt.c", "openmp"),
    ],
    "size": [8000,10000,15000,20000],
    "threads": [24],  # Use None for sequential, use integers [1, 2, 4] for OpenMP
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
    def sub_info(msg): print(f"[{Logger._ts()}] [RUN]   |-- {msg}") # New visually distinct log
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

def load_intel_environment(): # loads setvars when needed
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

    def _compile_special_file(self, path, compiler_name, flags, binary, cache_key):
        special_scripts = {
            "matrixmult_library.c": "build_matrixmult.sh",
            "omp_matrixmult_library.c": "omp_build_matrixmult.sh",
        }

        build_script = path.parent / special_scripts[path.name]

        if not os.access(build_script, os.X_OK):
            Logger.sub_info(f"Build script is not executable. Applying chmod +x to: {build_script}")
            os.chmod(build_script, 0o755)

        env = os.environ.copy()
        env["SRC_FILE"] = path.name
        env["COMPILER"] = compiler_name
        env["EXECUTABLE"] = str(binary)
        env["EXTRA_FLAGS"] = " ".join(flags)

        shell_cmd = (
            f'SRC_FILE="{env["SRC_FILE"]}" '
            f'COMPILER="{env["COMPILER"]}" '
            f'EXECUTABLE="{env["EXECUTABLE"]}" '
            f'EXTRA_FLAGS="{env["EXTRA_FLAGS"]}" '
            f'"{build_script}"'
        )

        Logger.info(f"About to compile special file: {path.name}")
        Logger.sub_info(f"Special build script : {build_script}")
        Logger.sub_info(f"Special compiler     : {compiler_name}")
        Logger.sub_info(f"Special flags        : {' '.join(flags) if flags else '(none)'}")
        Logger.sub_info(f"Special output bin   : {binary}")
        Logger.sub_info(f"Shell command sent   : {shell_cmd}")
        Logger.info("Compiling special file...")

        res = subprocess.run(
            [str(build_script)],
            check=False,
            env=env,
            capture_output=True,
            text=True
        )

        if res.returncode == 0:
            Logger.success(
                f"Special compilation successful: {path.name} "
                f"[compiler={compiler_name}, flags={' '.join(flags) if flags else '(none)'}]"
            )
        else:
            Logger.error(
                f"Error in special compilation for {path.name} "
                f"(exit code {res.returncode})\n"
                f"Compiler: {compiler_name}\n"
                f"Flags: {' '.join(flags) if flags else '(none)'}\n"
                f"STDERR:\n{res.stderr}\n"
                f"STDOUT:\n{res.stdout}"
            )

        self.cache[cache_key] = binary
        return binary

    def compile(self, file_path, ptype, compiler_name, user_flags):
        path = Path(file_path).resolve()
        flags = self._get_flags(ptype, compiler_name, user_flags)

        # Hash exact flags to guarantee unique binaries
        flag_hash = hashlib.md5(" ".join(flags).encode()).hexdigest()[:8]
        binary = self.out_dir / f"{path.stem}_{compiler_name}_{flag_hash}"

        cache_key = (str(path), compiler_name, tuple(flags))
        special_files = {"matrixmult_library.c", "omp_matrixmult_library.c"}
        is_special = path.name in special_files

        Logger.info(f"Preparing compilation for: {path.name}")
        Logger.sub_info(f"Resolved source path : {path}")
        Logger.sub_info(f"Compiler            : {compiler_name}")
        Logger.sub_info(f"Resolved flags      : {' '.join(flags) if flags else '(none)'}")
        Logger.sub_info(f"Output binary       : {binary}")
        Logger.sub_info(f"Special file        : {'yes' if is_special else 'no'}")

        # Memory cache first, but special files must always be recompiled
        if not is_special and cache_key in self.cache:
            cached_bin = self.cache[cache_key]
            Logger.info(f"CACHE HIT (Memory): {cached_bin.name}")
            return cached_bin

        # Disk cache only for non-special binaries
        if not is_special and binary.exists():
            Logger.info(f"CACHE HIT (Disk): {binary.name}")
            self.cache[cache_key] = binary
            return binary

        with StepTimer(f"Compiling {path.name} with {compiler_name} [{flag_hash}]"):

            # Special files are always recompiled
            if is_special:
                if binary.exists():
                    Logger.sub_info(f"Special binary already exists but will be recompiled: {binary}")
                return self._compile_special_file(path, compiler_name, flags, binary, cache_key)

            # Standard compilation
            cmd = [compiler_name] + flags + [str(path), "-o", str(binary)]
            Logger.sub_info(f"Standard compile command: {' '.join(cmd)}")

            res = subprocess.run(cmd, capture_output=True, text=True)

            if res.returncode != 0:
                Logger.error(f"Compile failed:\n{res.stderr}")
            else:
                Logger.success(f"Standard compilation completed successfully: {binary.name}")

        self.cache[cache_key] = binary
        return binary

class Executor:
    @staticmethod
    def run(binary, size, threads, repetitions=1):
        runtimes = []
        cmd = [str(binary), str(size)]
        if threads is not None:
            cmd.append(str(threads))

        Logger.info(f"Starting {repetitions} repetitions for: {binary.name} (Size: {size}, Threads: {threads})")

        for r in range(1, repetitions + 1):
            with StepTimer(f"Run {r}/{repetitions}"):
                res = subprocess.run(cmd, capture_output=True, text=True)
                
                if res.returncode != 0:
                    Logger.error(f"Execution crashed on Run {r}:\n{res.stderr}")

            found = False
            time_markers = (
                "Execution Time",
                "Tempo di esecuzione",
                "Tempo totale",
            )

            for line in res.stdout.splitlines():
                if any(marker in line for marker in time_markers):
                    try:
                        runtime = float(line.split()[-2])
                        Logger.sub_info(f"Recorded Time: {runtime}s")
                        runtimes.append(runtime)
                        found = True
                        break
                    except (ValueError, IndexError):
                        Logger.error(
                            f"Found a timing line but could not parse it on Run {r}:\n{line}\n\nFull output:\n{res.stdout}"
            )

            if not found:
                Logger.error(f"No execution time found in output on Run {r}:\n{res.stdout}")
        
        # Concept: Aggregate the list of runtimes into a single mean value
        mean_runtime = sum(runtimes) / len(runtimes)
        Logger.success(f"Average Runtime over {repetitions} runs: {mean_runtime:.4f}s")
        return mean_runtime

class BenchmarkFilenameBuilder:
    def __init__(self, params):
        self.params = params

    def _size_tag(self):
        sizes = sorted(self.params.get("size", []))
        if not sizes:
            return "n0"
        if len(sizes) == 1:
            return f"n{sizes[0]}"
        return f"n{sizes[0]}-{sizes[-1]}"

    def _thread_tag(self):
        threads = [t for t in self.params.get("threads", []) if t is not None]
        if not threads:
            return "t1"
        if len(threads) == 1:
            return f"t{threads[0]}"
        return f"t{min(threads)}-{max(threads)}"

    def _program_tag(self):
        progs = self.params.get("program", [])
        if not progs:
            return "prog0"
        return f"{len(progs)}prog"

    def _parallel_tag(self):
        types = {p[1] for p in self.params.get("program", [])}
        if types == {"sequential"}:
            return "seq"
        if types == {"openmp"}:
            return "omp"
        return "mixed"

    def _compiler_tag(self):
        compilers = self.params.get("compiler", [])
        if not compilers:
            return "cc"
        return compilers[0]

    def build(self):
        parts = [
            "bench",
            self._parallel_tag(),
            self._program_tag(),
            self._size_tag(),
            self._thread_tag(),
            self._compiler_tag()
        ]

        return "_".join(parts) + ".csv"

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
        return tuple(str(combo[k]) for k in self.params.keys())

    def _ask_output_filename(self):
        default_name = BenchmarkFilenameBuilder(self.params).build()
        print("\n" + "=" * 60)
        print("BENCHMARK PIPELINE START")
        print("=" * 60)
        print("Choose a name for the results CSV file.")
        print(f"Press ENTER to use the default name: {default_name}")
        print("-" * 60)
        user_input = input("Result file name: ").strip()
        if user_input == "":
            filename = default_name
            print(f"Using default file name: {filename}")
        else:
            filename = user_input+".csv"
            print(f"Using custom file name: {filename}")
        self.csv_path = self.compiler.out_dir / filename
        print("=" * 60 + "\n")

    def run(self):
        self._ask_output_filename()
        self.completed_runs = self._load_cache()

        keys, values = list(self.params.keys()), list(self.params.values())
        combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        Logger.info(f"Starting pipeline: {len(combinations)} total configurations.")

        # Open file once, append mode
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(keys + ["mean_runtime"])
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
                
                runtime = Executor.run(binary, combo["size"], combo["threads"], repetitions=REPETITIONS)
                
                writer.writerow(list(csv_key) + [runtime])
                f.flush() # Force save to disk immediately

        Logger.success(f"Pipeline complete. Results saved to {self.csv_path}")

# ==========================================================
# ======================== MAIN ===========================
# ==========================================================
if __name__ == "__main__":
    load_intel_environment() # loads setvars when needed
    pipeline = BenchmarkPipeline(PARAMETERS, OUTPUT_FOLDER)
    pipeline.run()