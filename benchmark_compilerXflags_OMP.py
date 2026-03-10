#!/usr/bin/env python3
"""
============================================================
C Benchmark Framework: Fast Extreme Performance Test
============================================================
"""

import subprocess
import csv
import sys
from pathlib import Path

# ==========================================================
# ================== CONFIGURATION =========================
# ==========================================================

C_FILE = "omp_matrixmult.c"

# Le tue griglie di test
N_VALUES = [5000, 10000, 15000]
THREADS_VALUES = [8, 16, 24]

COMPILERS = ["icc", "icx"]

# L'unica configurazione che ci interessa: MASSIMA POTENZA
MAX_OPT_FLAGS = {
    "icc": ["-qopenmp", "-O3", "-lm", "-xHost", "-fast", "-DALIGNED", "-DNOFUNCCALL", "-ipo"],
    "icx": ["-qopenmp", "-O3", "-lm", "-xHost", "-fast", "-DALIGNED", "-DNOFUNCCALL", "-ipo"]
}

# Output folder
OUTPUT_FOLDER = Path(__file__).parent / "benchmarks"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ==========================================================
# ============== FUNCTIONS / CLASSES ======================
# ==========================================================

class Compiler:
    @staticmethod
    def compile(source_file: Path, compiler: str, flags: list) -> Path:
        source_file = source_file.resolve()
        binary = OUTPUT_FOLDER / f"{source_file.stem}_{compiler}_extreme"

        print(f"\n[COMPILING] {compiler} con massima ottimizzazione...")
        print(f"  Flags: {' '.join(flags)}")

        cmd = [compiler, str(source_file), "-o", str(binary)] + flags

        try:
            compile_command = " ".join(cmd)
            subprocess.run(
                ["bash", "-c",
                 f"source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 && {compile_command}"],
                check=True, capture_output=True, text=True
            )
            print(f"  ✔ Compilazione completata → {binary.name}")

        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Compilazione fallita per {compiler}")
            print(e.stderr)
            sys.exit(1)

        return binary

class Executor:
    @staticmethod
    def run(binary: Path, n: int, threads: int) -> float:
        binary_path = str(binary.resolve())
        print(f"  → Calcolo in corso: N={n:<6} | Threads={threads:<3} ...", end="", flush=True)

        try:
            result = subprocess.run(
                [binary_path, str(n), str(threads)],
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError:
            print(f" [ERRORE di Esecuzione]")
            sys.exit(1)

        for line in result.stdout.splitlines():
            if "Tempo di esecuzione OpenMP" in line:
                time = float(line.split()[-2])
                print(f" {time:.4f} s")
                return time

        print(" [ERRORE: Tempo non trovato]")
        sys.exit(1)

# ==========================================================
# ====================== MAIN ==============================
# ==========================================================

def main():
    print("\n[START] Avvio test di massima performance")
    
    # Dizionario per salvare i risultati: results[compiler][thread][n]
    results = {c: {t: {} for t in THREADS_VALUES} for c in COMPILERS}

    # 1. Fase di compilazione ed esecuzione per ogni compilatore
    for compiler in COMPILERS:
        flags = MAX_OPT_FLAGS[compiler]
        binary = Compiler.compile(Path(C_FILE), compiler, flags)
        
        print(f"\n[TESTING] Avvio test per {compiler}...")
        for t in THREADS_VALUES:
            for n in N_VALUES:
                time = Executor.run(binary, n, t)
                results[compiler][t][n] = time

    # 2. Fase di stampa tabelle a schermo e salvataggio CSV
    print("\n" + "="*60)
    print(" RISULTATI FINALI ".center(60, "="))
    print("="*60)

    for compiler in COMPILERS:
        # Stampa a schermo
        print(f"\nCompilatore: [{compiler.upper()}]")
        header_str = f"{'Threads':<10} | " + " | ".join([f"N={n:<8}" for n in N_VALUES])
        print("-" * len(header_str))
        print(header_str)
        print("-" * len(header_str))
        
        for t in THREADS_VALUES:
            row_str = f"{t:<10} | "
            row_str += " | ".join([f"{results[compiler][t][n]:<10.4f}" for n in N_VALUES])
            print(row_str)
        
        print("-" * len(header_str))

        # Salvataggio su CSV dedicato per ogni compilatore
        output_csv = OUTPUT_FOLDER / f"Tabella_Estrema_{compiler.upper()}.csv"
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Intestazione CSV
            csv_headers = ["Threads"] + [f"N={n}" for n in N_VALUES]
            writer.writerow(csv_headers)
            
            # Righe CSV
            for t in THREADS_VALUES:
                row = [t] + [results[compiler][t][n] for n in N_VALUES]
                writer.writerow(row)
                
        print(f"-> Dati salvati in: {output_csv.name}")

    print("\n[FINISH] Tutto completato con successo!\n")

if __name__ == "__main__":
    main()