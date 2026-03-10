# Report Analisi Prestazioni: Moltiplicazione Matrici OpenMP (N=10000*)

Questo documento analizza le prestazioni del calcolo matriciale su scala maggiore. L'incremento della dimensione della matrice evidenzia ancora di più le differenze tra i compilatori e i limiti fisici dell'hardware (bandwidth di memoria).

## 📊 Tabella dei Risultati (Carico di lavoro esteso)

| Compilatore | Ottimizzazione | Thread (OMP_NUM_THREADS) | Tempo (Secondi) |
| :--- | :--- | :---: | :--- |
| **ICC** (Classic) | Nessuna (Base) | **1** | 362.063 s |
| **ICC** (Classic) | Nessuna (Base) | **8** | 74.129 s |
| **ICC** (Classic) | Nessuna (Base) | **24** | 43.912 s |
| **ICC** (Classic) | `-O3 -xHost` | **1** | 40.307 s |
| **ICC** (Classic) | `-O3 -xHost` | **8** | 5.368 s |
| **ICC** (Classic) | `-O3 -xHost` | **24** | **4.336 s** |
| **ICX** (LLVM)    | Nessuna (Base) | **1** | 40.072 s |
| **ICX** (LLVM)    | Nessuna (Base) | **8** | 6.049 s |
| **ICX** (LLVM)    | Nessuna (Base) | **24** | 5.234 s |
| **ICX** (LLVM)    | `-O3 -xHost` | **1** | 38.275 s |
| **ICX** (LLVM)    | `-O3 -xHost` | **8** | 6.078 s |
| **ICX** (LLVM)    | `-O3 -xHost` | **24** | 5.232 s |

---

## 🔬 Analisi dei Fenomeni Osservati

### 1. Il crollo del "Legacy" (ICC Base)
Il dato più impressionante sono i **362 secondi** (circa 6 minuti) di ICC in modalità base. Senza ottimizzazioni esplicite, il vecchio compilatore non riesce a vettorializzare il codice. Il processore lavora in modalità puramente scalare, sprecando gran parte della sua potenza di calcolo. 
Al contrario, **ICX base** impiega solo **40 secondi** per lo stesso compito: una velocità **9 volte superiore** dovuta alla capacità del motore LLVM di applicare ottimizzazioni SIMD (Single Instruction Multiple Data) anche senza parametri aggressivi.

### 2. Vettorializzazione e "Instruction Throughput"
Con l'attivazione di `-O3 -xHost`, la situazione si livella. Entrambi i compilatori riescono a "domare" l'architettura host. Passare da 362s a 40s (per ICC) o da 40s a 38s (per ICX) sul singolo thread conferma che la **vettorializzazione** è il vero motore della performance: una volta che il codice è vettorializzato correttamente, i margini di miglioramento sul calcolo puro si riducono.

### 3. Speedup e Scalabilità Parallela
Analizzando lo *Speedup* ($S = \frac{T_{1}}{T_{N}}$) per la configurazione ICC -O3:
* **8 Thread:** $S = \frac{40.3}{5.3} \approx 7.5x$ (Efficienza vicina all'ideale).
* **24 Thread:** $S = \frac{40.3}{4.3} \approx 9.3x$.

Qui notiamo un dato fondamentale delle slide del corso: lo **Speedup non è infinito**. Passando da 8 a 24 thread, nonostante la potenza di calcolo sia triplicata, il guadagno temporale è minimo. 

### 4. Il "Memory Wall" su larga scala
Con matrici così grandi, il sistema sposta una quantità massiccia di dati. Siamo arrivati alla saturazione del bus di memoria: i core della CPU sono pronti a calcolare, ma la RAM non riesce a fornire i dati a una velocità sufficiente per alimentare tutti i 24 thread contemporaneamente. Questo fenomeno, unito all'overhead di gestione di un team di thread così numeroso, spiega perché il tempo si assesta intorno ai 4-5 secondi indipendentemente dal compilatore usato.

---
**Conclusione:** Per matrici di queste dimensioni, l'ottimizzazione del codice (vettorializzazione) è più impattante della mera aggiunta di thread oltre una certa soglia (8-12 thread nel tuo sistema), confermando che il calcolo è diventato **Memory Bound**.