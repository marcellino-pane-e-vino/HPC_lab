# Analisi Dettagliata: L'Evoluzione della Moltiplicazione di Matrici in CUDA

La moltiplicazione di matrici è l'algoritmo "Hello World" del calcolo parallelo ad alte prestazioni. È un problema computazionalmente intenso: per due matrici di dimensione $N \times N$, il numero di operazioni matematiche cresce con $O(N^3)$, mentre i dati necessari crescono con $O(N^2)$. Questa discrepanza tra calcoli e dati è il motivo per cui l'architettura della memoria della GPU gioca un ruolo così vitale. 

Analizziamo i tre approcci, mettendoli a confronto dal punto di vista dell'architettura hardware e della scalabilità al variare di $N$ (da 5.000 a 20.000).

---

## 1. L'Approccio Naive: Il Collo di Bottiglia della Memoria Globale

Il primo kernel che abbiamo testato è la traduzione diretta dell'algoritmo sequenziale in un modello parallelo. Ogni thread calcola un singolo elemento della matrice risultante $C$.

### I Dati (Scalabilità)
* **N = 5000:** 1.26 secondi
* **N = 10000:** 10.16 secondi (aumento di ~8x)
* **N = 15000:** 42.35 secondi 
* **N = 20000:** 97.81 secondi 

### Analisi e Problematiche
Dal punto di vista della scalabilità pura, l'algoritmo rispetta la complessità teorica: raddoppiando $N$ (da 5k a 10k), il tempo aumenta di circa un fattore 8 ($2^3$). Tuttavia, le performance assolute sono estremamente basse. Perché?

Il problema principale è il **rapporto Compute-to-Global-Memory-Access (CGMA)**. Nel kernel Naive, per eseguire un singolo calcolo di moltiplicazione e addizione (2 operazioni floating-point), il thread deve fare 2 letture dalla Memoria Globale (una da $A$ e una da $B$). La Memoria Globale ha una latenza enorme (centinaia di cicli di clock) e una banda limitata. Stiamo costringendo la GPU a rileggere gli stessi dati migliaia di volte, saturando il bus di memoria. La GPU passa il tempo ad "aspettare" i dati, risultando in un'esecuzione *Memory Bound*.

Inoltre, il codice Naive utilizza variabili `double` (FP64). Su architetture come la T4, le unità di calcolo a doppia precisione sono nettamente inferiori rispetto a quelle a singola precisione, penalizzando ulteriormente l'esecuzione pura.

---

## 2. Il Kernel Ottimizzato: Tiling e Thread Coarsening



Qui facciamo un salto di qualità architetturale, introducendo lo sfruttamento della **Memoria Condivisa (Shared Memory)**, una memoria on-chip velocissima progettata appositamente per la cooperazione tra thread di uno stesso blocco.

### I Dati (Scalabilità)
* **N = 5000:** 0.26 secondi
* **N = 10000:** 1.69 secondi
* **N = 15000:** 6.09 secondi
* **N = 20000:** 14.49 secondi

### Analisi dell'Ottimizzazione
I tempi crollano drasticamente. A $N=20000$, passiamo da quasi 100 secondi a soli 14 secondi. Questo risultato è frutto di due tecniche combinate:

1.  **Tiling (Memoria Condivisa):** Invece di leggere singole celle, i thread di un blocco si coordinano per caricare un'intera sottomatrice (Tile) nella Shared Memory. Una volta che il Tile è caricato, tutti i thread del blocco possono accedere a quei dati a velocità pari a quella della cache L1, riutilizzandoli più volte. Se il Tile è di dimensione $32 \times 32$, riduciamo gli accessi alla lenta Memoria Globale di un fattore 32. Il limite si sposta così dalla memoria ai core di calcolo (*Compute Bound*).
2.  **Thread Coarsening:** Un singolo thread ora non calcola più un solo elemento, ma ben 4 elementi contemporaneamente (uso dei registri). Questo ci permette di caricare un valore di $A$ dalla Shared Memory in un registro ultrarapido, e moltiplicarlo per 4 valori diversi di $B$. Riduce ulteriormente il traffico, persino all'interno della Shared Memory.
3.  **Precisione FP32:** L'utilizzo dei `float` dimezza i byte da muovere per ogni operazione e sblocca la massima potenza di calcolo parallelo della GPU T4.

---

## 3. cuBLAS: L'Ottimizzazione a Livello di Ferro

Infine, abbiamo testato cuBLAS, la libreria fornita direttamente da NVIDIA.

### I Dati (Scalabilità)
* **N = 10000:** 0.40 secondi
* **N = 15000:** 1.29 secondi
* **N = 20000:** 3.11 secondi

### Analisi dell'Esecuzione
A $N=20000$, cuBLAS distrugge letteralmente il nostro (pur ottimo) kernel custom, girando quasi 5 volte più veloce (3.11s contro 14.49s). Cosa fa di speciale questa libreria?

Non si limita ad applicare il Tiling. I kernel di cuBLAS sono scritti scendendo al livello del linguaggio Assembly (PTX/SASS), cuciti millimetricamente sulle specifiche architetturali della GPU ospite. 
Dal profiler vediamo che usa un kernel chiamato `volta_sgemm_128x128_nn`. Questo significa che usa Tile mastodontici (128x128), ma la vera magia risiede nel modo in cui nasconde la latenza: utilizza tecniche di **Prefetching e Double Buffering**. Mentre i CUDA core sono impegnati a calcolare i dati del Tile attuale, i controller di memoria stanno già pre-caricando il Tile successivo nei registri, in modo che l'unità matematica non resti inattiva nemmeno per un ciclo di clock.

---

## Riepilogo Scalabilità

La seguente tabella riassume l'efficienza in base alla dimensione del problema. Man mano che $N$ cresce, il divario si fa sempre più estremo:

| Dimensione ($N$) | Naive (FP64) | Tiled + Coarsened (FP32) | cuBLAS (FP32) | Miglioramento (cuBLAS vs Naive) |
| :--- | :--- | :--- | :--- | :--- |
| **5.000** | 1.26 s | 0.26 s | N/A (Velocissimo) | ~ 5x (Stimato) |
| **10.000** | 10.16 s | 1.69 s | 0.40 s | **~ 25x** |
| **15.000** | 42.35 s | 6.09 s | 1.29 s | **~ 32x** |
| **20.000** | 97.81 s | 14.49 s | 3.11 s | **~ 31x** |

**Conclusione:** Per matrici di dimensioni limitate ($N < 1000$), un kernel naive potrebbe persino sembrare accettabile perché la memoria cache della GPU riesce a mascherare un po' di inefficienza. Ma appena scaliamo verso problemi reali del mondo Big Data o Deep Learning ($N=20000$), la mancata ottimizzazione della Memoria Globale porta a tempi di esecuzione inaccettabili. L'uso della Shared Memory e del Tiling è obbligatorio, mentre per le applicazioni in produzione, affidarsi a librerie ultra-ottimizzate come cuBLAS è la scelta standard.
