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
* **N = 5000:** 1.35 secondi
* **N = 10000:** 0.40 secondi
* **N = 15000:** 1.29 secondi
* **N = 20000:** 3.11 secondi

### Analisi dell'Esecuzione
A $N=20000$, cuBLAS distrugge letteralmente il nostro (pur ottimo) kernel custom, girando quasi 5 volte più veloce (3.11s contro 14.49s). Cosa fa di speciale questa libreria?

Non si limita ad applicare il Tiling. I kernel di cuBLAS sono scritti scendendo al livello del linguaggio Assembly (PTX/SASS), cuciti millimetricamente sulle specifiche architetturali della GPU ospite. 
Dal profiler vediamo che usa un kernel chiamato `volta_sgemm_128x128_nn`. Questo significa che usa Tile mastodontici (128x128), ma la vera magia risiede nel modo in cui nasconde la latenza: utilizza tecniche di **Prefetching e Double Buffering**. Mentre i CUDA core sono impegnati a calcolare i dati del Tile attuale, i controller di memoria stanno già pre-caricando il Tile successivo nei registri, in modo che l'unità matematica non resti inattiva nemmeno per un ciclo di clock.

---

## Riepilogo Scalabilità cuBLAS vs Naive

La seguente tabella riassume l'efficienza in base alla dimensione del problema. Man mano che $N$ cresce, il divario si fa sempre più estremo:

| Dimensione ($N$) | Naive (FP64) | Tiled + Coarsened (FP32) | cuBLAS (FP32) | Miglioramento (cuBLAS vs Naive) |
| :--- | :--- | :--- | :--- | :--- |
| **5.000** | 1.26 s | 0.26 s | 1.35 s | **Nessuno** |
| **10.000** | 10.16 s | 1.69 s | 0.40 s | **~25x** |
| **15.000** | 42.35 s | 6.09 s | 1.29 s | **~33x** |
| **20.000** | 97.81 s | 14.49 s | 3.11 s | **~31x** |


## Riepilogo Scalabilità e Confronto Finale

La seguente tabella unificata riassume l'efficienza in base alla dimensione del problema, mettendo in luce sia l'abisso prestazionale rispetto al codice Naive, sia il divario tra un'ottima implementazione manuale e una libreria iper-ottimizzata:

| Dimensione ($N$) | Naive (FP64) | Tiled + Coarsened (FP32) | cuBLAS (FP32) | Miglioramento (cuBLAS vs Naive) | Miglioramento (cuBLAS vs Tiled) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **5.000** | 1.26 s | 0.26 s | 1.35 s | **Nessuno (Overhead)** | **Tiled più veloce (~5x)** |
| **10.000** | 10.16 s | 1.69 s | 0.40 s | **~25x** | **~4.2x** |
| **15.000** | 42.35 s | 6.09 s | 1.29 s | **~33x** | **~4.7x** |
| **20.000** | 97.81 s | 14.49 s | 3.11 s | **~31x** | **~4.6x** |

**Conclusione:** Analizzando i dati, emerge una dinamica fondamentale del calcolo su GPU e dell'ottimizzazione della memoria. 

Per matrici "piccole" ($N \le 5000$), il tempo di esecuzione di cuBLAS (1.35 s) appare paradossalmente il peggiore di tutti, persino superiore a quello del kernel Naive (1.26 s) e nettamente battuto dal nostro kernel Tiled (0.26 s). Questo non significa che l'algoritmo di cuBLAS sia inefficiente, ma evidenzia il pesante *overhead* di inizializzazione della libreria: la creazione dell'handle (`cublasCreate`), la configurazione del contesto e le allocazioni di memoria interne richiedono più tempo dell'operazione matematica stessa. Su carichi di lavoro leggeri, un kernel custom snello parte e finisce prima che cuBLAS abbia finito di "scaldare i motori".

Tuttavia, appena scaliamo verso dimensioni da mondo reale ($N \ge 10000$), l'overhead iniziale viene diluito e scompare, lasciando emergere la vera gerarchia delle performance. La mancata ottimizzazione degli accessi in Memoria Globale del Naive porta a tempi inaccettabili (quasi 100 secondi a $N=20000$), saturando totalmente la banda passante. 

Il confronto più interessante è sicuramente quello tra **Tiled e cuBLAS**. Applicando fedelmente i concetti teorici (Shared Memory per abbattere gli accessi globali e Thread Coarsening sui registri per massimizzare il riutilizzo), il nostro kernel Tiled fa un lavoro eccezionale, abbattendo il tempo da 97 a 14 secondi. Eppure, cuBLAS risulta costantemente **quasi 5 volte più veloce** della nostra ottimizzazione manuale (~4.6x a $N=20000$). 

Questo ci insegna una lezione cruciale: per quanto possiamo ottimizzare il codice C/C++, le librerie fornite dai produttori (scritte direttamente in assembly PTX/SASS) utilizzano euristiche hardware-specifiche, blocchi di Tiling enormi (es. 128x128) e prefetching aggressivo (Double Buffering) per nascondere il 100% della latenza. Comprendere e implementare il Tiling con la Shared Memory è un passaggio accademico e ingegneristico fondamentale per capire come "pensa" la GPU, ma quando si passa in produzione, la regola d'oro è sempre: affidati a cuBLAS.
