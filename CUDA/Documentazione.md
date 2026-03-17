# **Ottimizzazione della Moltiplicazione di Matrici su GPU (N=15000)**

Questo documento analizza e confronta due diverse implementazioni per la moltiplicazione di due matrici quadrate di dimensione $15.000 \times 15.000$ in singola precisione (FP32) su architettura NVIDIA (GPU T4). L'obiettivo è evidenziare il salto prestazionale tra un kernel CUDA ottimizzato manualmente e l'utilizzo della libreria matematica di livello industriale fornita da NVIDIA.

## 1. Implementazione 1: Tiling Manuale in Shared Memory (`cuda_matrixmult`)

La prima versione utilizza un approccio "artigianale" ma altamente istruttivo. L'algoritmo non attacca la memoria globale della GPU in modo ingenuo, ma sfrutta una tecnica chiamata **Tiling**.

### Concetto Architetturale
Invece di far leggere ai thread i singoli elementi direttamente dalla lenta memoria globale (VRAM), l'elaborazione viene divisa in "piastrelle" (Tile) da $32 \times 32$ elementi. 
I thread di un blocco collaborano per prelevare una piastrella alla volta, caricarla nella **Shared Memory** (una cache ultra-veloce condivisa tra i thread dello stesso blocco) e poi eseguire tutte le moltiplicazioni incrociate su questi dati a latenza quasi nulla. La sincronizzazione (barriere) garantisce che nessun thread legga o scriva fuori tempo.

### Analisi del Profiling (`nvprof`)
L'esecuzione di questa implementazione ha generato i seguenti riscontri:

* **Tempo di esecuzione totale (Kernel):** $\approx 7.33$ secondi.
* **Carico di lavoro GPU:** Analizzando la sezione *GPU activities*, vediamo che il kernel `matMulTiledFloat` monopolizza le risorse fisiche occupando il **92.33%** del tempo di attività totale della scheda.
* **Trasferimenti di Memoria:** Le operazioni di copia da Host a Device (RAM $\rightarrow$ VRAM) e da Device a Host (VRAM $\rightarrow$ RAM) hanno richiesto rispettivamente circa $383$ ms e $226$ ms. In termini percentuali, questi trasferimenti pesano pochissimo (circa l'8% totale), confermando che il programma è **Compute-Bound**: il collo di bottiglia principale è la pura capacità aritmetica (ALU) e la latenza dei registri nel calcolare i 3.375 miliardi di operazioni.

---

## 2. Implementazione 2: La potenza di cuBLAS (`cuda_matrixmult_cuBLAS`)

La seconda versione abbandona la gestione manuale di griglie, blocchi e memoria condivisa, delegando l'intera operazione matematica a **cuBLAS** (CUDA Basic Linear Algebra Subprograms), la libreria di riferimento sviluppata direttamente dagli ingegneri NVIDIA.

### Concetto Architetturale
Richiamando la funzione standard `cublasSgemm` (Single precision General Matrix Multiply), stiamo chiedendo alla GPU di usare algoritmi scritti in puro linguaggio assembly. 
A differenza del nostro codice generico, cuBLAS riconosce fisicamente l'architettura su cui sta girando (la GPU T4 basata su architettura Turing/Volta) e, se possibile, accende i circuiti hardware specializzati (come i Tensor Cores) per macinare mini-matrici in singoli cicli di clock, gestendo il riposizionamento in memoria in modo totalmente trasparente.

### Analisi del Profiling (`nvprof`)
I risultati di questa versione sono sbalorditivi e rivelano dettagli profondi sul funzionamento dell'hardware:

* **Tempo di esecuzione totale (Kernel):** $\approx 1.35$ secondi.
* **Il vero nome del Kernel:** Il profiler non mostra più il nostro kernel, ma rivela che cuBLAS ha lanciato in background `volta_sgemm_128x128_nn`. Questo ci indica che la libreria utilizza istruzioni ultra-ottimizzate derivate dall'architettura Volta, usando piastrelle (Tile) gigantesche da $128 \times 128$ che noi, scrivendo codice C normale, non saremmo mai riusciti a mappare efficacemente nei registri senza mandare in crash il compilatore.
* **Intelligenza a Run-time:** Nelle chiamate API (*API calls*) emerge la funzione `cudaOccupancyMaxActiveBlocksPerMultiprocessor`. Questo significa che cuBLAS non tira a indovinare le dimensioni della griglia, ma interroga l'hardware in tempo reale per calcolare esattamente quanti blocchi lanciare per saturare al 100% i multiprocessori della scheda.

---

## 3. Confronto Diretto e "The Memory Wall"

Mettendo i due risultati fianco a fianco, emerge la vera lezione di questa ottimizzazione.

1. **Speedup Puro:** Passare dal tiling manuale a cuBLAS ha garantito un'accelerazione di **$\approx 5.4\times$**, portando i tempi di puro calcolo da oltre 7 secondi a poco più di un secondo.
2. **Spostamento del Collo di Bottiglia:** Questo è il dato più affascinante. Osserviamo i tempi fisici delle `cudaMemcpy` (Host $\rightarrow$ Device e Device $\rightarrow$ Host). In entrambe le versioni, spostare le matrici attraverso il cavo PCI-Express richiede circa $600$ millisecondi totali. La fisica non si batte: i byte sono gli stessi e il cavo è lo stesso.
3. **Il "Memory Wall":** Nel primo codice, quei $600$ ms di trasferimento erano impercettibili rispetto ai $7.3$ secondi di matematica (erano solo l'8% del tempo). Usando cuBLAS, la scheda esegue la matematica in un lampo ($1.35$s). Di conseguenza, quei $600$ ms di trasferimento dati sono improvvisamente diventati quasi il **30% del tempo totale di attività della GPU** ($19.81\%$ per l'invio, $10.49\%$ per la ricezione).

### Conclusione
Abbiamo spremuto l'unità logico-aritmetica della T4 così a fondo da colpire il "muro della memoria". Il programma non è più fermato da "quanto è brava la GPU a moltiplicare", ma da "quanto è veloce la scheda madre a fornire i numeri alla GPU". È il passaggio definitivo da un problema **Compute-Bound** a un problema **Memory/Bandwidth-Bound**.
