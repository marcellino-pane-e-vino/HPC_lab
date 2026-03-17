# Analisi e Ottimizzazione della Moltiplicazione di Matrici su GPU (N=15000)

Il presente documento espone l'analisi prestazionale di due diverse implementazioni per la moltiplicazione di due matrici quadrate di dimensione $15.000 \times 15.000$ su architettura NVIDIA (GPU T4). L'obiettivo è confrontare un kernel CUDA ottimizzato manualmente con l'utilizzo della libreria matematica di livello industriale cuBLAS, evidenziando le motivazioni architetturali alla base delle prestazioni ottenute.

## 1. Scelta del Formato Dati: Precisione Singola (FP32)

Prima di analizzare le implementazioni, è fondamentale giustificare la scelta del formato a virgola mobile a precisione singola (`float` a 32 bit, o FP32) rispetto alla doppia precisione (`double` a 64 bit). Questa decisione tecnica si basa su tre fattori critici dell'architettura hardware:

1.  **Dimezzamento del Footprint di Memoria:** Un elemento FP32 occupa 4 byte contro gli 8 byte di un FP64. Per tre matrici di dimensioni $15.000 \times 15.000$, l'occupazione totale scende da circa 5.4 GB a 2.7 GB. Questo dimezza il carico sul bus PCI-Express durante i trasferimenti Host-to-Device e Device-to-Host.
2.  **Ottimizzazione della Shared Memory:** Poiché la Shared Memory per ogni blocco è fisicamente limitata, l'utilizzo di FP32 permette di caricare "piastrelle" (tile) più grandi a parità di byte, aumentando il riuso dei dati nei registri e riducendo gli accessi alla lenta memoria globale.
3.  **Throughput Hardware:** Le GPU moderne, e in particolare l'architettura Turing (su cui è basata la T4), dispongono di un numero di core dedicati alle operazioni FP32 e INT32 nettamente superiore rispetto a quelli FP64. L'utilizzo della precisione singola sblocca il massimo potenziale teorico delle ALU (Arithmetic Logic Unit) della scheda.



---

## 2. Implementazione 1: Tiling Manuale in Shared Memory

La prima versione analizzata implementa la moltiplicazione tramite un kernel CUDA scritto ad hoc, utilizzando la tecnica del **Tiling**.

### Concetto Architetturale
Per evitare che i thread accedano ripetutamente alla memoria globale (caratterizzata da elevata latenza), l'algoritmo suddivide il calcolo in blocchi (Tile) di dimensione $32 \times 32$. I thread appartenenti al medesimo blocco collaborano per caricare una singola Tile nella **Shared Memory**. Successivamente, mediante l'ausilio di barriere di sincronizzazione (`__syncthreads()`), eseguono le moltiplicazioni sui dati presenti in questa memoria cache ultra-veloce. Il processo viene ripetuto facendo scorrere la Tile lungo la riga e la colonna delle matrici di origine.



### Analisi del Profiling (`nvprof`)
L'esecuzione ha restituito i seguenti risultati:
* **Tempo di esecuzione totale (Kernel):** $\approx 7.33$ secondi.
* **Carico di lavoro GPU:** Il kernel `matMulTiledFloat` impegna la GPU per il **92.33%** del tempo di esecuzione complessivo.
* **Trasferimenti di Memoria:** Le copie Host-to-Device (RAM $\rightarrow$ VRAM) e Device-to-Host (VRAM $\rightarrow$ RAM) richiedono complessivamente circa $600$ ms. 
* **Diagnosi:** Rappresentando i trasferimenti di memoria solo l'8% del tempo totale, questa implementazione risulta fortemente **Compute-Bound**. Il collo di bottiglia risiede unicamente nella capacità di calcolo del kernel rispetto all'enorme mole di operazioni richieste (circa 3.375 miliardi di moltiplicazioni-addizioni).

---

## 3. Implementazione 2: Ottimizzazione Avanzata tramite cuBLAS

La seconda versione delega l'intera operazione alla libreria **cuBLAS** (CUDA Basic Linear Algebra Subprograms), sfruttando routine implementate a livello assembly da NVIDIA.

### Concetto Architetturale
Richiamando la funzione `cublasSgemm`, il controllo passa a un codice altamente specializzato che adatta l'esecuzione all'hardware sottostante. A differenza dell'approccio manuale, cuBLAS è in grado di interrogare l'architettura a run-time, massimizzando l'occupazione dei multiprocessori e, se l'architettura lo consente, attivando i Tensor Cores per accelerare le operazioni matriciali.



### Analisi del Profiling (`nvprof`)
I log del profiler hanno evidenziato una trasformazione radicale:
* **Tempo di esecuzione totale (Kernel):** $\approx 1.35$ secondi.
* **Il Kernel `volta_sgemm_128x128_nn`:** cuBLAS ha automaticamente sostituito l'approccio generico invocando questo specifico kernel ottimizzato. La sigla indica l'utilizzo di una Tile logica enorme, di dimensioni **$128 \times 128$**. Mappare una Tile del genere a livello di codice C standard esaurirebbe rapidamente i registri disponibili, ma l'implementazione assembly di cuBLAS riesce a orchestrare il caricamento in cache L1, Shared Memory e file di registri in modo perfetto, azzerando i cicli di stallo (stall cycles) delle ALU.
* **Occupazione Dinamica:** È stata rilevata la chiamata API `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, confermando che cuBLAS calcola dinamicamente la dimensione ottimale della griglia per mantenere il 100% del silicio attivo.

---

## 4. Analisi Comparativa: L'Impatto del "Memory Wall"

Il confronto tra le due implementazioni evidenzia un cambio di paradigma nell'analisi dei colli di bottiglia.

1.  **Accelerazione (Speedup):** Il passaggio da un Tiling manuale a cuBLAS ha prodotto uno speedup di **$\approx 5.4\times$**, innalzando le prestazioni effettive a circa 5 TeraFLOPS (mille miliardi di operazioni in virgola mobile al secondo).
2.  **Costanza del Trasferimento Dati:** I tempi necessari per trasferire i dati via bus PCI-Express (`cudaMemcpy`) sono rimasti identici in entrambi i test (circa $600$ ms totali), essendo vincolati da limiti fisici hardware (banda passante della scheda madre e dimensione in byte delle matrici).
3.  **Spostamento del Collo di Bottiglia:** Nel primo scenario, i $600$ ms di trasferimento rappresentavano una quota marginale rispetto ai $7.33$ secondi di elaborazione aritmetica. Nell'implementazione cuBLAS, a fronte di un tempo di calcolo di soli $1.35$ secondi, i medesimi $600$ ms sono giunti a pesare per quasi il **30% del tempo totale** di attività della GPU.



### Conclusioni
Le ottimizzazioni hardware introdotte da cuBLAS (tile da 128x128, uso intensivo dei registri e istruzioni specifiche per l'architettura) hanno saturato completamente le unità aritmetiche della scheda. Di conseguenza, il sistema ha raggiunto il cosiddetto **Memory Wall**: l'algoritmo non è più limitato dalla capacità di calcolo (Compute-Bound), bensì dalla velocità con cui l'infrastruttura di interconnessione (PCIe) riesce a fornire i dati alla memoria della GPU (Bandwidth-Bound).
