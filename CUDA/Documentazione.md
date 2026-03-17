# Analisi dell'Ottimizzazione per la Moltiplicazione di Matrici su GPU (N=15000)

Il presente documento illustra il processo di ottimizzazione e l'analisi prestazionale relativi alla moltiplicazione di due matrici dense di dimensione $15.000 \times 15.000$ in singola precisione (FP32). L'operazione richiede l'esecuzione di oltre 3,375 miliardi di moltiplicazioni e altrettante addizioni, per un totale di circa 6,75 Tera-operazioni. I test sono stati condotti su un'architettura NVIDIA T4. 

Di seguito vengono messe a confronto due diverse metodologie di implementazione, analizzando i dati di profilazione per comprendere le dinamiche architetturali sottostanti.

---

## 1. Implementazione Custom: Tiling in Shared Memory

La prima implementazione esaminata si avvale di un kernel CUDA sviluppato ad hoc, progettato per mitigare le latenze di accesso alla memoria globale della GPU tramite la tecnica del **Tiling**. 


Invece di far accedere i singoli thread direttamente alla memoria VRAM (capiente ma caratterizzata da un'alta latenza), il calcolo viene suddiviso in "piastrelle" (Tile) da $32 \times 32$ elementi. I thread all'interno di un blocco collaborano per caricare una piastrella alla volta all'interno della *Shared Memory* (una memoria cache a bassissima latenza condivisa a livello di blocco multiprocessore). Una volta caricati i dati, vengono eseguite le operazioni aritmetiche, per poi procedere iterativamente alla piastrella successiva.

**Risultati della profilazione:**
L'esecuzione di questa implementazione ha registrato un tempo di calcolo puro (latenza del kernel) di **7,33 secondi**.
L'analisi dei log rivela che la GPU ha dedicato oltre il 92% del tempo totale di attività all'esecuzione delle operazioni matematiche. Di contro, il trasferimento fisico dei dati dalla memoria RAM dell'host alla VRAM del device (tramite bus PCI-Express) ha richiesto circa 600 millisecondi, incidendo solo per l'8% sul tempo complessivo. 
Questo scenario configura un classico problema **"Compute-Bound"**: il fattore limitante delle prestazioni è la pura capacità di calcolo (il throughput delle ALU), mentre la larghezza di banda verso la memoria non rappresenta un collo di bottiglia significativo.

---

## 2. Implementazione Industriale: Utilizzo della libreria cuBLAS

La seconda implementazione sostituisce il kernel custom con l'utilizzo di **cuBLAS**, la libreria di algebra lineare ottimizzata nativamente da NVIDIA. 


La profilazione rivela che cuBLAS non si limita a lanciare un kernel statico. Prima dell'esecuzione vera e propria, la libreria effettua un'interrogazione dell'hardware (tramite chiamate API come `cudaOccupancyMaxActiveBlocksPerMultiprocessor`) per determinare il numero esatto di multiprocessori disponibili sulla T4. Sulla base di questi dati, viene lanciato un kernel altamente specializzato (denominato `volta_sgemm_128x128_nn`), scritto in linguaggio assembly e capace di processare enormi piastrelle da $128 \times 128$ elementi, massimizzando l'occupazione dei registri e attivando le istruzioni hardware più efficienti disponibili sul silicio.

**Risultati della profilazione:**
Il tempo di esecuzione del kernel è sceso drasticamente a **1,35 secondi**, registrando uno speedup di oltre 5 volte rispetto alla versione custom e sviluppando una potenza di calcolo pratica di circa 5 TeraFLOPS.

---

## 3. Analisi Comparativa: L'impatto del "Memory Wall"

Il confronto tra le due implementazioni permette di osservare empiricamente il limite fisico noto come **Memory Wall** (o muro della memoria).

Il tempo necessario per trasferire le matrici originali dalla CPU alla GPU attraverso il bus PCIe è rimasto invariato in entrambi i test (circa 600 millisecondi totali), trattandosi di un limite fisico dettato dall'hardware di interconnessione e dal volume dei dati (byte).


Tuttavia, l'impatto relativo di questo trasferimento cambia radicalmente:
* Nel primo scenario (Tiling manuale), 600 millisecondi di trasferimento risultavano marginali rispetto agli oltre 7 secondi necessari per l'elaborazione matematica.
* Nel secondo scenario (cuBLAS), l'ottimizzazione estrema del calcolo ha ridotto il tempo del kernel a soli 1,35 secondi. Di conseguenza, i 600 millisecondi di trasferimento dati sono arrivati a occupare **quasi il 30% del tempo totale** di attività della scheda.

**Conclusioni:**
L'adozione di software altamente ottimizzato ha permesso di saturare completamente la capacità logico-aritmetica dell'architettura T4. Questo processo ha causato uno spostamento del collo di bottiglia del sistema: l'algoritmo è passato dall'essere limitato dalla velocità di calcolo (*Compute-Bound*) all'essere pesantemente limitato dalla larghezza di banda del bus di sistema (*Bandwidth-Bound*), dimostrando come, a regimi di altissime prestazioni, il tempo speso per fornire i dati ai processori diventi il fattore critico predominante.
