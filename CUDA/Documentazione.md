# Analisi Architetturale: Ottimizzazione della Moltiplicazione di Matrici in CUDA

La moltiplicazione di matrici ($C = A \times B$) è un problema di classe $O(N^3)$ per la computazione e $O(N^2)$ per i dati. Su architetture parallele come le GPU, la sfida principale non è quasi mai la potenza di calcolo grezza, ma la **gestione della gerarchia di memoria**. Affamare i core della GPU (lasciandoli in attesa dei dati) è l'errore più comune. 

In questa analisi confronteremo tre approcci eseguiti su un'architettura **NVIDIA Tesla T4 (Turing, sm_75)**, scalando la dimensione $N$ da 5.000 a 20.000.

---

## L'Elefante nella Stanza: Discrepanza dei Tipi di Dato (L'Errore nel Confronto)

Prima di analizzare i tempi, dobbiamo evidenziare una fortissima anomalia nei codici forniti:
* Il kernel **Naive** utilizza variabili di tipo `double` (**FP64**, doppia precisione).
* I kernel **Tiled** e **cuBLAS** utilizzano variabili di tipo `float` (**FP32**, singola precisione).

Questa non è una differenza da poco. La Tesla T4 è una GPU progettata per il machine learning inferenziale, dove la singola precisione (FP32) domina. I suoi core FP64 sono "castrati" via hardware per non cannibalizzare le schede data-center di fascia superiore: le prestazioni FP64 della T4 sono pari a **$1/32$** delle prestazioni FP32. 
Di conseguenza, il tempo disastroso del kernel Naive (quasi 100 secondi per $N=20000$) non è dovuto *solo* all'inefficienza della memoria, ma al fatto che stiamo costringendo l'hardware a eseguire istruzioni per cui non è ottimizzato. Un confronto puramente equo richiederebbe che tutti i kernel usassero `float`.

Inoltre, analizzando l'output del notebook cuBLAS, il comando per testare $N=5000$ ha in realtà profilato un binario chiamato con parametro 15000. Pertanto, il dato per $N=5000$ su cuBLAS è formalmente mancante (N/D).

---

## 1. Kernel Naive: Il "Memory Wall"

Il primo approccio mappa direttamente ogni elemento di $C$ a un singolo thread. 

```c
sum += a[row * n + k] * b[k * n + col];
