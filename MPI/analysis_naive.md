# Analisi Performance: Moltiplicazione di Matrici con MPI

Questo progetto analizza le prestazioni di un algoritmo di moltiplicazione di matrici quadrate ($2000 \times 2000$) parallelizzato con MPI. I test sono stati eseguiti variando il numero di processi ($np$) per misurare la scalabilità del sistema.

## 📊 Tabella dei Risultati

| Processi ($np$) | Tempo Totale (s) | Speedup ($S$) | Efficienza ($E$) |
| :--- | :--- | :--- | :--- |
| 1 | 25.6804 | 1.00x | 100% |
| 4 | 7.9186 | 3.24x | 81% |
| 8 | 4.0636 | 6.32x | 79% |
| **16** | **2.4501** | **10.48x** | **65%** |
| 32 | 3.2090 | 8.00x | 25% |
| 64 | 3.9870 | 6.44x | 10% |

---

## 📈 Analisi Tecnica

I dati mostrano un comportamento tipico dei sistemi a memoria distribuita, caratterizzato da tre fasi distinte:

### 1. Scalabilità Ottimale (fino a 16 processi)
Fino a $np=16$, il tempo di esecuzione diminuisce in modo significativo. Questo accade perché la taglia della matrice ($n=2000$) genera un carico di lavoro (complessità $O(n^3)$ sufficientemente grande da "nascondere" il tempo speso per inviare i dati tra i processi.



### 2. Il Punto di Ottimo ($np=16$)
Con 16 processi si ottiene il tempo minimo (**2.45 secondi**). Oltre questo punto, il beneficio aggiunto da nuovi processori non riesce più a compensare il costo della coordinazione MPI.

### 3. Degradazione e Legge di Amdahl ($np > 16$)
Dalle prove con 32 e 64 processi si nota un aumento dei tempi. Questo fenomeno è spiegato dai seguenti fattori:

* **Overhead di Comunicazione:** Funzioni come `MPI_Bcast` e `MPI_Gather` devono gestire un numero maggiore di messaggi. Il tempo speso a "parlare" supera quello speso a "calcolare".
* **Saturazione Hardware:** Probabilmente il numero di core fisici della macchina è stato superato. Quando i processi sono più dei core, il sistema operativo deve scambiarli continuamente (context switching), rallentando tutto.
* **Legge di Amdahl:** La velocità è limitata dalla parte del codice che non può essere parallelizzata (es. l'invio iniziale dei dati dal Rank 0).



---

## ⚙️ Dettagli Implementativi
- **Kernel:** Ottimizzato con ordine dei cicli `i-k-j` per massimizzare l'uso della cache.
- **Comunicazione:** Utilizzo di comunicazioni collettive (`MPI_Bcast`, `MPI_Scatter`, `MPI_Gather`) per una distribuzione efficiente del carico.
- **Fabric:** Test eseguiti con `I_MPI_FABRICS=shm` (Shared Memory) per minimizzare la latenza di comunicazione locale.



## 🚀 Come replicare i test
Compila il codice:
```bash
mpicc matrixmult_mpi.c -o matrixmult_mpi
```
---

### Un piccolo appunto sui risultati
Efficienza crolla al **10%** con 64 processi? Questo è il segnale classico che stiamo usando troppi operai per un lavoro troppo piccolo. Se volessimo far tornare l'efficienza alta con 64 processi, dovremmo aumentare $n$ (ad esempio a 5000 o più), in modo che ogni processo abbia di nuovo abbastanza calcoli da fare per "giustificare" il tempo perso a scambiarsi i messaggi.
