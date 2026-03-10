# Analisi Prestazionale: Compilatori Intel ICC vs ICX su Matrici OpenMP

Questo report documenta il comportamento dell'algoritmo di moltiplicazione tra matrici (5000x5000) sfruttando l'ecosistema di compilazione Intel. L'obiettivo è confrontare lo storico compilatore *Intel C++ Compiler Classic* (`icc`) con il suo moderno successore basato su LLVM, *Intel oneAPI DPC++/C++ Compiler* (`icx`), misurando i tempi di esecuzione e l'efficienza dello *speedup* al variare dei thread e dei flag di ottimizzazione.

## 📊 Tabella dei Risultati Sperimentali

| Compilatore | Ottimizzazione | Flag Architettura | Thread (OMP_NUM_THREADS) | Tempo (Secondi) |
| :--- | :--- | :--- | :---: | :--- |
| **ICC** | Nessuna (Base) | `-xHost` | *Tutti* (Default) | 5.389 s |
| **ICC** | Nessuna (Base) | `-xHost` | **1** | 47.789 s |
| **ICC** | Nessuna (Base) | `-xHost` | **8** | 9.213 s |
| **ICC** | Nessuna (Base) | `-xHost` | **24** | 5.164 s |
| **ICC** | `-O3` | `-xHost` | **1** | 5.047 s |
| **ICC** | `-O3` | `-xHost` | **8** | 0.816 s |
| **ICC** | `-O3` | `-xHost` | **24** | 0.604 s |
| **ICX** | Nessuna (Base) | `-xHost` | **1** | 4.995 s |
| **ICX** | Nessuna (Base) | `-xHost` | **8** | 0.840 s |
| **ICX** | Nessuna (Base) | `-xHost` | **24** | 0.607 s |
| **ICX** | `-O3` | `-xHost` | **1** | 4.790 s |
| **ICX** | `-O3` | `-xHost` | **8** | 0.660 s |
| **ICX** | `-O3` | `-xHost` | **24** | 0.656 s |

---

## 🔬 Analisi Approfondita dei Dati
Leggendo le metriche, emergono dinamiche fondamentali sul funzionamento dei compilatori in ambito HPC (High Performance Computing), confermando empiricamente i principi architetturali illustrati a lezione.

### 1. Il passaggio di testimone: ICC vs ICX e il "miracolo" LLVM
La prima cosa che salta all'occhio durante l'uso di `icc` è il *remark #10441*. Il compilatore storico di Intel è stato deprecato in favore di `icx`. E guardando i dati "Base" (senza `-O3`), capiamo subito il perché.
Quando usiamo `icc` senza spingere le ottimizzazioni al massimo, un singolo thread impiega **quasi 48 secondi** per completare il calcolo. Quando passiamo a `icx` con le stesse identiche istruzioni (solo `-xHost`), il tempo per un singolo thread è di **circa 5 secondi**. 

Cosa significa questo? Che `icx`, appoggiandosi all'infrastruttura LLVM, è estremamente più aggressivo e intelligente "di fabbrica". Riesce a individuare i pattern matematici dei cicli annidati e ad applicare la vettorializzazione hardware in automatico, senza aver bisogno che lo sviluppatore forzi esplicitamente i flag di ottimizzazione più alti.

### 2. L'impatto esplosivo dell'ottimizzazione (-O3)
Se con `icx` il flag `-O3` lima solo qualche frazione di secondo (perché il grosso del lavoro SIMD è già stato fatto in partenza), con il vecchio `icc` il parametro `-O3` è letteralmente questione di vita o di morte prestazionale. 
Passare da 47.7 secondi a 5.04 secondi (a parità di 1 singolo thread) significa aver ordinato a `icc` di srotolare brutalmente i cicli (*loop unrolling*) e di compattare le operazioni scalari in enormi registri vettoriali (SIMD). È una dimostrazione didatticamente perfetta di quanto il software possa fare la differenza sullo stesso identico pezzo di silicio, ricalcando gli esempi di auto-parallelizzazione dei report del compilatore.

### 3. Scalabilità e il "Memory Wall"
Concentriamoci sui run più prestanti in assoluto (quelli con `-O3` e `-xHost`) per calcolare lo *Speedup* parallelo.

* **Da 1 a 8 Thread:** Con ICX passiamo da 4.79s a 0.66s. Abbiamo uno *speedup* impressionante di **~7.2x**. Il lavoro viene diviso quasi perfettamente. I thread macinano calcoli in parallelo sfruttando al meglio la memoria allocata con la First-Touch policy NUMA che abbiamo implementato nel codice.

* **Da 8 a 24 Thread:** Qui la curva crolla. Sia ICC (0.604s) che ICX (0.656s) si fermano intorno ai 6 decimi di secondo. Triplicare la forza lavoro non serve a nulla, perché non stiamo più misurando la velocità della CPU, ma quella della RAM. Abbiamo sbattuto violentemente contro il **Memory Wall**: i core della CPU sono diventati così mostruosamente veloci grazie alla vettorializzazione e alla parallelizzazione che finiscono il loro lavoro in attesa che la memoria di sistema sia fisicamente in grado di consegnargli i prossimi blocchi della matrice.
