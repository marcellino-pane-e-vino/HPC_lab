# Analisi Performance: MPI Matrix Multiplication - Tiled vs Naive

Questo documento analizza i risultati ottenuti con la versione **Tiled** della moltiplicazione di matrici ($n=2000$) e ne confronta l'efficienza rispetto alla versione **Naive** precedentemente testata.

## 📊 Confronto Risultati: Naive vs Tiled

| Processi ($np$) | Tempo Naive (s) | Tempo Tiled (s) | Guadagno (Tiled vs Naive) |
| :--- | :--- | :--- | :--- |
| 1 | 25.6804 | **16.8715** | **+34.3%** |
| 4 | 7.9186 | **4.2206** | **+46.7%** |
| 8 | 4.0636 | **2.1358** | **+47.4%** |
| 16 | 2.4501 | **1.4465** | **+40.9%** |
| 32 | 3.2090 | **1.6060** | **+49.9%** |
| 64 | 3.9870 | **1.7632** | **+55.7%** |

---

## 📈 Analisi delle Performance (Versione Tiled)

Calcolo dello **Speedup** ($S$) e dell'**Efficienza** ($E$) per la versione Tiled:

| Processi ($np$) | Tempo Tiled (s) | Speedup ($S$) | Efficienza ($E$) |
| :--- | :--- | :--- | :--- |
| 1 | 16.8715 | 1.00x | 100% |
| 4 | 4.2206 | 3.99x | 99.8% |
| 8 | 2.1358 | 7.90x | 98.7% |
| **16** | **1.4465** | **11.66x** | **72.8%** |
| 32 | 1.6060 | 10.50x | 32.8% |
| 64 | 1.7632 | 9.56x | 14.9% |



---

## 🔍 Perché la versione Tiled è molto più veloce?

Il miglioramento drastico (quasi il 50% di tempo in meno in quasi tutti i test) è dovuto principalmente a due fattori tecnici fondamentali presenti nelle slide del corso:

### 1. Cache Locality (Località dei dati)
Nella versione **Naive**, la CPU carica una riga di $A$ e cerca di moltiplicarla per le colonne di $B$. Poiché la matrice è $2000 \times 2000$, la matrice $B$ è troppo grande per stare nella cache L1/L2. La CPU è costretta a "buttare via" dati dalla cache continuamente per far spazio a quelli nuovi (**Cache Thrashing**), dovendo leggere costantemente dalla RAM (che è molto più lenta).

Nella versione **Tiled** ($BS=32$):
* Si lavora su "mattonelle" (tiles) di dati che entrano perfettamente nella cache veloce della CPU.
* I dati caricati vengono riutilizzati molte più volte prima di essere espulsi.
* Il numero di accessi alla memoria RAM diminuisce drasticamente.

### 2. Efficienza Computazionale Locale
Notiamo che anche con **$np=1$**, la versione Tiled è molto più rapida (16.8s contro 25.6s). Questo dimostra che l'ottimizzazione del codice seriale (il "kernel") è il primo passo fondamentale: se ogni processo lavora meglio singolarmente, l'intero sistema parallelo ne beneficia.



---

## 📉 Analisi del Degradamento (np > 16)

Anche la versione Tiled, nonostante l'efficienza maggiore, soffre del rallentamento oltre i 16 processi.

1. **Overhead di Comunicazione Invariato:** Il tempo speso in `MPI_Bcast` e `MPI_Scatter` è identico tra le due versioni, poiché la quantità di dati scambiati è la stessa. Quando il calcolo diventa troppo veloce (1.4 secondi), il peso della comunicazione diventa proporzionalmente enorme.
2. **Scalabilità Forte:** Poiché il tempo di calcolo locale è diminuito grazie al tiling, il "collo di bottiglia" della comunicazione viene raggiunto ancora prima rispetto alla versione naive.
3. **Hardware Limits:** Il fatto che il tempo aumenti passando da 16 a 32 e 64 suggerisce che l'architettura fisica non supporti correttamente l'esecuzione di così tanti processi simultanei (probabilmente a causa del numero di core o della banda passante della memoria condivisa).



---

## 💡 Conclusione del confronto

Il passaggio alla versione Tiled ha portato a un miglioramento dell'efficienza a 16 processi dal **65% al 72.8%**. 

La lezione principale di questo benchmark è che **il calcolo parallelo non sostituisce l'ottimizzazione del codice**: una versione seriale efficiente (tiled) unita a una buona distribuzione MPI è la chiave per ottenere performance reali in HPC. 

Per l'esame: sottolinea come il tiling riduca i **cache misses**, permettendo alle unità di calcolo di non rimanere "ozio" in attesa dei dati dalla memoria centrale.
