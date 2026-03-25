
# Perché il codice MPI crasha? I 3 colpevoli principali

Quando un programma MPI si blocca improvvisamente (spesso con errori come **Segmentation Fault** o **Killed**) per specifiche combinazioni di numero di processi ($P$) e dimensioni della matrice ($N$), i problemi si riducono quasi sempre a memoria o logica di divisione. 

Ecco i tre motivi fondamentali:

## 1. Out of Memory (Esaurimento della RAM) 💥
Se il codice è implementato in modo *naive* (senza ottimizzazioni sulla memoria distribuita), capita che **ogni singolo processo allochi l'intera matrice** invece della sola porzione necessaria.
* **Il Calcolo:** Una matrice 20000x20000 di tipo `double` pesa circa 3.2 GB. Avendo bisogno di 3 matrici (A, B e C per il risultato), ogni processo occupa quasi **9.6 GB**.
* **Il Crash:** Se lanci $P = 24$ processi, il sistema richiederà improvvisamente **~230 GB di RAM**. Il sistema operativo (OOM Killer) interviene e "uccide" l'applicazione per proteggere il computer.

## 2. Segmentation Fault (Il problema dell'indivisibilità) ✂️
Per lavorare in parallelo, si divide il numero di righe/colonne $N$ per il numero di processi $P$. Ma se $N$ non è un multiplo esatto di $P$?
* **Il Calcolo:** Se $N = 10000$ e $P = 24$, la divisione $10000 / 24$ fa $416.66$ (non intero).
* **Il Crash:** Se il codice non ha una logica apposita per gestire il "resto" matematico, gli ultimi processi finiranno per calcolare indici errati, cercando di leggere o scrivere celle di memoria **fuori dai bordi dell'array**. Il processore rileva l'accesso illegale e genera un blocco immediato (SegFault).

## 3. Integer Overflow (Il tetto dei 32-bit) 🔢
Quando $N$ diventa gigantesco, l'indice lineare dell'array supera lo spazio fisico del contenitore matematico.
* **Il Calcolo:** Nei cicli `for`, l'indice in C è solitamente un tipo `int` a 32-bit, il cui valore massimo è circa **2.14 miliardi**.
* **Il Crash:** Se l'indice di una cella arriva, ad esempio, a 2.5 miliardi, la variabile "sfora" e diventa un numero negativo. Il programma tenta di accedere a una cella con indice negativo (es. `-350000`) causando un crash. Questo limite strutturale dei 2.14 miliardi affligge anche le variabili di conteggio delle funzioni standard di invio dati MPI.

---

### 💡 Regola d'oro per il Debug:
* Crasha solo quando **$N$ è molto grande** e aumenti **$P$**? ➡️ Quasi certamente è il **Caso 1 (RAM)**.
* Crasha su matrici medie ma con un **$P$ "dispari" o sbilanciato** (come 24)? ➡️ Quasi certamente è il **Caso 2 (Resto)**.
