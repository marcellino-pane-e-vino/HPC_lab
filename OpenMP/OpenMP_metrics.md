## Tabella dei Risultati

| Compilatore | Ottimizzazione | Flag Architettura | Thread (OMP_NUM_THREADS) | Tempo (Secondi) |
| :--- | :--- | :--- | :---: | :--- |
| **ICX** (Intel) | Nessuna (Default) | `-xHost` | *Tutti i core logici* | 0.643 s |
| **ICX** (Intel) | `-O3` | `-xHost` | *Tutti i core logici* | **0.662 s** |
| **ICX** (Intel) | `-O3` | `-xHost` | **1** | 4.786 s |
| **ICX** (Intel) | `-O3` | `-xHost` | **8** | 0.673 s |
| **ICX** (Intel) | `-O3` | `-xHost` | **24** | **0.627 s** |

## Analisi e Conclusioni

1. **Impatto delle Ottimizzazioni del Compilatore:**
   Il passaggio da un'assenza di ottimizzazioni (`-O0`) al livello massimo (`-O3`) nel compilatore GCC ha portato a una riduzione del tempo di calcolo da 17.3s a 5.6s. Il compilatore applica tecniche di loop unrolling e riordino delle istruzioni, riducendo le penalità legate ai *cache miss* e facilitando il flusso dati verso i core.

2. **Vettorializzazione SIMD (Il vantaggio Intel):**
   Il compilatore Intel `icx`, accoppiato al flag `-xHost` (che istruisce il compilatore a generare codice per l'architettura host specifica), mostra prestazioni sbalorditive, riducendo il tempo a circa 0.6 secondi. Questo drastico calo evidenzia l'eccezionale capacità di `icx` di effettuare l'auto-vettorializzazione del ciclo più interno, convertendo le singole operazioni scalari in istruzioni SIMD (es. AVX/AVX2). L'hardware processa più dati contemporaneamente all'interno di ogni singolo ciclo di clock.

3. **Scalabilità OpenMP e Speedup:**
   Analizzando i run effettuati con ICX variando il numero di thread:
   * Con 1 thread il tempo è di 4.786 s.
   * Con 8 thread il tempo scende a 0.673 s, registrando uno *speedup* eccellente (circa 7.1x), dimostrando una scalabilità quasi lineare grazie alla corretta mitigazione del *False Sharing* e all'uso dello scheduling statico che abbassa drasticamente l'overhead di OpenMP a runtime.
   * Aumentando ulteriormente i thread a 24, il guadagno diventa marginale (0.627 s). Questo appiattimento della curva delle prestazioni indica il raggiungimento del "Memory Wall": la capacità di calcolo dei processori supera la larghezza di banda massima del bus di memoria RAM.
