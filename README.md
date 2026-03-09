Che metriche mettiamo??

In un report accademico o professionale che documenti la parallelizzazione o l'ottimizzazione di un programma, è fondamentale presentare una combinazione di metriche di tempo, efficienza e utilizzo delle risorse hardware per giustificare le scelte implementative e analizzare il comportamento del codice.
Di seguito sono elencate le metriche e i calcoli di confronto principali estratti dalle fonti:
1. Metriche Core di Performance Parallela
Queste metriche definiscono quanto bene il programma scala all'aumentare delle risorse computazionali (p processori/core):

    Speedup (Aumento di velocità): È il rapporto tra il tempo di esecuzione sequenziale T(n,1) e il tempo di esecuzione parallelo T(n,p) per un problema di dimensione n

.

    Calcolo: S(n,p)=T(n,p)T(n,1)​

.
Nel report, confronta lo speedup ottenuto con lo speedup lineare ideale (S=p) e discuti eventuali deviazioni dovute a overhead di comunicazione o squilibrio del carico (load imbalance)

    .

Efficienza: Misura la frazione di tempo in cui le risorse sono utilizzate produttivamente
.

    Calcolo: E(n,p)=pS(n,p)​

.
Un'efficienza vicina a 1 (o 100%) indica un ottimo utilizzo dell'hardware

        .

2. Analisi della Scalabilità
È necessario distinguere tra due tipi di test per mostrare la robustezza dell'ottimizzazione
:

    Strong Scaling (Scalabilità forte): Aumenta il numero di processori mantenendo fissa la dimensione totale del problema. L'obiettivo è ridurre il tempo totale di esecuzione

.
Weak Scaling (Scalabilità debole): Aumenta il numero di processori mantenendo fisso il carico di lavoro per ogni processore. L'obiettivo è risolvere problemi più grandi nello stesso lasso di tempo

    .

3. Metriche di Throughput e Calcolo Astratto

    GFLOPS / TFLOPS: Miliardi o trilioni di operazioni in virgola mobile al secondo. Forniscono una misura della velocità assoluta di calcolo

.
Intensità Aritmetica: Il rapporto tra operazioni in virgola mobile eseguite e byte trasferiti dalla memoria
.

    Questa metrica è essenziale per il modello Roofline, che confronta le prestazioni effettive con i limiti teorici del processore (Peak Performance) e della larghezza di banda (Memory Wall)

        .

4. Analisi dell'Efficienza Hardware (Micro-architettura)
Per ottimizzazioni come la vectorization o l'ottimizzazione della cache, dovresti riportare:

    Cache Miss Ratio: La percentuale di accessi alla memoria che non trovano i dati nella cache (L1, L2 o L3)

. Un calo drastico dei miss indica un'ottima località dei dati
.
TLB Miss Rate: Fondamentale quando si lavora con set di dati molto grandi (working set) che eccedono la memoria cache, causando costose traduzioni di indirizzi virtuali
.
CPI (Cycles Per Instruction): Numero medio di cicli di clock per istruzione. Un CPI alto può indicare stalli dovuti all'attesa della memoria o dipendenze tra i dati
.
Utilizzo della Banda (Bandwidth utilization): Byte per ciclo o GB/s trasferiti, confrontati con la capacità massima del Front Side Bus (FSB) o dell'interconnessione

    .

5. Metriche Economiche e di Sostenibilità (HPC Professionale)
In report professionali, è spesso richiesto il calcolo del Total Cost of Ownership (TCO)
:

    Performance/Prezzo: Ad esempio, MFLOPS per dollaro (MFLOPS/US$). Serve a dimostrare che l'ottimizzazione software ha reso l'investimento hardware più redditizio

.
Costi Energetici: Consumo in KWatt e impronta di CO2 del calcolo, poiché i costi energetici nel ciclo di vita di un supercomputer possono eguagliare i costi di acquisto

    .

6. Leggi Teoriche di Confronto

    Legge di Amdahl: Usala per calcolare lo speedup massimo teorico basandoti sulla frazione seriale del codice (s). Se il 10% del codice è seriale, lo speedup non supererà mai 10, indipendentemente dal numero di core

    .

Desideri che ti aiuti a creare una struttura dettagliata per il report, includendo come visualizzare graficamente queste metriche?
