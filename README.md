Che metriche mettiamo??

In un report accademico o professionale che documenti la parallelizzazione o l'ottimizzazione di un programma, è fondamentale presentare una combinazione di metriche di tempo, efficienza e utilizzo delle risorse hardware per giustificare le scelte implementative e analizzare il comportamento del codice.
Di seguito sono elencate le metriche e i calcoli di confronto principali estratti dalle fonti:
1. Metriche Core di Performance Parallela
Queste metriche definiscono quanto bene il programma scala all'aumentare delle risorse computazionali (p processori/core):

    - Speedup (Aumento di velocità):Calcolo: S(n,p)=T(n,p)T(n,1)​
    - Efficienza: Misura la frazione di tempo in cui le risorse sono utilizzate produttivamente
    - Calcolo: E(n,p)=pS(n,p)​
    - Analisi della Scalabilità (strong vs weak scaling)
    - Metriche di Throughput e Calcolo Astratte (GFLOPS / TFLOPS)
    - Intensità Aritmetica: Il rapporto tra operazioni in virgola mobile eseguite e byte trasferiti dalla memoria
    - Roofline
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



--------------
Intel Advisor: 

1- Trovare percorso a intel advisor:

find /opt/intel -name "advixe-vars.sh" 2>/dev/null dovrebbe dare come output: 
/opt/intel/oneapi/advisor/2023.2.0/advixe-vars.sh

2- icx -g -qopenmp -xHost -O3 -fma -lm omp_matrixmult_tiling_hardcoded.c -o omp_matrixmult_tiling_hardcoded

3- advixe-cl --collect survey --project-dir ./progetto_advisor -- ./omp_matrixmult_tiling_hardcoded

4- advixe-cl --collect tripcounts -flops --project-dir ./progetto_advisor -- ./omp_matrixmult_tiling_hardcoded

5- advixe-cl --collect roofline --project-dir ./progetto_advisor -- ./omp_matrixmult_tiling_hardcoded

6- advixe-cl --report roofline --project-dir ./progetto_advisor --report-output=./report_finale.html

poi aprire con Google =)


