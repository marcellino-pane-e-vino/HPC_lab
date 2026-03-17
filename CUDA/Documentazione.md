# La Storia di Due Codici: Come abbiamo spinto la GPU oltre i suoi limiti

Immagina di avere una missione titanica: prendere due griglie di numeri gigantesche, ciascuna formata da 15.000 righe e 15.000 colonne, e moltiplicarle tra loro. Stiamo parlando di eseguire oltre 3 miliardi e mezzo di operazioni matematiche nel minor tempo possibile. 

Per farlo, abbiamo a disposizione un motore fuoriserie: una GPU NVIDIA T4. Abbiamo testato due "assetti" diversi per questo motore. Il primo è un'ottimizzazione fatta a mano da noi, pezzo per pezzo; il secondo usa il software ufficiale della casa madre. 

Ecco il racconto di cosa è successo dentro il silicio della scheda video durante questi due test.

---

## Atto Primo: L'approccio "Artigianale" (Il nostro Tiling)

Nel primo esperimento, abbiamo scritto il codice dicendo esattamente a ogni singolo "operaio" (i thread della GPU) cosa fare. 

Sapevamo fin dall'inizio che far viaggiare gli operai continuamente verso il magazzino principale (la memoria globale della scheda, che è enorme ma lenta) sarebbe stato un disastro. Quindi abbiamo usato una tecnica chiamata **Tiling**. 


Abbiamo diviso l'immenso lavoro in piccole "piastrelle" da 32x32 numeri. I thread lavorano in squadra: vanno al magazzino, prendono una piastrella, la portano su una scrivania super-veloce che hanno in condivisione (la *Shared Memory*) e lì fanno tutti i calcoli a una velocità sbalorditiva. Finita una piastrella, passano alla successiva.

**I risultati sul campo:**
Il cronometro si è fermato a **7,33 secondi**. 
Guardando i log del profiler, abbiamo scoperto che la scheda video ha passato oltre il 92% di questo tempo a fare pura matematica, tenendo i processori costantemente sotto sforzo. 
E il tempo perso a spostare i dati dalla RAM del computer alla memoria della scheda video tramite il cavo della scheda madre? Ci ha messo circa 600 millisecondi. Un tempo ridicolo (appena l'8% del totale) rispetto alla mole di calcoli. In gergo tecnico, diciamo che eravamo in una situazione **"Compute-Bound"**: il nostro unico limite era quanto velocemente i chip riuscissero a moltiplicare.

---

## Atto Secondo: L'Artiglieria Pesante (cuBLAS)

Per il secondo esperimento, abbiamo buttato via la gestione manuale di operai, piastrelle e scrivanie. Abbiamo chiamato in causa **cuBLAS**, la libreria matematica scritta in linguaggio macchina direttamente dagli ingegneri che hanno costruito la GPU. 

Qui è successa una vera e propria magia nera dell'informatica. Il profiler ci ha svelato che cuBLAS non si è limitato a eseguire un calcolo: prima di partire, ha interrogato la nostra scheda video per capire esattamente quanti processori avesse a disposizione. Scoprendo di trovarsi su un'architettura avanzata, ha usato delle istruzioni segrete (chiamate `volta_sgemm_128x128_nn`) in grado di manipolare piastrelle immense, da 128x128 numeri, tenendo accesi i circuiti senza sprecare nemmeno un millesimo di secondo.

**I risultati sul campo:**
Il tempo di calcolo è letteralmente crollato a **1,35 secondi**. 
Abbiamo accelerato il programma di oltre 5 volte, semplicemente chiedendo al software NVIDIA di prendere il volante. Un risultato da 5 trilioni di operazioni al secondo.

---

## Il Gran Finale: Lo schianto contro il "Muro della Memoria"

È mettendo a confronto questi due test che emerge la lezione più affascinante di tutta l'architettura dei computer. 

Ricordi quei 600 millisecondi che servivano per far viaggiare i dati iniziali dal computer alla scheda video attraverso il cavo PCI-Express? 
La fisica non fa sconti a nessuno: che tu usi il codice artigianale o cuBLAS, i byte pesano sempre uguale e il cavo è sempre lo stesso. Quei 600 millisecondi sono rimasti invariati.

Ma guarda come cambia la prospettiva:
Nel primo test, 600 millisecondi di viaggio su 7,3 secondi di calcolo erano una bazzecola (l'8%). 
Nel secondo test, la GPU ha polverizzato i calcoli in appena 1,35 secondi. All'improvviso, quei 600 millisecondi di viaggio sono diventati quasi il **30% del tempo totale** in cui la scheda è rimasta accesa!

Abbiamo ottimizzato la matematica in modo così estremo da schiantarci contro quello che gli ingegneri chiamano il **"Memory Wall"** (il muro della memoria). La nostra GPU ora è così veloce a fare le moltiplicazioni che passa un terzo del suo tempo a girarsi i pollici, aspettando che il cavo della scheda madre finisca di inviarle i numeri su cui lavorare. 

Siamo passati da un problema in cui eravamo lenti a calcolare, a un problema in cui siamo limitati esclusivamente dalla velocità dei cavi di trasmissione. Abbiamo raggiunto il vero limite fisico della macchina.
