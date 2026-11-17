# Cubo: diagnostica integrale delle demo verificate

[Galleria completa](GALLERY.md) · [Didascalie](CAPTIONS.md) · [Selezione](selection.json) · [Verifiche](validation.json)

## Dati e copertura

Si usano soltanto le due dimostrazioni verificate selezionate per questo task: **episodio 0: 319 frame; episodio 49: 249 frame**. I rollout falliti sono esclusi da questa raccolta.

Dataset: [Giannipe/Multi-task-verified](https://huggingface.co/datasets/Giannipe/Multi-task-verified), revisione `9bd90bef6481029a3e00e3527c8f67ac04bc5b85`. Istruzione registrata: **Grab the red cube and place on the gray block**. Sono gli stessi due episodi usati nelle precedenti raccolte, qui elaborati interamente, a 30 Hz, con camera frontale e al polso.

**Tutti i 568 frame sono elaborati da ciascun checkpoint**, per 1136 predizioni di chunk e 2272 viste complessive fra due policy e due camere. La procedura usa lo stesso tokenizer early-BERT finale AFA e lo stesso TurboVLA VLI fine-tuned dell'analisi peg. Entrambi i checkpoint sono a 160000 micro-step; percorsi e SHA-256 sono nei manifest in `provenance/`.

Queste sono **dimostrazioni registrate del dataset di training**, con comportamento verificato, e non nuovi rollout riusciti delle policy. Il confronto rappresenta un replay offline delle osservazioni, degli stati e dell'istruzione originali. Non si esegue training e nessuna azione viene inviata al robot.

Inferenza in `eval()` e `inference_mode()`, FP32, batch uno, senza AMP/TF32. Per ogni frame si verifica che attivare la registrazione diagnostica non modifichi le predizioni; si controllano anche identità di RGB/stati/comandi fra i due replay, normalizzazione delle distribuzioni, geometria, decomposizione dei token e replay del transformer.

## Tensori estratti su ogni frame

- DINO: feature dense 256×768 per camera; CLS→key per tutti i 12 layer e tutte le 12 head, inclusi i pesi su CLS e quattro register token. Le figure spaziali eliminano questi cinque prefissi e normalizzano sulla griglia di 256 patch. Si conservano anche le masse prima della normalizzazione.
- AFA: tutte le 16 mappe di pooling per camera, routing competitivo P(query|patch), coordinate geometriche, valori aggregati 768D, proiezione semantica 256D, proiezione geometrica 256D e token fusi 256D. La relazione **fused = geometry + semantics** viene verificata sui dati.
- Query: vettori appresi prima e dopo il condizionamento early-BERT, attenzione query→parola su tutte le 12 head, padding e token speciali conservati. A parità di istruzione e camera questi pesi possono restare invarianti nel tempo: ciò non dimostra un'attenzione linguistica che segue l'immagine.
- Transformer AFA: token dopo il mixing e matrici self-attention delle quattro head, ottenute con replay separato. Non si modificano i moduli interni del forward produttivo. Si verifica l'accordo fra replay e uscita reale del transformer.
- ACT: cross-attention di tutti e tre i layer, tutte le otto head e tutte le dodici query di azione; chunk denormalizzati 12×6 e azioni/stati registrati.

Per TurboVLA le mappe ACT sono pesi sulle patch dense. Per AFA sono proiezioni dei pesi degli slot attraverso le mappe di pooling. Si condiziona alla camera per la distribuzione spaziale; le masse assolute per camera sono salvate e rappresentate separatamente. Questa proiezione non annulla il successivo mixing dei token e non è una salienza causale o una attention rollout completa. I due checkpoint differiscono anche nell'interazione linguistica VLI/early-BERT: il confronto non isola causalmente il tokenizer.

In entrambi i replay le diagnostiche di tutti i layer DINO usano la matrice QK completa, con la stessa forma del riferimento eager. Il forward produttivo SDPA resta invariato. Gli hash dell’estrattore e del riuso della pipeline sono conservati in `provenance/`.

## Significato delle figure

| Famiglia | Cosa rappresenta |
|---|---|
| `overview` | Nove pannelli: RGB, DINO CLS AFA, ACT AFA, geometria ed ellissi, routing vincente, keypoint colorati dalla semantica, feature DINO in PCA, semantica aggregata proiettata sull'RGB, token fusi proiettati con keypoint. |
| `all_slots/*attention` | Tutte le mappe spaziali dei 16 slot, ID 0–15 per righe. Mostrano quali patch ciascuna query aggrega. |
| `all_slots/*routing_rgb` | Tutte le responsabilità P(query|patch), applicate all'RGB su fondo bianco. Non sono ricostruzioni o segmentazioni annotate. |
| `timelines/*slots*` | Gli stessi ID lungo dodici osservazioni. Quattro tavole coprono tutti i 16 slot; nessun riordinamento nel tempo. |
| `timelines/*policies` | RGB; DINO TurboVLA; DINO AFA; ACT TurboVLA; ACT AFA. Gli stessi frame in tutte le righe. |
| `timelines/*geometry_semantics` | RGB; geometria e dispersione; keypoint colorati dalla semantica aggregata; proiezione della semantica; proiezione del token geometrico+semantico. |
| `timelines/*representation` | RGB; feature DINO TurboVLA; feature DINO AFA; semantica AFA; token fuso; token dopo il transformer. Ogni spazio ha una PCA esplicitamente distinta. |
| `heads_layers` | Layer DINO, head DINO finali, head ACT per ciascun layer e dodici query di azione. L'ordine è per righe, a partire da zero. |
| `query_words` | Pesi reali query→token BERT, mediati sulle head o separati per ciascuna head; tutti i 16 slot e token speciali. Scala assoluta 0–1. |
| `slot_self_attention` | Interazioni fra i 16 token nel transformer AFA, media o singole head. Pesi 0–1. |
| `spatial_similarity` / `semantic_similarity` | Coseno fra mappe spaziali dopo sottrazione dell'uniforme, oppure fra embedding semantici. Ridondanza spaziale e somiglianza semantica sono proprietà distinte. |
| `embedding_paths` | Traiettorie PCA di geometria, semantica e token fuso: colori degli ID fissi, punti ai frame mostrati. Non dimostrano cluster di oggetti. |
| `all_frame_metrics` | Entropia, peso ACT di ciascuno slot, rapporto delle norme geometria/semantica, coordinate e massa ACT per camera, su tutti i frame. |
| `action_replay` | Stato, comandi registrati e prima azione del chunk ricalcolato dalle due policy. Il replay è open-loop sulle osservazioni registrate: la differenza dai comandi non è una misura del successo di una nuova esecuzione. |

Le ellissi descrivono una deviazione standard delle distribuzioni spaziali; non sono extent predetti dal modello. I keypoint usano la vera griglia dei centri delle patch. Quelli nel padding possono scomparire nel crop grafico: non vengono spostati artificialmente. Mappe numericamente inattive e slot ridondanti restano inclusi.

## PCA e colori semantici

Per ciascuna camera si adatta una PCA separata per geometria, semantica proiettata, token fuso e token dopo transformer, usando **tutti gli slot di tutti i 568 frame**, con base e limiti condivisi fra episodi. Le tre componenti principali diventano canali RGB, con segni fissati e limiti ai quantili 2–98%. Questi colori rappresentano variazioni continue degli embedding e non nomi di classi o cluster annotati. Colori di spazi PCA diversi non sono direttamente confrontabili.

Per le feature DINO si usa una base comune ai due checkpoint entro ciascuna camera, calibrata deterministicamente sui frame 0,16,32,… e sull'ultimo frame di ogni episodio, escludendo le righe di padding dalla calibrazione. Questa è una riduzione dimensionale descrittiva in coordinate condivise; non garantisce allineamento semantico dei due encoder. I modelli PCA e la varianza spiegata sono in `pca_models.npz`.

Le proiezioni semantiche sull'RGB combinano il colore di tutti i token con il routing competitivo originale. Il colore sul keypoint invece appartiene alla semantica dello slot in quella posizione. Non si assegnano etichette manuali alle regioni.

## Selezione e stile

L'inferenza riguarda tutti i frame. Solo dopo si scelgono dodici osservazioni cronologiche per camera, una per ciascuno di dodici intervalli temporali uniformi, favorendo nitidezza e leggibilità con il criterio già documentato in [full_episodes](../full_episodes/README.md). Sei di questi frame ricevono anche atlanti di tutti gli slot e maschere RGB; un frame centrale riceve gli atlanti per head/layer/query. Il viewer permette di esaminare i frame intermedi e tutte le query senza questa selezione. Gli indici esatti sono in `selection.json`. Le figure restano illustrative e la selezione post-hoc non è una stima imparziale di superiorità del tokenizer.

Le immagini `clean/` non contengono titoli o testo. Il padding superiore e inferiore di 32 pixel viene rimosso **solo nel rendering**, dopo l'inferenza e l'interpolazione, lasciando celle 256×192. Non si tagliano aree scure della scena e non si rinormalizzano le probabilità dopo il crop. Le attention map condividono la scala log₂ rispetto all'uniforme, da 1× a 8×, e l'opacità massima 0.65. Le maschere RGB usano probabilità assolute. I grafici numerici mantengono gli assi necessari; le versioni `minimal_labels/` aggiungono soltanto nomi delle righe e timestamp. PNG e PDF mantengono la risoluzione nativa delle celle, senza fingere dettaglio spaziale oltre la griglia 16×16.

## Accesso e riproducibilità

Aprire `viewer/index.html` **localmente dopo aver scaricato la raccolta**: non richiede server, training, rete o librerie esterne. Il viewer carica i dati float32 delle mappe e PNG RGB originali di ogni frame; permette di scegliere episodio, camera, query e modalità e scaricare il canvas senza testo. Il viewer è uno strumento di esplorazione con interpolazione del browser; i file PNG/PDF pre-esportati usano il renderer scientifico Python. `panels/` contiene anche ogni pannello separato dei frame selezionati.

L'archivio completo di tutti i tensori per ogni frame rimane in `/data2/gpepe/qualitative_verified_complete_20261117/cube`. GitHub contiene le mappe di tutti i frame necessarie al viewer, figure, PCA, metriche e un sottoinsieme degli archivi completi in `selected_arrays/`. I file per frame completi di entrambi i checkpoint non sono tutti duplicati su GitHub. Gli script sono in `scripts/` nella repository principale; richiedono l'ambiente LeRobot e i pacchetti locali delle policy per ripetere l'estrazione. Il rendering legge gli archivi salvati senza eseguire inferenza.

`slot_metrics.csv` contiene una riga per ogni frame/camera/slot; `frame_metrics.csv` riporta le statistiche per camera. I manifest fissano checkpoint e revisione del dataset. `figure_manifest.json` e `CAPTIONS.md` descrivono l'ordine dei pannelli.

## Coerenza con peg, cubo e peluche

Le raccolte usano la stessa disposizione dei pannelli, gli stessi colori degli ID degli slot, le stesse scale per attention/routing e gli stessi checkpoint. Le PCA sono adattate separatamente per raccolta e camera: i loro colori non sono categorie semantiche e non vanno confrontati direttamente fra task. Tutti i 16 slot restano presenti, anche se diffusi, inattivi o ridondanti.

La proiezione dei token dopo il transformer usa il routing originario come ancoraggio grafico dell'ID; dopo il mixing non descrive tutto il campo ricettivo del token.

## Ripetere l'analisi

Usare `scripts/extract_verified_diagnostics.py` per il replay dei soli indici specificati in `source.json`, `scripts/plot_verified_diagnostics.py` per figure e viewer e `scripts/verify_verified_diagnostics.py` per i controlli completi. L'estrazione richiede l'ambiente LeRobot con i pacchetti locali delle due policy e gli encoder disponibili. Il rendering legge gli archivi salvati senza ulteriore inferenza.


## Esito delle verifiche

Verificati tutti i **1136 archivi frame/policy**. Predizioni con registrazione diagnostica attiva/disattiva: errore massimo **0.0** su tutti i frame. Decomposizione token geometria+semantica: errore massimo **0.0**. Gli altri errori numerici, gli indici completi e gli hash per frame sono in [validation.json](validation.json) e [archive_index.csv](archive_index.csv). Il viewer è collaudato su file locale, incluse tutte le sette modalità, camera al polso, query 15, ultimo frame ed esportazione PNG: [viewer_validation.json](viewer_validation.json).

## Viewer per episodio

[Episodio 0: tutti i 319 frame](viewer_episode_000.zip) · [Episodio 49: tutti i 249 frame](viewer_episode_049.zip)

Estrai ciascuno ZIP in una cartella separata e apri `viewer/index.html`. Non serve avviare un server. La cartella `viewer/` nella repository comprende entrambi gli episodi insieme.
