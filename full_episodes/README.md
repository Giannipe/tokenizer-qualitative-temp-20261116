# Tutti i frame: selezione delle immagini pulite

[Galleria](GALLERY.md) · [Didascalie e ordine delle righe](CAPTIONS.md) · [Scarica le figure consigliate](selected_clean_figures.zip)

Questa raccolta sostituisce il campionamento iniziale di pochi frame con l'analisi di **tutti i 4.117 frame dei 12 episodi selezionati**, per entrambe le policy: TurboVLA-AFA e TurboVLA fine-tunate sul dataset reale. Non comprende tutti i 150 episodi del dataset: comprende interamente i sei episodi verificati e i sei fallimenti già scelti.

## Copertura

| Task | Episodi verificati, frame | Episodi falliti, frame |
|---|---|---|
| Cubo | 0: 319; 49: 249 | 0: 216; 1: 316 |
| Peluche | 50: 365; 99: 295 | 0: 309; 1: 471 |
| Peg/cubo | 100: 440; 149: 272 | 0: 582; 1: 283 |

Totali per policy: 1.940 frame verificati + 2.177 frame di fallimenti = **4.117 osservazioni**, ciascuna con due camere. Due policy producono quindi 8.234 predizioni di chunk e 16.468 viste elaborate. Non viene effettuato training e nessuna azione viene eseguita sul robot.

L'inferenza viene eseguita in evaluation e inference mode, FP32 senza AMP/TF32, batch size uno. Le due GPU vengono usate separatamente per i due checkpoint. Immagini e stati attraversano i processor salvati su CPU e sono poi trasferiti sulla GPU. La lettura dei video è sequenziale e rispetta i timestamp e gli offset degli episodi. Gli indici vengono verificati contro l'intervallo completo `0..T-1`.

La diagnostica rimane quella documentata nella [precedente raccolta](../paper_style/README.md): CLS→patch DINO dell'ultimo layer, pooling spaziale AFA, routing, coordinate e cross-attention dell'ultimo ACT. Gli hook vengono confrontati con una predizione senza diagnostiche all'inizio, a metà e alla fine di ogni episodio. Le verifiche non consistono in una seconda esecuzione senza hook di tutti i frame.

## Scelta degli slot dopo l'analisi completa

Per ciascun episodio e camera si esaminano tutte le 16 query su tutti i frame. Si selezionano quattro query per la leggibilità della figura, conservando gli stessi quattro ID in tutte le colonne della sequenza.

Si calcolano:

- concentrazione spaziale: uno meno l'entropia normalizzata sulle 256 patch;
- massa nella parte RGB utile, escludendo le patch del padding solo da questa metrica;
- continuità: uno meno la variazione totale fra distribuzioni di frame adiacenti;
- peso della query nel decoder ACT, condizionato alla camera e limitato per la selezione;
- massa effettiva della mappa prima della rinormalizzazione, per penalizzare query numericamente inattive.

Il punteggio di ciascuna query a ogni frame è:

$$
Q_{tk}=a_{tk}\left[0.42(1-H_{tk})+0.22V_{tk}+0.16C_{tk}+0.20W_{tk}\right],
$$

dove $a$ è la massa effettiva limitata a $[0,1]$, $V$ la massa nel contenuto, $C$ la continuità e $W=\min(16w_k/(2\sum_jw_j),1)$ il termine ACT. Questi coefficienti sono una scelta editoriale esplicita per selezionare esempi, non un indicatore validato di correttezza semantica.

Si media $Q$ sull'intero episodio. Le quattro query vengono scelte progressivamente penalizzando, con peso 0.35, la massima similarità spaziale positiva con quelle già scelte. La similarità è il coseno delle mappe dopo sottrazione del livello uniforme, mediato sui frame. Questo limita la selezione di quattro mappe quasi identiche.

Non viene affermato che gli slot selezionati rappresentino categorie annotate, che siano i soli slot importanti o che le loro mappe causino il successo. I dati di tutti gli slot rimangono disponibili in [all_frame_slot_metrics.csv](all_frame_slot_metrics.csv), inclusi quelli non mostrati.

## Scelta degli otto frame

La selezione dei fotogrammi avviene dopo aver elaborato l'intero episodio. Per ottenere una sequenza distribuita nel tempo, si divide in otto intervalli un avanzamento che combina in parti uguali tempo trascorso e movimento cumulativo dello stato del robot, standardizzato entro l'episodio. In assenza di movimento si usa il tempo.

All'interno di ogni intervallo si favoriscono nitidezza dell'RGB e qualità delle mappe delle quattro query selezionate. La nitidezza è stimata dal Laplaciano dell'immagine in scala di grigi e scalata con quantili entro episodio/camera. Il punteggio del frame combina qualità media delle query (0.65) e nitidezza (0.35), con una piccola penalizzazione della distanza dal centro dell'intervallo e una distanza minima fra fotogrammi successivi.

Gli otto indici esatti, i timestamp, le query e i punteggi sono conservati in [selection.json](selection.json). Gli stessi indici sono utilizzati nel confronto con TurboVLA: non vengono scelti frame diversi per favorire visivamente una policy. La scelta è post-hoc e usa le mappe AFA; per questo non costituisce una valutazione imparziale di superiorità delle attention map fra modelli.

La galleria propone una sequenza verificata per task, scegliendo fra le due dimostrazioni e le due camere in base alla qualità della selezione. Conserva anche le altre dimostrazioni e tutti i fallimenti. Le immagini consigliate sono esempi illustrativi di aggregazione leggibile, non una stima della distribuzione media dei comportamenti.

## Immagini senza nero e senza testo

Il modello continua a ricevere l'immagine 256×256 con il padding del checkpoint. Il ritaglio è applicato **solo al display**, dopo l'inferenza e dopo l'interpolazione delle mappe: vengono eliminate le righe 0–31 e 224–255, ottenendo celle RGB 256×192. Non si alterano feature, coordinate o predizioni e non si rinormalizza l'attenzione dopo il crop. La scala resta riferita alle 256 patch originali.

La cartella `clean/` contiene soltanto le immagini affiancate: nessun titolo, legenda, timestamp, metrica o ID scritto sopra. Fra le celle rimane un sottile spazio bianco. Non si eliminano oggetti o aree scure appartenenti alla scena.

La cartella `minimal_labels/` offre gli stessi pannelli con sole etichette delle righe e timestamp, per riconoscere gli slot durante la revisione. Le spiegazioni sono nel file di didascalie. Entrambe le versioni sono esportate in PNG a 300 dpi e PDF.

Ogni sequenza contiene due tipi di tavola:

| Tipo | Righe, dall'alto al basso |
|---|---|
| `tokens` | RGB; DINO CLS→patch del checkpoint AFA; quattro slot AFA nell'ordine indicato in didascalia |
| `policies` | RGB; DINO CLS TurboVLA; DINO CLS AFA; ACT TurboVLA; ACT AFA proiettata tramite le mappe del tokenizer |

Le mappe usano la stessa scala logaritmica relativa all'uniforme, saturata a 8×, e la stessa opacità. Non si aumenta il contrasto indipendentemente per rendere bella ogni mappa. Il confronto ACT richiede le cautele già documentate: VLI ed early-BERT differiscono e la proiezione AFA non è una salienza causale o una attention rollout completa.

## Archivio e riproducibilità

Le diagnostiche integrali di ogni frame sono conservate localmente in `/data2/gpepe/qualitative_full_episodes_20261116/`, accessibile dalla workspace attraverso `exports/tokenizer_qualitative/full_episodes`. I file per episodio non sono tutti duplicati su GitHub per contenere le dimensioni dell'archivio. La repository contiene le figure, gli array dei frame visualizzati, i punteggi di tutti i frame/slot, la copertura completa e gli hash degli archivi locali.

Gli script sono `scripts/analyze_full_episodes.py` e `scripts/select_clean_full_episodes.py` nella repository; nella workspace si trovano sotto `SOT_Gianni/gap_policy/scripts/`. I checkpoint, il dataset verificato e i relativi encoder devono essere disponibili nei percorsi indicati dal manifest per ripetere l'inferenza. Il rendering può essere ripetuto leggendo gli archivi già prodotti.

I precedenti campioni da 60 e 42 osservazioni rimangono raccolte esplorative separate. Questa versione usa tutti i frame dei 12 episodi selezionati e una selezione editoriale successiva: il nuovo conteggio deve essere usato se queste figure vengono descritte nell'Experimental Setup.

## Comandi usati

L'estrazione richiede l'ambiente LeRobot con i pacchetti locali `lerobot_policy_turbovla`, `lerobot_policy_turbovla_afa` e `lerobot_afa` installati. Gli script della repository delle figure non sostituiscono il codice della policy. Nell'ambiente della workspace si usa `/home/gpepe/miniconda3/envs/lerobot/bin/python`.

```bash
python scripts/analyze_full_episodes.py \
  --manifest analysis/manifest.json \
  --checkpoint /data2/gpepe/lerobot_runs/turbovla_afa_early_bert_multitask_full_80k_20261110/checkpoints/160000/pretrained_model \
  --output /data2/gpepe/qualitative_full_episodes_20261116 \
  --device cuda:0

python scripts/analyze_full_episodes.py \
  --manifest analysis/manifest.json \
  --checkpoint /data2/gpepe/lerobot_runs/turbovla_base_vli_multitask_full_80k_20261111/checkpoints/160000/pretrained_model \
  --output /data2/gpepe/qualitative_full_episodes_20261116 \
  --device cuda:1

python scripts/select_clean_full_episodes.py \
  /data2/gpepe/qualitative_full_episodes_20261116 \
  --output full_episodes
```

I file di copertura e i manifest sono conservati in `provenance/`. [validation.json](validation.json) riporta i controlli sugli array integrali e sulle immagini esportate.

## Verifiche ottenute

Sono stati verificati tutti gli indici `0..T-1` dei 24 archivi episodio/policy e la finitezza di tutti gli array. Gli stati e i timestamp coincidono fra i due modelli; le pipeline di preprocessing salvate nei checkpoint sono identiche. Su tutti i frame AFA, l'errore massimo nella somma del routing competitivo è 4.77 × 10⁻⁷ e quello nella ricostruzione delle coordinate geometriche dalle mappe è 7.36 × 10⁻⁷. Nei 36 controlli per policy, le predizioni con e senza registrazione diagnostica coincidono esattamente.
