# Figure qualitative offline di TurboVLA-AFA

Le figure effettivamente generate sono in
`/home/gpepe/ws/exports/tokenizer_qualitative/analysis/figures/`.
Aprire `index.html` per la galleria. Ogni tavola è esportata sia in PNG a
300 dpi sia in PDF con testo incorporato e immagini raster. Non servono training,
robot o simulatore. È stata usata inferenza FP32 su CPU.

## Dati e checkpoint

Checkpoint confermato dall'utente:

```text
/data2/gpepe/lerobot_runs/turbovla_afa_early_bert_multitask_full_80k_20261110/checkpoints/160000/pretrained_model
```

La configurazione effettiva usa DINOv3 ViT-B/16 a 256×256, 16 slot per vista,
pooler separati front/wrist, attention single-head, competitive routing,
fusione additiva geometria/semantica e condizionamento `early_bert`.
Le attention BERT delle query hanno invece 12 head. Non va descritta questa
run come tokenizer multi-head o come variante `vli`.

| Task | Rollout di fallimento | Episodi verificati selezionati |
|---|---|---|
| 1, cubo sul blocco grigio | `Giannipe/rollout_tokenizer_attention_task1_20261116_154605`, ep. 0 e 1 | 0 e 49 |
| 2, peluche nella scatola | `Giannipe/rollout_tokenizer_attention_task2_20261116_155022`, ep. 0 e 1 | 50 e 99 |
| 3, peg per spingere il cubo | `Giannipe/rollout_tokenizer_attention_task3_20261116_155319`, ep. 0 e 1 | 100 e 149 |

Le dimostrazioni provengono da `Giannipe/Multi-task-verified`, copia locale
`/data2/gpepe/datasets/Multi-task-verified-9bd90be`. Sono il primo e l'ultimo
episodio di ciascun task: una selezione deterministica, non basata sulle mappe.
I sei fallimenti sono etichettati secondo l'indicazione dell'utente.
Altri repository di rollout sul profilo non sono inclusi perché non appartengono
ai tre dataset indicati; in particolare `successgrasp` non implica successo
dell'intero task.

Per ciascuno dei 12 episodi si estraggono i frame al 10%, 30%, 50%, 70% e 90%
della durata: 60 osservazioni e 120 viste. Le tavole di dettaglio usano il
midpoint di ciascun episodio; le sequenze temporali comprendono tutti e cinque
gli istanti. Le percentuali temporali **non allineano le fasi del gesto**.
Per uno studio puntuale del contatto o dell'istante del fallimento si può usare
`--fractions` con altri valori in una nuova cartella di analisi.

`manifest.json` conserva percorsi, revisioni complete dei dataset, metadati dei
frame, istruzioni, configurazione e SHA-256 dei pesi. `arrays/*.npz` contiene
RGB dopo preprocessing, mappe, routing, keypoint, semantica aggregata,
token fusi, attenzione linguistica, attenzione dell'ultimo layer ACT e azioni
predette. Non vengono alterati checkpoint o dataset originali.

## Che cosa mostrano le tavole

* `comparisons/`: i due fallimenti e le due dimostrazioni per ogni task e vista,
  con geometria, semantica e proiezione dell'attenzione delle azioni.
* `overview/`: RGB realmente fornito alla policy; media e dispersione spaziale
  degli slot; assegnazione alle query; canvas semantico; entropia del routing;
  proiezione dell'attenzione ACT sulle mappe degli slot.
* `slots/`: tutti i 16 slot, senza selezionare solo quelli più interpretabili;
  overlay dell'attenzione e entropia normalizzata per ciascuno.
* `diagnostics/`: similarità fra mappe, similarità fra embedding semantici e
  attenzione query→parole del condizionamento early-BERT.
* `timelines/`: evoluzione delle posizioni degli slot e della proiezione ACT.
* `embeddings/`: PCA bidimensionale dei token realmente emessi dal tokenizer,
  dopo la fusione geometria/semantica e prima del transformer AFA.

## Convenzioni e limiti di interpretazione

**Input e coordinate.** Si usano i processor salvati nel checkpoint, incluso
resize con padding e normalizzazione. L'RGB visualizzato è ricostruito invertendo
la normalizzazione del tensore effettivamente fornito alla policy. Il padding
nero rimane visibile. La lettura video segue i PTS con gli offset per episodio
del formato LeRobot v3 e controlla la sincronizzazione entro un frame.
Le coordinate del tokenizer, definite sui punti di griglia da −1 a 1, vengono
convertite nei centri delle patch per la sovrapposizione sul raster RGB.
Le ellissi rappresentano la covarianza della distribuzione spaziale dell'attenzione
con semiassi a una deviazione standard: non sono bounding box o intervalli di
confidenza e non vengono fornite al modello come feature di extent.

**Heatmap.** Se `a_k` è l'attenzione spaziale normalizzata dello slot su `N`
patch, il valore mostrato è `clip(log2(max(N*a_k, 1)), 0, 3)`: 0 equivale
al livello uniforme, 1 a 2×, 2 a 4× e 3 ad almeno 8×. Anche l'opacità cresce
con questo valore. La scala è identica per tutti gli slot e gli episodi,
senza normalizzazione min/max indipendente. Le mappe sono interpolate per la
visualizzazione: la risoluzione nativa resta 16×16, non una segmentazione fine.
I valori sotto il livello uniforme non ricevono colore; questo va mantenuto
esplicito nelle didascalie.

**Routing e slot.** Per questo checkpoint si usa il vero `P(query | patch)`
salvato dal competitive routing. Il colore identifica la query vincente e
l'opacità riflette la sua probabilità rispetto al livello uniforme `1/K`.
L'entropia del routing è divisa per `log K`. Un'assegnazione netta indica
competizione fra slot; non prova che ogni slot rappresenti un oggetto distinto.
Slot diversi possono localizzarsi nella stessa regione. Gli ID/colori sono
stabili nella stessa vista; pooler separati non garantiscono corrispondenza
fra S00 frontale e S00 polso.

**Geometria + semantica.** Il canvas usa gli embedding effettivamente aggregati
dal tokenizer e trasformati dal suo `semantic_encoder`. La PCA è stimata una
volta su tutti gli slot dei due gruppi, separatamente per ciascuna camera.
I primi tre componenti determinano i colori, scalati con i quantili 2–98%
condivisi; le mappe di routing riportano questi colori sulle patch RGB.
Questi sono colori di similarità negli embedding, non categorie semantiche
annotate, nomi di oggetti o una ricostruzione RGB. Non confrontare colori PCA
fra camere diverse. `semantic_pca.npz` salva la trasformazione usata.

**Spazio dei token.** La PCA 2D dei token fusi è anch'essa condivisa fra tutti
i frame della stessa camera; sono riportate le frazioni di varianza spiegata.
La separazione può riflettere l'identità delle query apprese. Non è una prova
automatica di clustering per istanza o semantica. Nessun algoritmo impone
artificialmente 16 cluster ai dati.

**Similarità.** La similarità spaziale è il coseno dopo sottrazione del livello
uniforme; evita che un fondo piatto comune sembri un picco condiviso. La
similarità semantica è il coseno fra i vettori aggregati dal tokenizer.
Queste due proprietà non sono equivalenti.

**Attenzione della policy.** Si registra l'attenzione di cross-attention
dell'ultimo layer ACT, mediando head e 12 query di azione. La massa assegnata
alla vista viene riportata separatamente. Per il canvas si rinormalizzano i
pesi sulla vista e si combinano le mappe spaziali degli slot. La scala è lo
stesso lift sul livello uniforme. È una proiezione diagnostica: prima di ACT
il transformer AFA ha già mescolato informazione fra slot. Non è una attention
rollout completa, una mappa di salienza causale o una misura dell'importanza
dell'oggetto per il successo. L'attenzione early-BERT mostrata nelle matrici
è invece la media sulle head delle query verso i token di testo, inclusi
`[CLS]` e `[SEP]`.

**Successi e fallimenti.** Le dimostrazioni verificate contengono movimenti
corretti ma sono osservazioni del dataset usato per il training, non rollout
riusciti della policy o un test held-out. Le azioni vengono ricalcolate offline
sullo stato registrato, senza esecuzione del robot. Il confronto è descrittivo;
due episodi per task non permettono conclusioni statistiche generali o causali.

## Rigenerazione

Nell'ambiente locale i pacchetti LeRobot, TurboVLA, AFA e TurboVLA-AFA sono già
disponibili. Dipendenze di analisi: PyTorch, PyArrow, PyAV, NumPy, Matplotlib,
Pillow; `huggingface_hub` serve solo per acquisire i dati.

```bash
cd /home/gpepe/ws
PYTHON=/home/gpepe/miniconda3/envs/lerobot/bin/python

"$PYTHON" SOT_Gianni/gap_policy/scripts/tokenizer_qualitative.py \
  --checkpoint /data2/gpepe/lerobot_runs/turbovla_afa_early_bert_multitask_full_80k_20261110/checkpoints/160000/pretrained_model \
  --failure-root exports/tokenizer_qualitative/datasets \
  --success-root /data2/gpepe/datasets/Multi-task-verified-9bd90be \
  --output exports/tokenizer_qualitative/analysis \
  --device cpu --threads 4

MPLCONFIGDIR=/tmp/afa-matplotlib OPENBLAS_NUM_THREADS=1 \
"$PYTHON" SOT_Gianni/gap_policy/scripts/plot_tokenizer_qualitative.py \
  exports/tokenizer_qualitative/analysis --dpi 300
```

Per CUDA si può scegliere `--device cuda` e una nuova cartella di analisi.
Il renderer legge solo gli array salvati, senza caricare il modello. Accetta
`--sections comparisons overview slots diagnostics timelines embeddings` per
rigenerare solo le famiglie desiderate. L'estrazione riparte dai file mancanti
e rifiuta cartelle con checkpoint o selezione di frame differenti.

## Didascalie suggerite

**Confronto:** “Offline visualization of the AFA tokenizer on failed policy
rollouts and verified demonstrations. Rows show both recorded episodes per
condition for the same task at their temporal midpoint. Columns show spatial
slot means and covariance ellipses, pooled semantic features projected to RGB
with a shared per-camera PCA, and a diagnostic projection of the final ACT
cross-attention onto slot maps. Demonstrations provide successful recorded
behavior and do not constitute closed-loop policy successes.”

**Slot:** “Spatial attention of all 16 AFA slots for a recorded observation.
Heatmaps use a common logarithmic scale relative to uniform spatial attention,
clipped between 1× and 8×, over the policy's preprocessed RGB input. Markers
denote attention-weighted patch coordinates; normalized entropy quantifies
spatial concentration. Coincident maps reveal spatial redundancy rather
than distinct object instances.”

**Embedding:** “Two-dimensional PCA of the fused AFA tokenizer outputs before
the AFA transformer. Colors identify learned query slots within one camera;
markers distinguish failed rollouts from verified demonstrations. All tasks
and conditions share the same projection. Separation of learned slot
representations does not establish object-instance clustering.”
