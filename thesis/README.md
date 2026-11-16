# Figure ufficiali per la tesi — AFA tokenizer

[Galleria Markdown](GALLERY.md) · [Galleria HTML](figures/index.html) · [Didascalie in inglese](figures/CAPTIONS.md) · [Scarica PNG e PDF](thesis_figures.zip)

Questa raccolta contiene **4 candidate per il testo principale e 8 tavole di appendice**, ciascuna in PDF e PNG a 300 dpi. Le figure sono generate dai tensori reali del checkpoint early-BERT finale, senza training. I PDF incorporano testo e annotazioni vettoriali; RGB e heatmap rimangono raster.

## Selezione consigliata per il testo principale

| Priorità | Figura | Messaggio |
|---|---|---|
| 1 | [M01 — Target selection](figures/main/M01_target_selection.pdf) | Dimostrazione del peluche e rollout che sposta il cubo: contesto frontale e stessa query S06 del polso nel tempo |
| 2 | [M02 — Cube slot evolution](figures/main/M02_cube_slot_evolution.pdf) | Stesse query S05, S11 e S15 lungo avvicinamento, presa, trasporto e posa del cubo |
| 3, se si discutono i limiti | [M03 — Pooling diversity](figures/main/M03_pooling_diversity.pdf) | Una singola osservazione con slot concentrati, quasi uniformi e sovrapposti |
| Alternativa temporale | [M04 — Tool slot evolution](figures/main/M04_tool_slot_evolution.pdf) | Query S06, S09 e S13 del polso durante uso del peg; alternativa a M02 |

Non è necessario inserire tutte e quattro: suggerisco **M01 + M02**, aggiungendo **M03** se c'è spazio per discutere ridondanza e diffusione. M04 può sostituire M02 se il capitolo privilegia il task con lo strumento.

## Appendice

| Figura | Contenuto |
|---|---|
| [A01](figures/appendix/A01_cube_spatial_language.pdf) | Cubo: quattro slot frontali a tre istanti, affiancati all'attenzione delle stesse query verso i token BERT |
| [A02](figures/appendix/A02_failure_spatial_language.pdf) | Rollout con oggetto errato: quattro slot del polso e rispettive distribuzioni query→parole |
| [A03](figures/appendix/A03_tool_spatial_language.pdf) | Uso del peg: corrispondenza fra mappe spaziali e attenzione linguistica |
| [A04](figures/appendix/A04_all_query_word_maps.pdf) | Tutte le 16 query, per tre task e due camere; stessa scala di probabilità |
| [A05](figures/appendix/A05_all_slots_cube_front.pdf) | Tutti i 16 slot frontali dell'osservazione usata per M03 |
| [A06](figures/appendix/A06_all_slots_failure_wrist.pdf) | Tutti i 16 slot del polso durante la presa del cubo nel rollout del peluche |
| [A07](figures/appendix/A07_tool_failure_evolution.pdf) | Evoluzione temporale degli slot in un rollout fallito del task con il peg |
| [A08](figures/appendix/A08_demo_spatial_language.pdf) | Dimostrazione del peluche: stesse query e stessa istruzione di A02, con immagini differenti |

Per un'appendice breve sceglierei **A02, A04 e A05**; A08 è particolarmente utile se si vuole confrontare lo stesso condizionamento linguistico fra dimostrazione e fallimento.

## Come sono stati scelti i frame e gli slot

Sono state esaminate contact sheet RGB dei 12 episodi della precedente analisi: dieci istanti regolarmente distribuiti fra il 5% e il 95% della durata. Da queste sono stati scelti i frame che rendono visibili interazioni e cambiamenti di scena. La nuova raccolta comprende **21 osservazioni di sei episodi, con entrambe le camere: 42 viste**.

Si tratta di una selezione illustrativa e dichiarata, non di un campione casuale o di un confronto controllato degli esiti. Gli indici esatti sono in [selection.json](selection.json). Le descrizioni delle fasi sono note di lettura delle immagini e non annotazioni di ground truth. Non si impone che colonne di episodi diversi abbiano lo stesso stato iniziale o la stessa dinamica.

Gli ID degli slot sono fissati per l'intera sequenza. Non si cambia query per inseguire un oggetto fra un frame e l'altro e non si applica alcun matching temporale. I sottoinsiemi sono stati scelti per esporre comportamenti spaziali interpretabili, diffusione e sovrapposizioni; A04–A06 forniscono anche viste complete senza selezione delle query. Nessuna query viene rinominata “cubo”, “peluche” o “peg”.

## Convenzioni grafiche

Tutte le heatmap spaziali visualizzano `clip(log2(max(N*p, 1)), 0, 3)` con **N=256**, colore magma e opacità `0.62 * valore / 3`. La scala comune indica 1×, 2×, 4× e almeno 8× l'attenzione uniforme. Non si aumenta separatamente il contrasto di una mappa diffusa. I puntini sono i baricentri delle mappe, riportati nei centri delle patch. `H` è l'entropia spaziale divisa per `log(256)`.

Il preprocessing e l'immagine 256×256 sono preservati, incluse le bande nere di padding. Queste patch partecipano sia al modello sia alle statistiche: una mappa diffusa o localizzata sul padding non viene nascosta con un crop. [slot_metrics.csv](figures/slot_metrics.csv) riporta anche la massa sul padding per ogni slot e frame.

Nelle tavole spaziale–linguistiche le righe corrispondono agli stessi ID degli slot. Il pannello di destra conserva **tutti i token dell'istruzione**, inclusi `[CLS]` e `[SEP]`, e mostra probabilità assolute mediate sulle 12 head, su scala comune 0–0.5. Il limite superiore è stato verificato contro tutti i valori: nessuna probabilità osservata viene saturata. I numeri nelle celle sono arrotondati a due decimali; `0.00` può corrispondere a un valore positivo piccolo.

La distribuzione early-BERT è calcolata sulle query apprese e sul testo prima del pooling dell'immagine. In questa esecuzione la differenza massima tra frame della stessa istruzione e camera è **0.0**, anche fra dimostrazione del peluche e relativo fallimento. Per questo la matrice linguistica viene mostrata una volta, accanto alle mappe spaziali che cambiano. Una query che attribuisce peso alla parola “plush” non è per definizione una maschera del peluche.

I colori dei puntini identificano le query, non classi semantiche. I pooler delle camere sono separati: lo stesso ID fra frontale e polso non implica la stessa funzione. La selezione dei frame e l'attenzione, da sole, non stabiliscono la causa di un fallimento. Le dimostrazioni appartengono al dataset di training e non rappresentano successi closed-loop misurati in questa analisi.

## Riproduzione e verifica

- [analysis/manifest.json](analysis/manifest.json): checkpoint, revisioni dei dataset e osservazioni effettive.
- [analysis/arrays/](analysis/arrays/): tensori delle 21 osservazioni, inclusi i pesi di ogni singola head linguistica.
- [analysis/validation.json](analysis/validation.json): controllo della finitezza, del routing, della geometria e della sincronizzazione.
- [figures/figure_manifest.json](figures/figure_manifest.json): frame e slot di ogni figura, didascalie, scala e hash del renderer.
- [figures/validation.json](figures/validation.json): controlli degli export e dei collegamenti.
- [latex_figures.tex](latex_figures.tex): esempi di inclusione delle tre figure principali in LaTeX.

Nella workspace originale:

```bash
AFA_PYTHON=/home/gpepe/miniconda3/envs/lerobot/bin/python

"$AFA_PYTHON" SOT_Gianni/gap_policy/scripts/tokenizer_qualitative.py \
  --checkpoint /data2/gpepe/lerobot_runs/turbovla_afa_early_bert_multitask_full_80k_20261110/checkpoints/160000/pretrained_model \
  --failure-root exports/tokenizer_qualitative/datasets \
  --success-root /data2/gpepe/datasets/Multi-task-verified-9bd90be \
  --selection-manifest exports/tokenizer_qualitative/thesis/selection.json \
  --output exports/tokenizer_qualitative/thesis/analysis --device cpu --threads 4

MPLCONFIGDIR=/tmp/afa-matplotlib OPENBLAS_NUM_THREADS=1 \
"$AFA_PYTHON" SOT_Gianni/gap_policy/scripts/plot_thesis_tokenizer.py \
  exports/tokenizer_qualitative/thesis --dpi 300
```

Nel repository GitHub gli script si trovano sotto `scripts/`. Il renderer legge solo i tensori salvati; non richiede inferenza o training. I percorsi dei dataset nel manifest si riferiscono alla workspace originale e vanno aggiornati se si trasferiscono i dati su un'altra macchina, preservando task, episodi e indici dei frame.
