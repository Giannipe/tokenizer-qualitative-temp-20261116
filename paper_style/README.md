# Figure qualitative in stile paper: DINO, AFA e confronto fra policy

[Galleria e didascalie](figures/GALLERY.md) · [Galleria HTML](figures/index.html) · [Verifiche](validation.json)

Questa raccolta traduce lo stile grafico di AFA, TokenLearner e Slot Attention in diagnostiche effettive del nostro checkpoint. Contiene **due candidate per il testo principale e sette tavole di appendice**, in PNG a 300 dpi e PDF. Tutte le inferenze sono offline, senza training, su osservazioni di dimostrazioni verificate.

## Le due figure per il testo principale

| Figura | Che cosa mostra | Messaggio sostenibile |
|---|---|---|
| [M01 — DINO→AFA](figures/main/M01_dino_to_afa.pdf) | Tre task, due osservazioni ciascuno; RGB, attention CLS del backbone e quattro mappe AFA dello stesso checkpoint | Il backbone fornisce feature contestualizzate; diverse query AFA le aggregano con distribuzioni spaziali differenti, incluse query diffuse |
| [M02 — Token nel tempo](figures/main/M02_token_attention_through_time.pdf) | Quattro frame dell'uso del peg, camera al polso; RGB e quattro query mantenute fisse nel tempo | Le mappe cambiano con l'osservazione; la stessa identità di query non garantisce associazione stabile a un oggetto |

M01 è una figura verticale, leggibile a larghezza di pagina. M02 riprende la struttura temporale di TokenLearner. Le immagini includono il padding effettivo e le mappe condividono una sola scala. I colori degli ID delle query non rappresentano categorie di oggetti.

## Appendice e materiale di supporto

| Figura | Contenuto |
|---|---|
| [A01](figures/appendix/A01_dino_to_afa_wrist.pdf) | Stessa struttura DINO→AFA, camera al polso |
| [A02](figures/appendix/A02_dino_to_afa_second_episodes.pdf) | DINO→AFA nei secondi episodi selezionati dei tre task |
| [A03](figures/appendix/A03_cube_temporal_attention.pdf) | Evoluzione delle query durante la dimostrazione del cubo |
| [A04](figures/appendix/A04_plush_temporal_attention.pdf) | Evoluzione delle query durante la dimostrazione del peluche |
| [A05](figures/appendix/A05_routing_weighted_rgb.pdf) | RGB, query vincente e RGB pesato dal routing, in una disposizione ispirata a Slot Attention |
| [A06](figures/appendix/A06_policy_attention_comparison.pdf) | Attenzione ACT delle policy TurboVLA e TurboVLA-AFA sugli stessi input |
| [A07](figures/appendix/A07_backbone_attention_comparison.pdf) | Stessa diagnostica CLS→patch nei due backbone fine-tunati |

Per un'appendice breve sceglierei A01, A02 e A07. A05 è utile se si vuole spiegare l'assegnazione spaziale, ma i suoi pannelli individuali sono **RGB pesato dal routing, non ricostruzioni da un decoder**. A06 richiede una discussione esplicita della fusione multimodale; non lo userei da solo per dimostrare la superiorità di AFA. Gli atlanti completi di tutti i 16 slot restano nella [raccolta diagnostica originale](../analysis/figures/index.html) e nella raccolta curata precedente `thesis/`, conservata in locale.

## Campione e selezione

Si usano gli stessi sei episodi verificati già selezionati in `Giannipe/Multi-task-verified`: 0/49 per il cubo, 50/99 per il peluche e 100/149 per il peg. La selezione comprende:

- 30 osservazioni ai tempi relativi 10%, 30%, 50%, 70% e 90% dei sei episodi;
- 12 osservazioni aggiuntive dei tre episodi curati in precedenza tramite ispezione RGB delle interazioni.

In totale: **42 osservazioni e 84 viste per checkpoint**, elaborate sia da TurboVLA-AFA sia da TurboVLA. I frame sono gli stessi nelle due policy. L'RGB preprocessato è stato verificato identico byte per byte nei rispettivi array FP32. Stato e istruzione provengono dalla stessa riga/episodio del dataset.

M01 usa il primo episodio selezionato di ogni task, ai tempi relativi 30% e 70%; A02 usa il secondo. La scelta dei frame precede l'estrazione delle attention CLS. M02 usa i frame 14, 95, 176 e 257 dell'episodio 149; A03 e A04 riusano le selezioni curate del cubo e del peluche. Le etichette delle fasi sono letture illustrative dell'RGB, non annotazioni di ground truth.

Le query mostrate sono fisse, senza riordino per frame: frontale S02/S05/S06/S04, polso S06/S09/S13/S04. La selezione è illustrativa: include pattern concentrati, sovrapposti e diffusi osservati nella precedente analisi, e mantiene S04 anche quando quasi uniforme. Non si tratta di un campionamento casuale o di una selezione automatica di slot semanticamente corretti. Gli stessi numeri delle query nelle due camere non implicano corrispondenza: i pooler sono separati.

## Come otteniamo la diagnostica CLS di DINO

Si intercettano gli input dell'ultimo layer di self-attention DINOv3, prima della sua esecuzione. Sugli stessi hidden state si ricalcolano le proiezioni query e key e si applicano gli stessi positional embedding rotatori, RoPE. Si calcola soltanto la riga di softmax della query CLS verso tutte le chiavi:

$$
B_{hi}=\operatorname{softmax}_{i}\left(q_{h,\mathrm{CLS}}k_{hi}^{\top}/\sqrt{d_h}\right).
$$

Il backend SDPA del normale forward non viene cambiato e nessun output della policy viene sostituito. La diagnostica è stata confrontata con la corrispondente riga del calcolo eager esplicito del layer. Si media sulle 12 head, si escludono le chiavi CLS e i quattro register token e si rinormalizza sulle 256 patch:

$$
p_i^{\mathrm{CLS}}=\frac{\frac{1}{12}\sum_h B_{hi}}{\sum_{j\in\mathrm{patch}}\frac{1}{12}\sum_h B_{hj}},\qquad i\in\mathrm{patch}.
$$

La quota assegnata alle patch prima della rinormalizzazione è riportata come `patch mass`. Le patch di padding rimangono incluse. La mappa ha risoluzione nativa 16×16.

**CLS→patch è una diagnostica del backbone.** Non rappresenta direttamente l'attenzione della policy TurboVLA, che utilizza i token delle patch, né una mappa causale dell'importanza dei pixel.

## Come visualizziamo le mappe AFA

Si usano le vere mappe `latest_attention` del tokenizer, precedenti al transformer AFA. Per la visualizzazione vengono rinormalizzate spazialmente; non si sostituiscono con Grad-CAM, PCA o segmentazioni esterne. I pesi del routing competitivo e la relativa normalizzazione con epsilon sono verificati numericamente.

Per entrambe le tipologie di attention, DINO e AFA, si visualizza la stessa quantità:

$$
L_i=\operatorname{clip}\left(\log_2(\max(Np_i,1)),0,3\right),\quad N=256.
$$

I livelli sulla barra corrispondono a ≤1×, 2×, 4× e ≥8× rispetto all'attenzione uniforme. La mappa è interpolata bilinearmente e sovrapposta all'RGB con opacità $0.65L/3$. Valori sotto il livello uniforme rimangono trasparenti; valori sopra 8× saturano. Non si usa un min/max indipendente per rendere ogni mappa artificialmente contrastata. La scala descrive attenzione relativa alla propria distribuzione spaziale, non probabilità di successo o certezza semantica.

In A05 si usa invece il routing competitivo $R_{ki}=P(\mathrm{query}\ k\mid\mathrm{patch}\ i)$: ciascuna vista individuale è $R_k I+(1-R_k)\,\mathrm{bianco}$. Questi pannelli non attraversano un decoder e non dimostrano instance segmentation.

## Due confronti distinti

**M01/A01/A02:** backbone e tokenizer all'interno dello stesso checkpoint AFA. Servono a studiare come il modulo aggrega le rappresentazioni disponibili; non costituiscono un'ablazione di una policy con il tokenizer rimosso.

**A07:** stessa diagnostica CLS nei due checkpoint addestrati separatamente. Permette di osservare differenze negli encoder dopo il training. Il modello TurboVLA usa `vli`, quello AFA usa `early_bert`: cambiano anche l'architettura di interazione e il percorso di apprendimento. Il confronto non isola causalmente il solo pooling.

**A06:** stessa cross-attention dell'ultimo layer ACT, mediata sulle otto head e sulle 12 query di azione. In TurboVLA i pesi sono disposti sulle posizioni dei token visivi dopo VLI; in AFA sono proiettati sulle patch tramite le mappe del tokenizer, che precedono il transformer AFA. Si normalizzano sulla singola camera e si mostra separatamente la massa assoluta di attenzione alla vista. Il diverso numero di token e la diversa fusione influenzano queste quantità: una mappa piatta dopo VLI non dimostra che il modello ignori gli oggetti, perché ogni token può già contenere informazione globale. Questa proiezione non è una attention rollout completa.

## Verifiche e provenienza

I due modelli vengono eseguiti in FP32 su CPU, batch size uno, evaluation e inference mode. Per **tutte le 42 osservazioni di ciascuna policy** si ripete l'inferenza con gli hook disabilitati: la differenza massima della predizione è **0.0**. I tensori AFA riprodotti coincidono con quelli già salvati nelle precedenti analisi: differenza massima **0.0** per mappe, coordinate e token fusi.

Lo scarto massimo del replay CLS rispetto al calcolo eager è circa $1.19\times10^{-7}$. Lo scarto massimo dei timestamp video è circa $4.45\times10^{-7}$ secondi. Sono state verificate finitezza degli array, normalizzazione delle distribuzioni spaziali e identità degli input RGB fra le policy. I risultati sono in [validation.json](validation.json).

[afa/manifest.json](afa/manifest.json) e [turbovla/manifest.json](turbovla/manifest.json) conservano hash dei checkpoint, configurazioni, frame, selezione e definizione della diagnostica. I tensori sono in `afa/arrays/` e `turbovla/arrays/`; includono le feature DINO dense per i sei midpoint a supporto di eventuali analisi successive. La selezione curata originale è conservata in [event_selection.json](event_selection.json).

## Rigenerazione

Gli script sono `scripts/extract_paper_attention.py` e `scripts/plot_paper_attention.py` nella repository; nella workspace originale si trovano in `SOT_Gianni/gap_policy/scripts/`. Richiedono anche gli helper dei precedenti script, LeRobot, i pacchetti TurboVLA/AFA e i checkpoint/asset locali. Il modello e il dataset verificato completo non sono inclusi in questo archivio.

Esempio dalla workspace originale:

```bash
cd /home/gpepe/ws
AFA_PYTHON=/home/gpepe/miniconda3/envs/lerobot/bin/python
AFA_SCRIPTS=SOT_Gianni/gap_policy/scripts
AFA_EXPORT=exports/tokenizer_qualitative/paper_style

"$AFA_PYTHON" "$AFA_SCRIPTS/extract_paper_attention.py" \
  --source-manifest exports/tokenizer_qualitative/analysis/manifest.json \
  --curated-selection "$AFA_EXPORT/event_selection.json" \
  --checkpoint /data2/gpepe/lerobot_runs/turbovla_afa_early_bert_multitask_full_80k_20261110/checkpoints/160000/pretrained_model \
  --output "$AFA_EXPORT/afa"

"$AFA_PYTHON" "$AFA_SCRIPTS/extract_paper_attention.py" \
  --source-manifest exports/tokenizer_qualitative/analysis/manifest.json \
  --curated-selection "$AFA_EXPORT/event_selection.json" \
  --checkpoint /data2/gpepe/lerobot_runs/turbovla_base_vli_multitask_full_80k_20261111/checkpoints/160000/pretrained_model \
  --output "$AFA_EXPORT/turbovla"

MPLCONFIGDIR=/tmp/afa-matplotlib OPENBLAS_NUM_THREADS=1 \
"$AFA_PYTHON" "$AFA_SCRIPTS/plot_paper_attention.py" "$AFA_EXPORT/afa" \
  --baseline "$AFA_EXPORT/turbovla" --output "$AFA_EXPORT/figures" --dpi 300
```

Le inferenze sono riprendibili dai file mancanti; il rendering usa solo gli array salvati. I controlli di identità della selezione considerano anche i metadati di provenienza: per cambiare selezione usare una nuova cartella di output.

## Riferimenti grafici e limiti delle conclusioni

- [AFA, figura 6](https://arxiv.org/pdf/2511.10762): confronti RGB/attention su scene corrispondenti. La [pagina degli autori](https://tsagkas.github.io/afa/index.html) distingue CLS attention e query AFA.
- [TokenLearner, sezione 6.4](https://arxiv.org/html/2106.11297v2#S6.SS4): RGB nel tempo e righe di attention map per token, prima dei moduli che mescolano informazione.
- [Slot Attention, figura 3](https://proceedings.neurips.cc/paper_files/paper/2020/file/8511df98c02ab60aea1b2356c013bc0f-Paper.pdf): disposizione per slot. Le nostre viste RGB pesate non sono le ricostruzioni e alpha mask prodotte dal decoder di quel lavoro.

Le dimostrazioni sono dati di training con comportamento corretto. Nessuna di queste nuove figure misura successi closed-loop, robustezza OOD o segmentazione annotata. Le figure possono documentare concentrazione, diffusione, ridondanza e variazione temporale delle distribuzioni, ma non dimostrano da sole che una query rappresenti una categoria di oggetti o causi l'esito del task.
