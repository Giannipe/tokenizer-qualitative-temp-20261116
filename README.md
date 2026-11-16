# AFA tokenizer: figure qualitative offline

[Nuove figure in stile paper: DINO→AFA e confronto TurboVLA](paper_style/README.md)

[Figure selezionate per la tesi: principali e appendice](thesis/GALLERY.md)

[Experimental Setup: metodo e significato delle figure](EXPERIMENTAL_SETUP_QUALITATIVE.md)

Archivio temporaneo privato per consultare e scaricare le figure della tesi. I PDF e PNG originali sono in `analysis/figures/`; gli script sono in `scripts/`.

[Galleria immagini su GitHub](GALLERY.md) · [Galleria HTML da aprire dopo il download](analysis/figures/index.html) · [Protocollo e didascalie](PROTOCOL.md)

Checkpoint early-BERT finale, 160000 micro-step. Sei fallimenti e sei dimostrazioni verificate; cinque frame per episodio e due camere: 60 osservazioni, 120 viste. Inferenza CPU FP32, nessun training.

## Figure da cui partire

- [Task 1, cubo: confronto frontale](analysis/figures/comparisons/task1_front.pdf) e [polso](analysis/figures/comparisons/task1_wrist.pdf).
- [Task 2, peluche: confronto frontale](analysis/figures/comparisons/task2_front.pdf) e [polso](analysis/figures/comparisons/task2_wrist.pdf).
- [Task 3, peg: confronto frontale](analysis/figures/comparisons/task3_front.pdf) e [polso](analysis/figures/comparisons/task3_wrist.pdf).
- [Tutti i 16 slot, fallimento del task 1](analysis/figures/slots/task1_failure_ep000_f00108_front.pdf).
- [Struttura dei token fusi, frontale](analysis/figures/embeddings/fused_tokens_front.pdf).

Tutte le figure sono disponibili in PDF e PNG a 300 dpi. Le dimostrazioni hanno movimenti corretti registrati nel dataset di training: non sono una misura del successo closed-loop della policy.

## Osservazioni descrittive

Si osservano regioni di attenzione concentrate, slot diffusi e slot ridondanti. Le medie su questi pochi episodi non indicano una separazione netta e sistematica fra fallimenti e dimostrazioni in termini di entropia o ridondanza delle mappe. La camera al polso mostra mediamente mappe più diffuse di quella frontale. Non si può attribuire causalmente un fallimento alla posizione di uno slot da queste sole figure.

La tavola di dettaglio del task 1, episodio fallito 0 al frame 108, contiene ad esempio uno slot frontale S04 con entropia normalizzata prossima a 0.98 e più slot focalizzati sulla stessa regione del robot. Il renderer conserva entrambi i fenomeni.

## Metriche medie sui cinque frame dei due episodi per gruppo

H spaziale è divisa per log N; H routing è divisa per log K. Il coseno spaziale rimuove prima il livello uniforme. Massa ACT è la frazione di cross-attention dell’ultimo layer assegnata alla vista, mediata su head e query di azione. Sono statistiche descrittive, senza intervalli di confidenza.

| Task | Gruppo | Camera | H spaziale | Coseno centrato | H routing | Massa ACT |
|---|---|---|---:|---:|---:|---:|
| 1 | failure | front | 0.344 | 0.137 | 0.060 | 20.9% |
| 1 | failure | wrist | 0.669 | 0.194 | 0.217 | 13.3% |
| 2 | failure | front | 0.384 | 0.122 | 0.074 | 19.7% |
| 2 | failure | wrist | 0.626 | 0.275 | 0.122 | 14.2% |
| 3 | failure | front | 0.366 | 0.102 | 0.079 | 23.2% |
| 3 | failure | wrist | 0.636 | 0.377 | 0.095 | 17.5% |
| 1 | verified_demo | front | 0.389 | 0.165 | 0.067 | 24.6% |
| 1 | verified_demo | wrist | 0.647 | 0.318 | 0.115 | 14.0% |
| 2 | verified_demo | front | 0.383 | 0.151 | 0.069 | 20.4% |
| 2 | verified_demo | wrist | 0.614 | 0.320 | 0.136 | 14.2% |
| 3 | verified_demo | front | 0.372 | 0.138 | 0.064 | 23.9% |
| 3 | verified_demo | wrist | 0.633 | 0.346 | 0.138 | 16.5% |

## Verifiche

I controlli in [validation.json](analysis/validation.json) verificano finitezza degli array, normalizzazione del routing, ricostruzione dei keypoint dalle attention map, dimensione delle azioni e sincronizzazione video. Differenza massima fra output della policy con e senza hook diagnostici: **0.0** nel frame di controllo.

[manifest.json](analysis/manifest.json) registra SHA-256 del checkpoint e revisioni complete dei quattro dataset. [metrics.csv](analysis/figures/metrics.csv) contiene tutte le 120 righe per vista. Gli array NPZ permettono di rifare il rendering senza ripetere inferenza.
