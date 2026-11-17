# Cubo e peluche: demo verificate, diagnostica completa

Si analizzano **solo due dimostrazioni verificate per task**, senza includere i rollout falliti nelle nuove raccolte. Sono gli stessi episodi scelti nelle analisi precedenti, questa volta con tutte le diagnostiche del peg su ogni frame.

| Task | Episodi del dataset | Frame per policy | Galleria | Metodo |
|---|---|---:|---|---|
| Cubo | 0 (319), 49 (249) | **568** | [Figure](cube_verified/GALLERY.md) | [Protocollo](cube_verified/README.md) |
| Peluche | 50 (365), 99 (295) | **660** | [Figure](plush_verified/GALLERY.md) | [Protocollo](plush_verified/README.md) |

Totale: **1.228 osservazioni per checkpoint**, 2.456 predizioni di chunk e 4.912 viste fra due camere e due checkpoint. Le raccolte contengono **302 figure PNG/PDF per task**, oltre alle versioni con etichette minime e ai pannelli separati. Dataset `Giannipe/Multi-task-verified`, revisione `9bd90bef6481029a3e00e3527c8f67ac04bc5b85`.

## Viewer offline: ogni frame e ogni slot

- Cubo: [episodio 0](cube_verified/viewer_episode_000.zip), [episodio 49](cube_verified/viewer_episode_049.zip).
- Peluche: [episodio 50](plush_verified/viewer_episode_050.zip), [episodio 99](plush_verified/viewer_episode_099.zip).

Scarica uno ZIP, estrailo e apri `viewer/index.html`. Include entrambe le camere, tutte le 16 query, mappe di pooling, routing RGB, keypoint con semantica aggregata e confronti con TurboVLA sugli stessi frame. Permette anche di scaricare PNG senza testo. Le cartelle `viewer/` delle due raccolte includono entrambi gli episodi del relativo task.

## Figure e dati

La struttura replica quella della [raccolta peg](successgrasp/README.md): atlanti di tutti gli slot, sequenze a dodici tempi, RGB pesato dal routing, attention DINO per head e layer, ACT per head/layer/query di azione, query linguistiche, geometria e semantica, rappresentazioni prima/dopo transformer, similarità e metriche temporali. PNG e PDF hanno padding nero rimosso solo nel display; le immagini clean non contengono testo.

Queste demo appartengono al dataset di training e mostrano comportamento registrato corretto. Il replay non costituisce un nuovo rollout riuscito delle policy. I checkpoint, le scale delle mappe e gli ID cromatici degli slot coincidono fra raccolte; le PCA sono adattate separatamente per task e camera, quindi i loro colori non sono categorie confrontabili fra task.

Gli array completi si conservano localmente in `/data2/gpepe/qualitative_verified_complete_20261117/{cube,plush}/`; GitHub include mappe di ogni frame per il viewer, figure, metriche, PCA e archivi completi di alcuni frame. Manifest, hash e controlli sono nei documenti di ciascun task.
