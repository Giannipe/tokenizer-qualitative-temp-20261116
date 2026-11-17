# Analisi qualitativa offline del tokenizer AFA

**Estensione alle sole demo verificate:** cubo, episodi 0/49 (568 frame), e peluche, episodi 50/99 (660 frame), elaborati integralmente con entrambi i checkpoint. Vedi [indice](VERIFIED_DEMOS.md) e sezione 17.

**Nuovo studio successgrasp:** tutti i 706 frame dei due rollout vengono elaborati con entrambi i checkpoint, conservando tutti i 16 slot per camera. Vedi [metodo completo](successgrasp/README.md) e sezione 16 per il testo inglese.

**Aggiornamento: analisi integrale dei 12 episodi.** Per le ultime figure pulite si usano **4.117 osservazioni per checkpoint** (tutti i frame dei sei fallimenti e delle sei dimostrazioni selezionate), non i precedenti campioni da 60 o 42 osservazioni. Il protocollo, la selezione successiva di slot/frame e il crop solo grafico sono documentati in [full_episodes/README.md](full_episodes/README.md). La sezione 15 in fondo contiene il testo inglese aggiornato.


**Distinzione fra le due analisi.** Le sezioni 1–13 descrivono la prima raccolta esplorativa: 60 osservazioni di fallimenti e dimostrazioni verificate. Le nuove figure in stile paper DINO→AFA e il confronto TurboVLA usano invece **42 osservazioni dei soli sei episodi verificati**, con due checkpoint. Il protocollo aggiornato è in [paper_style/README.md](paper_style/README.md); la sezione 14 riporta un testo inglese riferito a queste nuove figure. I due campioni non vanno confusi nella descrizione dell'Experimental Setup.

Questo documento descrive il protocollo usato per produrre le figure qualitative di TurboVLA-AFA: selezione delle osservazioni, inferenza sul checkpoint, estrazione dei tensori interni, costruzione delle visualizzazioni e criteri di interpretazione. La sezione finale propone un testo in inglese adattabile all'Experimental Setup della tesi.

Le visualizzazioni derivano dai tensori del modello e dalle immagini registrate. Nessuna figura è generata con un modello di image generation e nessuna segmentazione manuale viene usata per costruire le mappe.

## 1. Obiettivo e natura dell'esperimento

L'analisi studia come il tokenizer comprime le feature visive in un insieme di slot e come la policy consulta la rappresentazione risultante. In particolare, si osservano la concentrazione spaziale dei singoli slot, le loro sovrapposizioni, le feature semantiche aggregate e la struttura dei token che combinano geometria e semantica.

Si confrontano due condizioni: osservazioni di rollout reali falliti della policy e osservazioni di dimostrazioni verificate con comportamento corretto. In entrambe le condizioni si esegue offline lo stesso checkpoint, fornendo immagini, stato e istruzione registrati.

Non si esegue training, non si aggiorna alcun peso e non si comandano azioni al robot. Le azioni vengono predette e salvate, ma non eseguite. Di conseguenza, questa procedura è uno studio descrittivo delle rappresentazioni interne su osservazioni registrate; non stima il tasso di successo closed-loop della policy.

## 2. Checkpoint e configurazione effettiva

Si usa il checkpoint finale della run `turbovla_afa_early_bert_multitask_full_80k_20261110`, cartella `checkpoints/160000/pretrained_model`. Il valore 160000 è il contatore di micro-step del training; la run prevedeva accumulo dei gradienti pari a due e 80000 aggiornamenti dell'ottimizzatore, salvo eventuali aggiornamenti saltati da AMP.

| Componente | Configurazione usata nell'analisi |
|---|---|
| Encoder visivo | DINOv3 ViT-B/16, con i pesi contenuti nel checkpoint della policy |
| Input per vista | RGB 256 × 256 dopo preprocessing |
| Camere | Frontale (`front`) e polso (`wrist`) |
| Patch DINO per vista | Griglia 16 × 16: 256 patch, esclusi CLS e register token |
| Slot per vista | 16 query apprese |
| Pooler AFA | Due pooler separati, uno per camera |
| Attention spaziale AFA | Single-head, competitive routing, temperatura 0.5 |
| Condizionamento linguistico | `early_bert`: cross-attention delle query verso i token BERT, con 12 head |
| Fusione del tokenizer | Somma dell'encoding geometrico e dell'encoding semantico |
| Dimensione del token fuso | 256 |
| Transformer AFA successivo | Abilitato, un layer e quattro head |
| Decoder delle azioni | Tre layer ACT; cross-attention con otto head |
| Stato / azione | Sei componenti; chunk predetto di 12 azioni |
| Esecuzione dell'analisi | Batch size 1, CPU, FP32, quattro thread PyTorch |
| Modalità | `eval()` e `torch.inference_mode()`, senza autocast AMP |

La variante analizzata è quindi **early-BERT con tokenizer spaziale single-head**. Il numero di head del condizionamento linguistico, quello del transformer AFA e quello del decoder ACT appartengono a moduli diversi e non rendono multi-head il pooling spaziale di questo checkpoint.

Nel tokenizer non sono attivi top-k pruning (`top_k_ratio = 1`), windowing, sink/trash query, prior posizionali 2D o feature di extent. Le ellissi delle figure vengono calcolate a posteriori dalle mappe; non sono feature di extent fornite al modello.

## 3. Dataset e selezione degli episodi

Per ciascuno dei tre task si includono entrambi gli episodi dei repository di fallimento indicati dall'utente e due episodi di `Giannipe/Multi-task-verified`.

| Task | Istruzione registrata | Episodi falliti | Episodi verificati |
|---|---|---|---|
| 1 | Grab the red cube and place on the gray block | 0 e 1 del dataset task1 | 0 e 49 |
| 2 | Grab the plush and place in the box | 0 e 1 del dataset task2 | 50 e 99 |
| 3 | Grab the peg and push the red cube | 0 e 1 del dataset task3 | 100 e 149 |

I repository di fallimento sono:

- `Giannipe/rollout_tokenizer_attention_task1_20261116_154605`;
- `Giannipe/rollout_tokenizer_attention_task2_20261116_155022`;
- `Giannipe/rollout_tokenizer_attention_task3_20261116_155319`.

La selezione delle dimostrazioni è deterministica: primo e ultimo episodio disponibile per ciascuna istruzione, ordinando per indice di episodio. Non si scelgono le dimostrazioni in base all'estetica o all'interpretabilità delle mappe ottenute. I fallimenti sono quelli segnalati dall'utente; non viene ricavata un'etichetta di fallimento dalle attention map.

Le dimostrazioni selezionate appartengono al dataset usato per il training della policy. Sono dunque riferimenti con comportamento corretto su dati di training, non un insieme held-out e non rollout riusciti prodotti dal checkpoint durante questa analisi.

### Campionamento temporale

Per un episodio di lunghezza $T$ si selezionano cinque frame:

$$
t_j = \operatorname{round}\!\left(r_j(T-1)\right),
\qquad r_j \in \{0.1,0.3,0.5,0.7,0.9\}.
$$

Il campione comprende quindi 12 episodi, 60 osservazioni e 120 viste. La frequenza nominale dei dataset è 30 Hz. Le figure di dettaglio e i confronti usano il frame al 50% della durata; le sequenze temporali comprendono tutti e cinque gli istanti. Le PCA usano tutti i frame selezionati.

Questo campionamento rende riproducibile la scelta dei frame, ma non allinea contatto, presa, trasporto o istante del fallimento. Il midpoint di un episodio fallito può precedere l'errore che ne determina l'esito.

## 4. Ricostruzione delle osservazioni e preprocessing

I dataset sono letti nel formato LeRobot v3. Per ogni osservazione si identifica univocamente la riga tramite indice di episodio e indice del frame. Lo stato proviene dalla stessa riga; l'istruzione corrisponde al task dell'episodio.

I video possono contenere più episodi nello stesso file. Per ciascuna camera il timestamp richiesto è:

$$
t_{\mathrm{video}} = t_{\mathrm{inizio\ episodio,\ camera}} + t_{\mathrm{osservazione}}.
$$

PyAV decodifica il frame con presentation timestamp, PTS, più vicino al timestamp richiesto. Lo script verifica uno scarto non superiore a $1.01/f_{\mathrm{dataset}}$, cioè poco più di un frame. Gli offset dei due flussi sono gestiti separatamente.

Le immagini RGB decodificate sono convertite in float nell'intervallo $[0,1]$. Immagini e stato attraversano i processor salvati con il checkpoint, conservando le statistiche di training. Per i frame 640 × 480 il resize proporzionale produce 256 × 192 e aggiunge 32 pixel neri sopra e sotto, ottenendo 256 × 256. Non viene applicato un crop.

Per visualizzare l'RGB si inverte la normalizzazione del tensore realmente fornito al modello, anziché sovrapporre mappe a un'immagine con trasformazioni spaziali differenti. Le bande di padding rimangono visibili. Le 256 patch includono anche il padding: l'analisi non elimina queste patch e usa $N=256$ come riferimento per entropia e attenzione uniforme.

## 5. Inferenza ed estrazione dei tensori

Il checkpoint completo viene caricato in modalità strict e posto in evaluation. Ogni osservazione viene elaborata indipendentemente con `predict_action_chunk()`, che calcola un nuovo chunk; non si usa una coda di azioni già predette. Non si applicano augmentation o perturbazioni del prompt.

Gli hook sui pooler leggono output e diagnostiche già prodotti durante il normale forward. Per le cross-attention early-BERT e ACT, che normalmente non restituiscono i pesi, si ripete soltanto la piccola operazione di attention sugli stessi input, con `need_weights=True` e `average_attn_weights=False`. Questa ripetizione diagnostica non sostituisce l'output usato dalla policy e non riesegue l'intero encoder per ricostruire arbitrariamente una mappa.

Su una prima osservazione di controllo si confronta la predizione con e senza hook. La differenza massima osservata è 0.0. Questa verifica riguarda il frame di controllo, non una seconda esecuzione di tutte le 60 osservazioni.

Per ogni osservazione si salva un archivio NPZ con:

| Quantità | Forma o significato |
|---|---|
| RGB preprocessato, per camera | 256 × 256 × 3 |
| Attention spaziale, per camera | 16 × 16 × 16: slot × altezza × larghezza |
| Routing competitivo, per camera | Probabilità delle 16 query sulle 256 patch |
| Coordinate, per camera | 16 × 2 |
| Embedding semantici, per camera | 16 × 256, dopo `semantic_encoder` |
| Token fusi, per camera | 16 × 256, prima del transformer AFA |
| Attention early-BERT, per camera | 12 head × 16 query × token di testo |
| Attention ACT | Otto head × 12 query di azione × token di memoria |
| Output delle azioni | Chunk denormalizzato 12 × 6 |
| Altri dati | Azione registrata al frame corrente, token BERT e timestamp video richiesti/decodificati |

Le azioni registrate e predette sono conservate per tracciabilità, ma le 104 tavole non costituiscono un confronto di traiettorie eseguite o una misura dell'errore di azione.

## 6. Dal routing ai token: quantità da distinguere

Le formule seguenti descrivono il checkpoint single-head effettivamente analizzato. Per semplicità si omettono gli indici di batch, tempo e camera.

### Probabilità di assegnazione di una patch agli slot

Siano $k=1,\ldots,K$ gli slot, $i=1,\ldots,N$ le patch e $\ell_{ki}$ i logit di compatibilità query–patch prodotti dalle proiezioni del modello. Con temperatura $\tau=0.5$, il competitive routing calcola:

$$
R_{ki}=\frac{\exp(\ell_{ki}/\tau)}{\sum_{j=1}^{K}\exp(\ell_{ji}/\tau)}.
$$

La normalizzazione è sulle query: per ogni patch $i$, $\sum_k R_{ki}=1$. Si può leggere $R_{ki}$ come probabilità di assegnazione allo slot $k$ nel routing del modello, non come probabilità calibrata di appartenenza a una classe o a un oggetto annotato.

### Distribuzione spaziale del singolo slot

Per aggregare feature e coordinate, il modello normalizza spazialmente il routing di ogni slot:

$$
A_{ki}=\frac{R_{ki}}{\sum_{n=1}^{N}R_{kn}+\varepsilon}.
$$

Nel codice del tokenizer $\varepsilon=10^{-8}$. La somma spaziale di $A_k$ è quindi circa uno. Il renderer rinormalizza ancora queste mappe per ottenere distribuzioni $p_{ki}=A_{ki}/\sum_n A_{kn}$ usate nelle statistiche e nelle heatmap.

La distinzione è essenziale: **$R$ confronta gli slot su una patch; $p$ confronta le patch per uno slot**. Uno slot può vincere su molte patch di sfondo e avere una mappa spaziale diffusa; un altro può ricevere poche patch ma avere una mappa molto concentrata.

### Geometria e semantica aggregate

Se $g_i\in\mathbb{R}^2$ è la coordinata della patch nella griglia normalizzata e $V_i$ è la feature value proiettata, il tokenizer calcola:

$$
\mu_k = \sum_i A_{ki}g_i,
\qquad s_k = \sum_i A_{ki}V_i.
$$

Per questa configurazione senza extent, il token emesso dal tokenizer è:

$$
z_k = f_{\mathrm{coord}}(\mu_k)+f_{\mathrm{sem}}(s_k)\in\mathbb{R}^{256}.
$$

Il vettore salvato come `semantics` è $e_k=f_{\mathrm{sem}}(s_k)$; quello salvato come `tokens` è $z_k$. Dopo questa fusione, i token attraversano il transformer AFA, la proiezione visiva e l'aggiunta dell'embedding di vista prima di essere consultati dal decoder ACT.

Le feature DINO sono contestualizzate dall'encoder: una mappa concentrata su una patch non significa che il token possa contenere esclusivamente informazione dei pixel di quella patch.

## 7. Costruzione delle visualizzazioni

### 7.1 Attention map per ciascun token: `slots/`

Le tavole “What each slot pools” sono griglie 4 × 4 contenenti tutti i 16 slot. In ogni riquadro si sovrappone all'RGB la distribuzione spaziale di uno slot, si indica il suo baricentro e si riporta la sua entropia.

Per mantenere una scala comparabile fra slot, frame e condizioni, si visualizza il logaritmo dell'attenzione relativa alla distribuzione uniforme:

$$
L_{ki}=\operatorname{clip}\!\left(\log_2\!\left(\max(Np_{ki},1)\right),0,3\right).
$$

Il valore 0 corrisponde al livello uniforme o inferiore, 1 a 2×, 2 a 4× e 3 ad almeno 8×. Non viene applicata una normalizzazione min/max indipendente a ogni slot. I valori sotto il livello uniforme non ricevono colorazione; quelli sopra 8× sono saturati.

Si usa la colormap `magma`; l'opacità dell'overlay cresce come $0.75L/3$. La mappa viene interpolata bilinearmente per sovrapporla all'RGB: la risoluzione nativa resta 16 × 16. L'interpolazione migliora la leggibilità grafica senza aggiungere evidenza spaziale.

L'entropia spaziale normalizzata riportata sopra ogni slot è:

$$
H_k^{\mathrm{sp}}=\frac{-\sum_i p_{ki}\log p_{ki}}{\log N}.
$$

Valori vicini a zero indicano concentrazione su poche patch; valori vicini a uno indicano una distribuzione quasi uniforme. Un'attenzione concentrata non implica automaticamente un oggetto corretto o una decisione corretta.

### 7.2 Geometria degli slot

I punti rappresentano le coordinate medie $\mu_k$ prodotte dal tokenizer. Per allineare i punti al raster, le coordinate definite sugli estremi della griglia vengono riportate nei centri fisici delle patch. Per una griglia $H_g\times W_g$ si usa:

$$
\mu_k^{\mathrm{img}}=\operatorname{diag}\!\left(\frac{W_g-1}{W_g},\frac{H_g-1}{H_g}\right)\mu_k.
$$

Le ellissi derivano dalla dispersione delle patch attorno a queste coordinate, pesata con $p_k$. Indicando con $x_i$ i centri delle patch nel riferimento dell'immagine:

$$
\Sigma_k=\sum_i p_{ki}(x_i-\mu_k^{\mathrm{img}})(x_i-\mu_k^{\mathrm{img}})^\top.
$$

L'orientamento segue gli autovettori di $\Sigma_k$ e i semiassi sono le radici dei suoi autovalori. La rappresentazione coincide, entro gli effetti numerici della normalizzazione con epsilon, con un'ellisse di dispersione a una deviazione standard. Non è una bounding box, un intervallo di confidenza o una regione che garantisce una determinata percentuale di massa per distribuzioni arbitrarie.

Un baricentro fra due oggetti può derivare da una distribuzione bimodale: per capire che cosa aggrega lo slot bisogna consultare anche la sua heatmap.

### 7.3 Assegnazione spaziale e competizione fra slot

Il pannello “Soft slot assignment” assegna a ogni patch il colore della query con massimo routing:

$$
k^*(i)=\operatorname*{arg\,max}_k R_{ki}.
$$

L'opacità è:

$$
\alpha_i=0.65\,\frac{\max_k R_{ki}-1/K}{1-1/K}.
$$

L'ID vincente è quindi discreto, mentre la sua evidenza visiva dipende dalla probabilità del vincitore rispetto al livello uniforme. L'assegnazione è visualizzata sulla griglia delle patch, senza trasformarla in una segmentazione fine degli oggetti.

Il pannello “Slot competition entropy” usa invece:

$$
H_i^{\mathrm{route}}=\frac{-\sum_k R_{ki}\log R_{ki}}{\log K}.
$$

Valori bassi indicano che una query domina quella patch; valori alti indicano che più query ricevono probabilità simili. Questa entropia è calcolata **sugli slot per una patch**, mentre $H_k^{\mathrm{sp}}$ è calcolata **sulle patch per uno slot**. Le due quantità rispondono a domande diverse e non vanno scambiate.

### 7.4 Canvas semantico sovrapposto all'RGB

Per ciascuna camera si raccolgono gli embedding $e_k$ di tutti i 60 frame selezionati: 960 vettori, includendo entrambi i gruppi e tutti i task. Si centra questa matrice e si calcola una PCA senza standardizzazione per componente o whitening. I primi tre componenti vengono associati ai tre canali cromatici.

Se $u_{kc}$ è il punteggio PCA dello slot nel componente $c$, la trasformazione cromatica è:

$$
C_{kc}=\operatorname{clip}\!\left(\frac{u_{kc}-q_{0.02,c}}{q_{0.98,c}-q_{0.02,c}},0.08,0.92\right),
$$

dove i quantili sono stimati sulla stessa raccolta di 960 vettori. La direzione di ciascun componente è fissata imponendo positivo il loading di massimo valore assoluto. La PCA e i limiti cromatici restano condivisi fra i frame della stessa camera; non vengono ricalcolati separatamente per ogni immagine.

I colori degli slot vengono riportati sulla griglia RGB mediante il routing:

$$
S_i=\sum_k R_{ki}C_k,
\qquad I_{\mathrm{sem}}=0.38I_{\mathrm{RGB}}+0.62\,\operatorname{upsample}(S).
$$

Questa costruzione mostra variazioni di embedding proiettate nello spazio dell'immagine. Non è una ricostruzione RGB, una classificazione di oggetti o la PCA delle feature dense di tutte le patch. I colori sono assegnati agli embedding già aggregati dagli slot e poi distribuiti spazialmente tramite il routing.

Un'immagine quasi monocromatica non prova che la scena contenga un solo oggetto: può riflettere embedding proiettati simili o il predominio di alcuni slot nel routing. Le proiezioni sono stimate separatamente per frontale e polso, quindi i colori semantici non sono confrontabili direttamente fra le due camere.

### 7.5 Similarità spaziale e semantica: `diagnostics/`

La prima matrice confronta le mappe degli slot mediante coseno dopo sottrazione del livello uniforme:

$$
\widetilde p_k=p_k-\frac{1}{N}\mathbf{1},\qquad
M^{\mathrm{sp}}_{kl}=\frac{\widetilde p_k^\top\widetilde p_l}{\|\widetilde p_k\|\,\|\widetilde p_l\|}.
$$

Le divisioni sono regolarizzate numericamente. Una mappa esattamente uniforme produce un vettore centrato nullo e contribuisce con similarità zero. La sottrazione evita che un fondo piatto comune venga confuso con un picco condiviso.

Le celle fuori diagonale vicine a +1 indicano pattern spaziali simili; valori prossimi a zero indicano bassa correlazione; valori negativi indicano deviazioni dal livello uniforme in direzioni opposte. Similarità elevata suggerisce ridondanza spaziale, ma da sola non dimostra collasso dell'intero tokenizer.

La seconda matrice usa il coseno ordinario degli embedding semantici:

$$
M^{\mathrm{sem}}_{kl}=\frac{e_k^\top e_l}{\|e_k\|\,\|e_l\|}.
$$

Osservare la stessa regione e produrre embedding simili sono proprietà distinte. Entrambe le matrici usano una scala condivisa da −1 a +1.

### 7.6 Attenzione delle query al linguaggio

La terza matrice delle diagnostiche visualizza l'attenzione early-BERT, mediata sulle 12 head: righe = query degli slot, colonne = token BERT dell'istruzione, inclusi `[CLS]` e `[SEP]`. La scala va da 0 a 1 e i token di padding sono mascherati dal modello.

Questo modulo condiziona le query apprese con il testo **prima** del pooling visivo. Nella configurazione corrente le sue query non ricevono le feature dell'immagine prima della cross-attention linguistica. A parità di istruzione e camera, quindi, questa matrice può essere invariata fra frame: descrive il condizionamento testuale delle query, non un grounding parola–pixel ricalcolato dall'immagine. Non sono state eseguite ablation del prompt per dimostrare l'effetto causale delle singole parole.

### 7.7 Attenzione del decoder delle azioni proiettata sull'immagine

Si usa la cross-attention dell'ultimo dei tre layer ACT. Se $B_{ham}$ è il peso della head $h$, query di azione $a$ e token di memoria $m$, si calcola:

$$
\bar b_m=\frac{1}{8\cdot12}\sum_{h=1}^{8}\sum_{a=1}^{12} B_{ham}.
$$

La memoria contiene prima i 32 token visivi, ordinati per camera, poi i token linguistici e infine due token di stato. Per una vista $v$, si estrae il sottoinsieme dei suoi 16 pesi $\bar b_{vk}$ e si conserva separatamente la massa assoluta:

$$
m_v=\sum_k\bar b_{vk}.
$$

Il valore `decoder mass to this view` nell'overview è questa frazione dell'attenzione totale. Per costruire il canvas della singola vista si rinormalizza invece sulla vista:

$$
\beta_{vk}=\frac{\bar b_{vk}}{m_v},\qquad
D_{vi}=\sum_k\beta_{vk}p_{vki}.
$$

Si visualizza $D_v$ usando la stessa trasformazione logaritmica rispetto all'uniforme delle heatmap degli slot; l'opacità cresce come $0.8L/3$. Se la massa della vista fosse nulla, lo script produrrebbe una mappa nulla.

La rinormalizzazione implica che una heatmap intensa non dimostra da sola che quella camera riceva molta attenzione complessiva: per questo va letta insieme a $m_v$. Inoltre, la media sulle 12 query elimina le differenze fra i passi del chunk predetto.

La proiezione usa le mappe del tokenizer precedente al transformer AFA, mentre ACT consulta token che hanno già attraversato quel transformer e la proiezione visiva. È dunque una **proiezione diagnostica action-to-slot**, non una attention rollout completa attraverso tutti i layer e non una misura di salienza causale dei pixel.

### 7.8 PCA dei token fusi: `embeddings/`

Una seconda analisi PCA usa i vettori $z_k$, che includono geometria e semantica, prima del transformer AFA. Per ogni camera la PCA è stimata congiuntamente sui 960 token selezionati, senza normalizzazione L2, standardizzazione delle componenti o whitening.

Ogni punto rappresenta uno slot in un frame. Il colore identifica la query, il cerchio vuoto un'osservazione di fallimento e la croce una dimostrazione verificata. I tre pannelli dei task condividono proiezione e assi. Per ciascun task vengono rappresentati 320 token: quattro episodi × cinque frame × 16 slot.

Le percentuali sugli assi riportano la varianza spiegata da PC1 e PC2. La vicinanza in due dimensioni descrive soltanto la proiezione di vettori originariamente a 256 dimensioni. Eventuali gruppi possono dipendere da identità della query, contenuto, geometria o contesto della scena. Non si applica k-means, non si impone un numero di cluster e non si verifica una corrispondenza con istanze annotate.

## 8. Organizzazione delle 104 tavole

| Famiglia | Numero | Contenuto |
|---|---:|---|
| `comparisons/` | 6 | Una tavola per task e camera; quattro episodi al midpoint, con geometria, semantica e proiezione ACT |
| `overview/` | 24 | Una tavola per episodio e camera; sei pannelli: RGB, geometria, assegnazione, semantica, entropia del routing, proiezione ACT |
| `slots/` | 24 | Una tavola per episodio e camera; tutte le 16 attention map al midpoint |
| `diagnostics/` | 24 | Similarità spaziale, similarità semantica e attention early-BERT al midpoint |
| `timelines/` | 24 | Una tavola per episodio e camera; cinque istanti, con posizioni degli slot e proiezione ACT |
| `embeddings/` | 2 | Una tavola per camera; PCA dei token fusi su tutti i frame e task |

Le figure temporali mostrano inferenze indipendenti ordinate nel tempo. Non viene introdotto un algoritmo di tracking o un'associazione delle istanze fra frame. Il colore di uno slot identifica la stessa query appresa nella stessa camera, ma non garantisce che essa segua sempre lo stesso oggetto.

I pooler delle due camere hanno parametri separati: S00 frontale e S00 polso non sono automaticamente lo stesso slot semantico. Anche i colori PCA hanno una trasformazione separata per camera. I colori degli ID degli slot, quelli del canvas semantico e quelli delle heatmap hanno funzioni differenti.

## 9. Metriche descrittive e unità di analisi

`analysis/figures/metrics.csv` contiene 120 righe, una per osservazione e camera. Per ogni riga si riportano l'entropia spaziale media sui 16 slot, la similarità spaziale centrata media sulle coppie fuori diagonale, l'entropia del routing media sulle patch, la massa ACT della vista e lo scarto massimo di sincronizzazione dei video dell'osservazione.

Le tabelle di riepilogo riportano la media aritmetica di queste quantità sui cinque frame dei due episodi di ogni task, gruppo e camera. Sono medie descrittive: non vengono prodotti test d'ipotesi, intervalli di confidenza o stime di generalizzazione. I cinque frame dello stesso episodio sono correlati e non rappresentano cinque esecuzioni indipendenti del task.

Queste metriche aiutano a leggere le figure, ma non stabiliscono che uno slot sia semanticamente corretto o che la sua posizione sia la causa del successo o del fallimento.

## 10. Riproducibilità e controlli effettuati

L'estrazione e il rendering sono separati: il primo script salva i tensori, il secondo crea tutte le figure senza caricare il modello. Cambiare font, layout o selezione delle famiglie di tavole non richiede nuovo training o nuova inferenza.

I riferimenti principali nell'archivio sono:

- [`analysis/manifest.json`](analysis/manifest.json): selezione esatta dei frame, istruzioni, configurazione, device, versione PyTorch, revisioni complete dei quattro dataset e hash del checkpoint;
- [`analysis/arrays/`](analysis/arrays/): i 60 archivi NPZ delle osservazioni;
- [`analysis/validation.json`](analysis/validation.json): esito dei controlli;
- [`analysis/figures/semantic_pca.npz`](analysis/figures/semantic_pca.npz): media, base e limiti cromatici della PCA semantica per camera;
- [`analysis/figures/metrics.csv`](analysis/figures/metrics.csv): metriche per osservazione e camera;
- [`analysis/figures/export_metadata.json`](analysis/figures/export_metadata.json): conteggi e convenzioni di export.

Lo SHA-256 del file `model.safetensors` analizzato è:

```text
ece1039b4e80cb510a4581dcb4b3cd9b515a4eb5fd27769c7b6ab01c4fa14d71
```

L'esecuzione registrata usa PyTorch `2.10.0+cu128` su CPU. Il suffisso CUDA della distribuzione PyTorch non implica che l'analisi sia stata eseguita su GPU.

Sono stati controllati tutti i 60 archivi e le 120 viste: finitezza dei tensori, dimensioni attese, normalizzazione del routing e ricostruzione delle coordinate dalle mappe. Lo scarto massimo di ricostruzione dei keypoint è circa $1.01\times10^{-6}$ nelle coordinate normalizzate. Lo scarto video massimo è circa $8.90\times10^{-7}$ secondi. Nel frame di controllo la differenza fra predizioni con e senza hook è esattamente zero.

Ogni tavola è esportata in PNG a 300 dpi e in PDF con testo incorporato. Sono state verificate integrità dei 104 PNG, presenza dei corrispondenti 104 PDF e risoluzione dei collegamenti nella galleria HTML. L'RGB e le heatmap restano dati raster; la risoluzione grafica di export non aumenta la risoluzione spaziale delle mappe del modello.

## 11. Limiti da mantenere espliciti nella tesi

Il confronto coinvolge due episodi per condizione e task e frame scelti a tempi relativi, non fasi annotate. Contesto, disposizione degli oggetti e traiettorie possono differire fra dimostrazioni e rollout. Una differenza nelle mappe o negli embedding non può quindi essere attribuita al solo esito dell'episodio.

Le dimostrazioni sono dati di training con comportamento corretto, mentre i fallimenti sono rollout della policy. Non si può presentare questo confronto come un esperimento controllato successo-versus-fallimento della stessa policy a parità di stato iniziale, né come valutazione held-out.

L'analisi non dispone di maschere di oggetti o corrispondenze query–istanza annotate. Regioni colorate, punti vicini e gruppi nella PCA non dimostrano da soli segmentazione semantica, clustering per istanza o tracking stabile. Le attention map mostrano i pesi di aggregazione del modello; non sostituiscono esperimenti causali con occlusioni, ablation o interventi controllati.

## 12. Testo in inglese adattabile all'Experimental Setup

### Offline qualitative analysis of the AFA tokenizer

We analyzed the visual representations produced by the final real-world TurboVLA-AFA early-BERT checkpoint using offline replay of recorded observations. The checkpoint uses a DINOv3 ViT-B/16 encoder and compresses the 16 × 16 patch grid of each camera into 16 learned slot tokens. The front and wrist cameras have separate AFA poolers. Spatial pooling is single-head with competitive routing at temperature 0.5, while a separate 12-head early-BERT cross-attention module conditions the learned queries on the instruction. Each tokenizer output combines an encoded spatial mean with an encoded attention-weighted semantic feature through additive fusion.

For each of three manipulation tasks, we included two recorded failed policy rollouts and two verified demonstrations from the training dataset. Demonstrations were selected deterministically as the first and last episode of each task. Five observations were sampled at 10%, 30%, 50%, 70%, and 90% of each episode, yielding 60 observations and 120 camera views. Detailed panels used the temporal midpoint, while temporal sequences and embedding projections used all sampled observations. Relative episode times were not treated as aligned manipulation phases.

The complete policy was evaluated with batch size one in FP32 on CPU, in evaluation mode and with gradients disabled. RGB frames were decoded using episode-specific video offsets and the nearest presentation timestamp. Recorded images and robot states were processed with the checkpoint's saved processors, including aspect-ratio-preserving resizing and padding to 256 × 256. The displayed RGB input was recovered from the actual normalized policy input. Predicted 12-action chunks were stored but were not executed on the robot.

We extracted per-slot spatial attention, competitive routing probabilities, spatial means, encoded pooled semantics, fused tokenizer outputs, early-BERT query-to-token attention, and final-layer ACT cross-attention. Spatial heatmaps used a common logarithmic scale relative to uniform attention, clipped between 1× and 8×, rather than independent per-map contrast normalization. We additionally visualized spatial dispersion, normalized spatial and routing entropy, centered attention-map cosine similarity, and semantic cosine similarity. Pooled semantic features were mapped to RGB colors using a shared three-component PCA fitted separately for each camera over all sampled slots and conditions. A separate shared two-dimensional PCA visualized the fused tokenizer outputs before the AFA transformer. Final-layer ACT attention, averaged over heads and action queries, was projected onto tokenizer maps for visualization, with the absolute attention mass assigned to each camera reported separately.

All figures were generated from saved model tensors without further training. Instrumentation was checked against an uninstrumented prediction on one control observation, obtaining zero maximum absolute output difference. These visualizations describe internal aggregation and attention patterns. Verified demonstrations provide examples of correct recorded behavior, rather than measured policy successes, and attention patterns or projected embedding groups do not by themselves establish object-instance clustering or causal explanations of task failure.

## 13. Codice della pipeline

Nel repository delle figure gli script sono [`scripts/tokenizer_qualitative.py`](https://github.com/Giannipe/tokenizer-qualitative-temp-20261116/blob/main/scripts/tokenizer_qualitative.py) per l'estrazione e [`scripts/plot_tokenizer_qualitative.py`](https://github.com/Giannipe/tokenizer-qualitative-temp-20261116/blob/main/scripts/plot_tokenizer_qualitative.py) per il rendering. Nella workspace originale si trovano sotto `SOT_Gianni/gap_policy/scripts/`.

Il [protocollo operativo della repository](https://github.com/Giannipe/tokenizer-qualitative-temp-20261116/blob/main/PROTOCOL.md) contiene i comandi di rigenerazione e le didascalie. La [galleria delle figure](https://github.com/Giannipe/tokenizer-qualitative-temp-20261116/blob/main/GALLERY.md) permette di consultare le sei famiglie di tavole. Questo documento descrive il setup e il significato delle misure; eventuali interpretazioni dei singoli fallimenti vanno sviluppate separatamente nella discussione dei risultati.

## 14. Testo inglese per le nuove figure DINO→AFA e il confronto TurboVLA

### Paired backbone and tokenizer visualization

For the compact qualitative figures, we replayed the same six verified demonstrations through the fine-tuned TurboVLA-AFA early-BERT and TurboVLA VLI checkpoints. The observation set comprised 30 frames sampled at five fixed relative times in each episode and 12 additional frames selected by RGB inspection of manipulation events, yielding 42 observations and 84 camera views per checkpoint. The original exploratory failure-versus-demonstration analysis used a separate 60-observation selection. All demonstrations were training examples, and no actions were executed during offline analysis.

We extracted final-layer DINOv3 CLS-to-patch attention by replaying the query/key computation, including rotary positional embeddings, on the exact hidden states encountered during the normal forward pass. The production SDPA computation was preserved. Attention was averaged over 12 heads, CLS/register keys were excluded, and patch weights were spatially normalized. The retained patch mass before normalization was reported separately. We contrasted this backbone diagnostic with four fixed AFA spatial pooling maps from the same checkpoint and inputs. All maps shared a logarithmic attention-relative-to-uniform scale, clipped between 1× and 8×. Temporal panels retained the same four query identities across all displayed frames; individual slots were not reordered to maximize apparent object correspondence.

Supporting figures compared the same CLS diagnostic in the two fine-tuned backbones and the final ACT cross-attention of the two policies. The ACT comparison displayed dense token-position weights for TurboVLA and a projection through tokenizer maps for AFA, with conditional spatial normalization and absolute camera attention mass reported separately. These diagnostics do not constitute a causal tokenizer ablation: the policies also differ in VLI versus early-BERT interaction, and the AFA projection precedes subsequent token mixing. Instrumented and uninstrumented predictions were compared on every selected observation for both policies, obtaining zero maximum absolute difference. All displayed RGB inputs were verified identical across the two policies.

## 14. Tavole curate per il testo principale e l'appendice

Alla raccolta sistematica iniziale di 60 osservazioni è stata affiancata una selezione illustrativa per l'impaginazione della tesi. La nuova raccolta comprende 21 osservazioni di sei episodi, con entrambe le camere, e produce quattro candidate per il testo principale e otto tavole di appendice. I frame sono stati scelti dopo l'ispezione di contact sheet RGB per rendere leggibili interazioni e cambiamenti di scena; non sono campioni casuali o coppie con stato iniziale controllato. L'analisi iniziale e i suoi conteggi restano distinti da questa selezione.

Le sequenze mantengono gli stessi ID degli slot senza riassegnazioni temporali. Le mappe spaziali usano la medesima scala logaritmica 1×–8× sul livello uniforme, con opacità massima 0.62, e preservano l'intera immagine preprocessata con il padding. I pannelli query→word dell'appendice riportano probabilità assolute mediate sulle 12 head, su scala comune 0–0.5 verificata per non saturare alcun valore osservato. Tutti i token di testo sono conservati. La matrice linguistica è mostrata una volta accanto a mappe spaziali a tre tempi: l'invarianza delle sue distribuzioni a parità di istruzione e camera è stata verificata, con differenza massima 0.0.

La [guida alla selezione delle figure](thesis/README.md) indica quali due o tre tavole privilegiare nel testo. Le [didascalie complete in inglese](thesis/figures/CAPTIONS.md), la [selezione esplicita dei frame](thesis/selection.json) e il [manifest delle figure](thesis/figures/figure_manifest.json) documentano il contenuto di ogni tavola. Il nuovo script `plot_thesis_tokenizer.py` effettua solo rendering dei tensori estratti; l'opzione `--selection-manifest` dello script di estrazione permette di riprodurre gli stessi frame.


## 15. Full-episode analysis and post-hoc figure selection

We performed offline replay of every observation from twelve selected real-world episodes: two recorded policy failures and two verified demonstrations for each of the three tasks. This yielded 4,117 observations per checkpoint (2,177 failure observations and 1,940 verified-demonstration observations), with synchronized front and wrist camera inputs. Both the final TurboVLA-AFA early-BERT checkpoint and the fine-tuned TurboVLA VLI checkpoint processed the same observations, states and instructions. This covers the selected episodes completely, rather than the entire demonstration dataset. The verified episodes were drawn from the training dataset and illustrate correct recorded behavior; they are not measured closed-loop policy successes. Inference used evaluation mode, FP32 without AMP or TF32, and batch size one. No model parameters were updated and no robot actions were executed.

For every frame, we saved final-layer DINOv3 CLS-to-patch attention, final-layer ACT cross-attention, and, for AFA, the sixteen spatial pooling maps, competitive routing weights and geometric coordinates for each camera. CLS attention was reconstructed with the model's rotary positional embeddings without replacing the production SDPA computation. Agreement with eager attention was checked numerically. The maximum discrepancy was below 2 × 10⁻⁷ for both checkpoints. Instrumented and uninstrumented predictions were compared at the beginning, midpoint and end of every episode, giving 36 checks per checkpoint and zero maximum absolute prediction difference. Video synchronization error was below 9 × 10⁻⁷ seconds.

After processing all frames, four fixed query identities were selected separately for each episode and camera. An explicitly editorial score combined spatial concentration, attention mass on the image content, temporal continuity and ACT query weight, with a redundancy penalty between selected spatial maps. We then selected eight chronological observations spread across a progress variable combining elapsed time and cumulative robot-state movement, favoring RGB sharpness and map legibility within each interval. The resulting query identities were held fixed across the sequence, and identical frame indices were used for both policies. All frame-level slot scores, selected indices and archive hashes are retained. Since this post-hoc selection favors readable AFA maps, it supports illustrative interpretation rather than an unbiased quantitative claim of superiority over TurboVLA.

For display, the 32-pixel top and bottom preprocessing bands were cropped after inference and map interpolation, yielding 256 × 192 RGB cells. Model inputs remained 256 × 256. Spatial distributions retained their original 256-patch normalization; no renormalization on the cropped content or independent per-image contrast enhancement was applied. All heatmaps use a shared logarithmic attention-relative-to-uniform scale, clipped between 1× and 8×. Figures contain image panels only, with query identities and row definitions supplied in separate captions. Optional minimally labeled versions are provided for inspection.

Token panels show RGB, the AFA checkpoint's DINO CLS diagnostic and four individual AFA pooling maps. Policy panels show RGB, DINO CLS attention from each checkpoint and final ACT attention from each checkpoint. TurboVLA ACT weights are displayed at dense token positions; AFA ACT weights are projected through spatial pooling maps, conditioned on the camera. This projection precedes subsequent token mixing and does not constitute a complete attention rollout or causal attribution. Because the checkpoints also differ in VLI versus early-BERT interaction, the paired comparison is not an isolated tokenizer ablation.

See [the full-episode protocol](full_episodes/README.md), [selection and coverage](full_episodes/selection.json), and [captions](full_episodes/CAPTIONS.md) for exact conventions.


## 16. Full diagnostic study of the success-grasp rollouts

We additionally analyzed both episodes in the user-provided real-world success-grasp recording, `Giannipe/rollout_tokenizer_successgrasp_task3_20261116_160533`, pinned to revision `3c0959dc0a994db91602a5152ac8f480a9f91cd2`. The episodes contain 349 and 357 observations at 30 Hz for the instruction “Grab the peg and push the red cube”. Every observation was processed by both fine-tuned checkpoints with synchronized front and wrist cameras, yielding 706 observations and 1,412 camera views per policy. The recording name identifies a reported successful grasp; full pushing-task completion was not inferred from attention patterns or automatically annotated.

Beyond final-layer diagnostics, we recorded CLS-to-key weights for all twelve DINO layers and twelve heads, dense patch features, all three ACT decoder layers with eight heads and twelve action queries, and every AFA query and spatial pooling distribution. For each AFA token, we retained its geometric encoding, projected pooled semantic encoding and fused output, and verified exact additive reconstruction of the fused representation on every frame. We also recorded early-BERT query-to-word attention, query vectors before and after conditioning, and post-AFA-transformer representations and self-attention. The producer forward pass remained unchanged; additional weights were obtained by separate replay. Predictions with and without diagnostic recording matched exactly on all 706 observations for each checkpoint.

We fitted separate three-component PCA projections to the geometric, semantic, fused and post-transformer representations using all sixteen tokens from every observation of both episodes, separately per camera. Fixed bases, deterministic signs and shared 2nd–98th percentile color limits were retained within each space and camera. Token colors were displayed at their predicted keypoints or projected through the original competitive-routing responsibilities. These colors visualize continuous embedding variation, rather than annotated categories or validated object clusters. Dense DINO features used a common basis across the two checkpoints within each camera, calibrated on a deterministic stride-16 frame sample excluding padding patches. Bases from different representation spaces are not directly comparable.

For static presentation, twelve chronological observations were selected per episode and camera, one from each of twelve equal temporal bins, favoring map legibility and RGB sharpness. All sixteen query identities were retained across four temporal panels, and the same observation indices were used in cross-policy comparisons. Full slot atlases and routing-weighted RGB views were exported at six of these times; head, layer and action-query atlases were exported at a representative middle time. An offline interactive viewer exposes every intermediate observation and all sixteen slots, allowing inspection beyond the illustrative static selection. Display cropping removes only the 32-pixel preprocessing bands above and below the image, after inference, without spatial renormalization.

Query-to-word distributions and conditioned query vectors were temporally invariant for the fixed instruction and camera in these recordings, with zero measured maximum difference. This follows their text-conditioned construction and should not be interpreted as image-dependent linguistic tracking. Numerical diagnostics, full-frame archive hashes, shared-scale figures and captions are provided in the [success-grasp collection](successgrasp/README.md). The paired VLI/early-BERT comparison remains descriptive rather than an isolated tokenizer ablation, and offline command agreement is not a closed-loop success metric.


## 17. Complete diagnostics on verified cube and plush demonstrations

We extended the complete diagnostic protocol to two verified training demonstrations for each of the cube-placement and plush-placement tasks from `Giannipe/Multi-task-verified`, pinned to revision `9bd90bef6481029a3e00e3527c8f67ac04bc5b85`. Cube episodes 0 and 49 contain 319 and 249 observations (568 total), while plush episodes 50 and 99 contain 365 and 295 observations (660 total). These are the same episode pairs used in the earlier exploratory analyses. Every frame was processed by both fine-tuned checkpoints with synchronized front and wrist observations, yielding 1,228 observations per checkpoint across four demonstrations. The newly added collections contain verified demonstrations only.

We retained the complete sixteen-slot diagnostic suite used for the peg recording: all twelve DINO layers and heads, three ACT decoder layers and eight heads, twelve action queries, early-BERT query-to-word attention, learned and conditioned queries, competitive spatial routing, geometric coordinates, pooled semantic features, geometric/semantic projected components, fused tokens and the post-AFA-transformer representations and self-attention. Separate full-matrix QK replay matched the eager attention formulation for all DINO layers in both checkpoints, while the production SDPA computation was preserved. Instrumented and recording-disabled predictions were compared on every observation, without updating weights or executing actions.

Static layouts use twelve chronological observations per episode and camera, selected after full inference within equal temporal bins for sharpness and map legibility. All sixteen fixed slot identities are displayed, including diffuse or redundant slots, and identical observation indices are used in paired TurboVLA/AFA figures. The offline viewers retain every intermediate frame and query for subsequent inspection. Shared attention/routing display scales, crop-only removal of preprocessing padding and separate captions follow the peg protocol. Embedding PCA bases are fitted per task collection and camera across both selected demonstrations; PCA colors should therefore not be interpreted as corresponding semantic categories between cube, plush and peg.

The verified recordings illustrate correct behavior from the training dataset; they do not measure closed-loop policy success on new executions. The paired checkpoints also differ in VLI versus early-BERT interaction, so the visual comparison does not isolate the tokenizer causally. Dataset selection, frame coverage, numerical checks and figures are documented in the [cube collection](cube_verified/README.md), [plush collection](plush_verified/README.md) and [combined index](VERIFIED_DEMOS.md).


## 18. Competitive routing appendix

We separated a normalization diagnostic from a trained-model comparison. For six previously selected observations (two per task), the competitive early-BERT checkpoint supplied identical query–patch logits for query-softmax responsibilities, subsequent spatially normalized competitive pooling, and an independent spatial-softmax diagnostic. Only the normalization formula changes in this controlled visualization; the independent branch is not a separately trained policy, and no counterfactual actions are executed. Individual query maps share the 1×–8× attention-lift scale and retain four preselected numeric identities (0,4,8,12); all sixteen queries are available in supporting atlases.

Additional panels compare the competitive single-head early-BERT checkpoint with the final noncompetitive twelve-head VLI checkpoint on identical recorded inputs. This comparison also changes architecture and language interaction, and therefore is not an isolated routing ablation. Patch winners are computed from spatially normalized geometric pooling maps in both checkpoints. Multi-head geometric maps do not exhaust the head-specific semantic pooling weights, and equal numeric query indices do not imply semantic alignment across models. Full definitions and captions are in [routing_appendix](routing_appendix/README.md).
