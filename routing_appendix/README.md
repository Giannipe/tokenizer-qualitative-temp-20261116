# Competitive routing: figure per l’appendice

[Galleria](GALLERY.md) · [Didascalie](CAPTIONS.md)

La figura aggiuntiva consigliata è `front_controlled_slot_maps.pdf`, accanto a `front_normalization_winners.pdf`: la prima mostra concentrazione e sovrapposizione di quattro query fisse, la seconda distingue le assegnazioni prima/dopo normalizzazione spaziale. La versione al polso e gli atlanti di tutti i 16 slot sono materiale di supporto.

## Confronto controllato della normalizzazione

Sul medesimo checkpoint early-BERT competitivo si estraggono gli score S query–patch durante il forward reale. Si confrontano R=softmax_query(S/tau), A_comp=R/(sum_patch R+epsilon) e A_ind=softmax_patch(S/tau), a temperatura tau=0.5. La figura dei vincitori usa la stessa opacità fissa in tutte le righe, senza confondere intensità di attention e confidenza. L’argmax di R coincide con quello degli score S; l’argmax di A_comp può cambiare dopo la rinormalizzazione per query.

A_ind è una trasformazione diagnostica degli stessi score: non è un nuovo checkpoint addestrato senza competizione, né una misura di performance della policy con routing disabilitato. Nessuna predizione viene sostituita o eseguita sul robot. I pesi di pooling competitivi e le responsabilità ricostruite sono verificati contro quelli realmente registrati; gli output con e senza registrazione sono confrontati.

## Confronto tra checkpoint realmente addestrati

Si usa anche il checkpoint finale a 160000 micro-step `turbovla_afa_mh12_vli_no_competitive_multitask_full_80k_20261116`, con 12 head e VLI. Quello competitivo finale usa invece single-head ed early-BERT. Le tavole `trained_models` confrontano queste due varianti sulle identiche osservazioni: non isolano causalmente il competitive routing. Per il confronto omologo dei vincitori si usa argmax delle mappe geometriche A in entrambi i modelli. Nella variante multi-head le mappe geometriche sono fuse dalle head e non esauriscono i pesi del pooling semantico. Non si assume corrispondenza semantica fra query di uguale indice nei due modelli.

## Dati e stile

Sei osservazioni, due per task: cubo e peluche dalle demo verificate, peg dai rollout successgrasp già analizzati. I frame provengono dalla precedente selezione temporale e sono identici per ogni confronto; non sono fasi annotate. La natura diversa delle registrazioni è riportata nel manifest. Scala comune 1×–8× rispetto all’uniforme su 256 patch; quattro ID 0/4/8/12 fissati prima del confronto e tutti i 16 disponibili negli atlanti. Nessun k-means, riassegnazione degli ID o segmentazione annotata. Crop del padding solo nel display; immagini clean senza testo, didascalie separate e versioni labeled per revisione. Gli array, gli score e i checkpoint SHA-256 sono nel manifest. Le mappe non provano segmentazione per oggetto o un miglioramento delle prestazioni.

La selezione dei frame deriva dalle precedenti tavole illustrative, scelte post-hoc anche per leggibilità delle mappe AFA. Il confronto non è una stima imparziale della superiorità fra modelli. [Verifiche numeriche](validation.json).
