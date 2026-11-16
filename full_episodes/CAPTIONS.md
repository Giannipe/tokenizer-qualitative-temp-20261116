# Didascalie e ordine delle righe

Le immagini clean non contengono testo, bordi neri di preprocessing o titoli. Le versioni minimal_labels aggiungono solo nomi delle righe e timestamp.

La prima riga è sempre RGB. Nei pannelli tokens seguono DINO CLS→patch del checkpoint AFA e le quattro query elencate. Nei pannelli policies seguono DINO CLS TurboVLA, DINO CLS AFA, ACT TurboVLA, ACT AFA proiettata tramite le mappe del tokenizer.

Le scale sono condivise: log2 dell’attenzione rispetto all’uniforme su 256 patch, saturato a 8×. I valori sotto uniforme sono trasparenti. Si elimina solo il padding nel display dopo aver calcolato e interpolato le mappe; non si rinormalizza sulla parte ritagliata.

La selezione è post-hoc e illustrativa: usa tutte le osservazioni, favorisce leggibilità e differenziazione delle query AFA, e non misura un vantaggio statistico o causale rispetto alla baseline. Gli stessi frame vengono usati per entrambi i checkpoint. Non si cambiano query fra colonne per simulare un tracking migliore.

## failure_task1_ep000_front

Task 1, failure, episodio 0, camera frontale. Frame: [17, 41, 64, 87, 99, 131, 169, 199]. Query AFA, dall’alto al basso: [8, 15, 3, 14]. Tutti i frame dell’episodio sono stati elaborati prima di questa selezione.

## failure_task1_ep000_wrist

Task 1, failure, episodio 0, camera polso. Frame: [0, 30, 54, 72, 119, 141, 169, 193]. Query AFA, dall’alto al basso: [15, 13, 7, 6]. Tutti i frame dell’episodio sono stati elaborati prima di questa selezione.

## failure_task1_ep001_front

Task 1, failure, episodio 1, camera frontale. Frame: [5, 27, 45, 117, 127, 189, 221, 299]. Query AFA, dall’alto al basso: [12, 15, 5, 14]. Tutti i frame dell’episodio sono stati elaborati prima di questa selezione.

## failure_task1_ep001_wrist

Task 1, failure, episodio 1, camera polso. Frame: [3, 28, 49, 95, 156, 211, 235, 299]. Query AFA, dall’alto al basso: [15, 13, 7, 6]. Tutti i frame dell’episodio sono stati elaborati prima di questa selezione.

## failure_task2_ep000_front

Task 2, failure, episodio 0, camera frontale. Frame: [27, 39, 71, 127, 184, 219, 251, 281]. Query AFA, dall’alto al basso: [11, 15, 3, 13]. Tutti i frame dell’episodio sono stati elaborati prima di questa selezione.

## failure_task2_ep000_wrist

Task 2, failure, episodio 0, camera polso. Frame: [28, 39, 79, 145, 183, 227, 267, 276]. Query AFA, dall’alto al basso: [15, 6, 13, 7]. Tutti i frame dell’episodio sono stati elaborati prima di questa selezione.

## failure_task2_ep001_front

Task 2, failure, episodio 1, camera frontale. Frame: [39, 51, 117, 220, 231, 304, 381, 415]. Query AFA, dall’alto al basso: [15, 8, 3, 14]. Tutti i frame dell’episodio sono stati elaborati prima di questa selezione.

## failure_task2_ep001_wrist

Task 2, failure, episodio 1, camera polso. Frame: [27, 103, 167, 218, 286, 307, 374, 435]. Query AFA, dall’alto al basso: [15, 7, 13, 12]. Tutti i frame dell’episodio sono stati elaborati prima di questa selezione.

## failure_task3_ep000_front

Task 3, failure, episodio 0, camera frontale. Frame: [7, 97, 119, 225, 323, 449, 467, 551]. Query AFA, dall’alto al basso: [11, 15, 3, 14]. Tutti i frame dell’episodio sono stati elaborati prima di questa selezione.

## failure_task3_ep000_wrist

Task 3, failure, episodio 0, camera polso. Frame: [0, 97, 130, 263, 304, 433, 505, 527]. Query AFA, dall’alto al basso: [15, 12, 7, 13]. Tutti i frame dell’episodio sono stati elaborati prima di questa selezione.

## failure_task3_ep001_front

Task 3, failure, episodio 1, camera frontale. Frame: [22, 51, 72, 129, 177, 201, 223, 251]. Query AFA, dall’alto al basso: [15, 5, 12, 14]. Tutti i frame dell’episodio sono stati elaborati prima di questa selezione.

## failure_task3_ep001_wrist

Task 3, failure, episodio 1, camera polso. Frame: [30, 62, 67, 129, 155, 191, 237, 279]. Query AFA, dall’alto al basso: [15, 7, 12, 13]. Tutti i frame dell’episodio sono stati elaborati prima di questa selezione.

## verified_demo_task1_ep000_front

Task 1, verified_demo, episodio 0, camera frontale. Frame: [30, 57, 68, 140, 153, 197, 245, 274]. Query AFA, dall’alto al basso: [11, 6, 12, 13]. Tutti i frame dell’episodio sono stati elaborati prima di questa selezione.

## verified_demo_task1_ep000_wrist

Task 1, verified_demo, episodio 0, camera polso. Frame: [19, 48, 66, 109, 173, 223, 245, 309]. Query AFA, dall’alto al basso: [15, 13, 6, 12]. Tutti i frame dell’episodio sono stati elaborati prima di questa selezione.

## verified_demo_task1_ep049_front

Task 1, verified_demo, episodio 49, camera frontale. Frame: [3, 53, 61, 107, 153, 183, 201, 245]. Query AFA, dall’alto al basso: [15, 11, 12, 10]. Tutti i frame dell’episodio sono stati elaborati prima di questa selezione.

## verified_demo_task1_ep049_wrist

Task 1, verified_demo, episodio 49, camera polso. Frame: [5, 29, 61, 124, 145, 157, 199, 243]. Query AFA, dall’alto al basso: [15, 6, 7, 13]. Tutti i frame dell’episodio sono stati elaborati prima di questa selezione.

## verified_demo_task2_ep050_front

Task 2, verified_demo, episodio 50, camera frontale. Frame: [1, 68, 91, 145, 159, 229, 263, 311]. Query AFA, dall’alto al basso: [15, 11, 12, 13]. Tutti i frame dell’episodio sono stati elaborati prima di questa selezione.

## verified_demo_task2_ep050_wrist

Task 2, verified_demo, episodio 50, camera polso. Frame: [15, 61, 69, 139, 187, 244, 261, 333]. Query AFA, dall’alto al basso: [15, 7, 13, 12]. Tutti i frame dell’episodio sono stati elaborati prima di questa selezione.

## verified_demo_task2_ep099_front

Task 2, verified_demo, episodio 99, camera frontale. Frame: [9, 64, 103, 119, 145, 193, 218, 269]. Query AFA, dall’alto al basso: [11, 6, 12, 13]. Tutti i frame dell’episodio sono stati elaborati prima di questa selezione.

## verified_demo_task2_ep099_wrist

Task 2, verified_demo, episodio 99, camera polso. Frame: [33, 46, 68, 129, 167, 211, 223, 287]. Query AFA, dall’alto al basso: [15, 7, 13, 12]. Tutti i frame dell’episodio sono stati elaborati prima di questa selezione.

## verified_demo_task3_ep100_front

Task 3, verified_demo, episodio 100, camera frontale. Frame: [34, 61, 81, 193, 259, 331, 345, 395]. Query AFA, dall’alto al basso: [15, 11, 14, 12]. Tutti i frame dell’episodio sono stati elaborati prima di questa selezione.

## verified_demo_task3_ep100_wrist

Task 3, verified_demo, episodio 100, camera polso. Frame: [11, 55, 113, 185, 241, 343, 353, 427]. Query AFA, dall’alto al basso: [15, 6, 9, 7]. Tutti i frame dell’episodio sono stati elaborati prima di questa selezione.

## verified_demo_task3_ep149_front

Task 3, verified_demo, episodio 149, camera frontale. Frame: [9, 34, 89, 107, 170, 197, 219, 267]. Query AFA, dall’alto al basso: [11, 6, 3, 2]. Tutti i frame dell’episodio sono stati elaborati prima di questa selezione.

## verified_demo_task3_ep149_wrist

Task 3, verified_demo, episodio 149, camera polso. Frame: [0, 43, 73, 123, 157, 199, 224, 253]. Query AFA, dall’alto al basso: [15, 13, 9, 6]. Tutti i frame dell’episodio sono stati elaborati prima di questa selezione.
