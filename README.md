# Detection-Epileptic-Seizures


- sistema de clasificació que digui si en aquell instant has tingut un atac epileptic o no
- Dividir en finestres d'1 segon el meu recording, per cadascuna d'aquetes finestres fer un classificador 

- NO des del punt de vista de detector d'anomalies, fer clasificador!

- opcio 1: Fer una xarxa, RGB amb una primera capa que rebi com a entrada (Ncanals, nhidden, __  ) fusio a nivell de input --> Fusion of EEG channels at input level (like image CNNs)

- opcio 2: en lloc de primer resumir i despres extraure informació, fusio a nivel de feature -->Fusion of EEG channels at feature level (after extracting relevant features for each sensor)

- Temporal information: ella quiere el modelo backbone (slide 9 powerpoint hacer lo de la derecha, izquierda NO)

- Los datos están fragmentados en múltiples archivos .npz (señal) y .parquet (metadata) por paciente.
parquet contiene paciente y numero de registro

- en parquet filename_interval NO NOS INTERESA, NOS INTERESA EL GLOBAL_INTERVAL

- Cuando añadamos el tiempo ver si la LSTM ayuda a predecir mejor
/import/fhome/maed/EpilepsyDataSet/
