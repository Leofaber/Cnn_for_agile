# Cnn_for_agile

* Avviare il virtual env di python con astropy
    * source /home/baroncelli/python2_virtual_env/bin/activate

* Avviare il file
    * python create_batch_dataset.py
        * Riduce la dimensione delle immagini (100x100) -> (96x96). 
        * Divide le immagini del dataset in batch numpy in codifica binaria. Ogni batch contiene 10000 immagini. Crea inoltre anche i batch delle corrispondenti labels.
        * Ogni batch è quindi un tensore (10000, 100, 100).


* Esportare la variabile PYTHONPATH
    * export PYTHONPATH="${PYTHONPATH}:/opt/DL/tensorflow/lib/python2.7/site-packages/"

* Avviare il file cnn_for_agile.py
