# Istruzioni per il server

## Installazione

Le ho dato accesso al repository. In alternativa può scaricare lo zip della release v1.0 dalla pagina del repository.

```bash
git clone --branch v1.0 https://github.com/paolominini/lexical-preference-shift.git
cd lexical-preference-shift
```

Serve Python 3.12. Crei l'ambiente virtuale:

```bash
python3.12 -m venv .venv
```

Poi controlli la versione di CUDA indicata da `nvidia-smi`. Se è inferiore a 13.0, installi prima torch con questo comando:

```bash
.venv/bin/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
```

Poi, in ogni caso:

```bash
.venv/bin/pip install -r requirements-cuda.txt
```

Una versione diversa di torch o di CUDA va bene lo stesso. La run 1.4B serve proprio a controllare che i risultati coincidano con i miei.

## Scelta della taglia

Lanci una volta il controllo dell'ambiente:

```bash
.venv/bin/python scripts/00_check_env.py
```

Il controllo scrive `reports/step_00_env.md`. Nella sezione D c'è una tabella con tutte le taglie e indica quali entrano su questa
macchina. In `config.yaml` modifichi solo `run.size` e, se serve, `paths.models_dir`. La prego di non cambiare nient'altro nel file.

Requisiti indicativi:

- 12b: GPU da almeno 40 GB, circa 160 GB di disco, circa 32 GB di RAM.
- 6.9b: GPU da almeno 24 GB, circa 90 GB di disco, 16-20 GB di RAM.
- 1.4b: GPU da almeno 6 GB, circa 20 GB di disco.

Se le risorse sono limitate:

- Con una GPU tra 24 e 40 GB usi `run.size: "6.9b"`. Con meno di 24 GB usi `"2.8b"`.
- Se la quota della home è piccola, imposti `paths.models_dir` sul percorso assoluto di una cartella su un disco dati.
- Se il disco non basta per i sei modelli, usi una taglia più piccola.
- Con uno scheduler o con limiti di tempo, lanci gli stessi comandi dentro un job. Se il job viene interrotto, rilanci 04: riprende da dove si era fermato.
- Lo script 00b deve scaricare i modelli da Hugging Face. Se i nodi di calcolo non hanno internet, lo lanci sul nodo di login.

## Esecuzione

Prima con `run.size: "1.4b"`, poi con la taglia scelta. Per ogni taglia:

```bash
.venv/bin/python scripts/00b_download_models.py
.venv/bin/python scripts/04_run_scoring.py --smoke
.venv/bin/python scripts/04_run_scoring.py
.venv/bin/python scripts/04b_gate_and_consolidate.py
```

Il comando con `--smoke` è una prova breve e non salva nulla. Sul mio Mac la run completa della 1.4B richiede circa 50 minuti.
Se 04 si ferma, basta rilanciare lo stesso comando.
Se si ferma sempre nello stesso punto, per esempio per mancanza di memoria, non insista: mi mandi il messaggio e passiamo a una taglia più piccola.
Se 04b si ferma con un errore, mi mandi il messaggio che stampa.
Se 04b stampa un avviso sullo spot check, non serve fare nulla: mi mandi i file come al solito.

## File da mandarmi

Dopo ogni taglia, prima di lanciare la successiva, perché la taglia successiva sovrascrive le tabelle in `reports/step_04_tables/`.
Per la 1.4B:

```bash
zip -r risultati_1.4b.zip results/scores_1.4b.parquet results/scores_1.4b_base_nobos.parquet reports/step_04_tables reports/step_00_env.md
```

Dopo avermi mandato i file della 1.4B può cancellare i modelli 1.4B (circa 17 GB) per fare spazio alla taglia successiva.
Si trovano nella cartella indicata in `paths.models_dir`, se l'ha impostata, altrimenti nella cache di Hugging Face (di solito `~/.cache/huggingface/hub`).
Sono le cartelle `models--EleutherAI--pythia-1.4b` e `models--ContextualAI--archangel_*_pythia1-4b`.

Per l'altra taglia, lo stesso comando con la taglia al posto di `1.4b` (per esempio `12b`).

Per qualsiasi problema mi scriva per email. Non serve aprire issue o pull request.
