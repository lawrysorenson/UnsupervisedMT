# WeaklySupervisedMT

## Install Dependancies

Run the following to install dependencies within a virtual conda environment. For simplicity, this install the cpu only version of pytorch. Reinstall pytorch if GPU usage is desired.

```
conda env create -f environment.yml
```

Run the following to activate the environment.

```
conda activate unsup
```

## Running the demo model

Run the following to demo a trained model. As this is intended as a proof of concept, sentences and weights file are hard coded.

```
python3 demo.py
```

By default, the script will look for the weights file `demo-weights`. The `[FA]` and `[EN]` tags determine the desired target language for the translation, representing Persian, and English respectively.

## Training the model

### Required Data

1. Data directories

Create the following directories in this directory that will be referenced by the code during training.

```
data/cleaning/
data/dicts/
data/output/
data/split/
data/tokenizers/
```

2. Bilingual Dictionary

Place a bilingual dictionary in `data/dicts/fa-en.txt` to be used as the seed for literal translations. During our testing, we employed full the Persian-English dictionary listed by MUSE, located [here](https://dl.fbaipublicfiles.com/arrival/dictionaries/fa-en.txt).

3. Cleaned Data

Place a plain text parallel corpus in the `data/cleaning/` directory in files `corpusname.en-US` and `corpusname.fa-IR`. The current code base was used on fully parralel corpus that were split so that parallel sentences were not included in both languages.

To perform this split, edit `sel.py` to specify the corpus name and the desired amount of parallel data to be used during training. These values are stored in `corename` and `parallel_size`, respectively.

Then run
```
python3 sel.py
```

This will randomly split the data and place it in the `data/split/` directory.

4. Train a tokenizer

Edit `corename` in `tokenizer.py` to specify the desired corpus, then run `tokenizer.py` to train a tokenizer on the train split of that corpus. This will save a tokenizer file in `data/tokenizers/`.

5. Train models

Edit `corename` in `back.py` and `unsup.py` to specify the training corpus, then run both files one at a time. `back.py` will train a baseline model with iterative backtranslation, while `unsup.py` will use the weakly supervised approach.

The `*.sh` files are provided as example scripts for running the tasks with SLURM commands. Likewise, the training scripts will look for a slurm output file to determine their job id, or they will default to an id of 1. This id determines their output files in `data/output/`.

`back.py` will create a `temp-weights` file in the directory it is run in that can be deleted once it has finished running.