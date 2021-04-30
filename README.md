# Differential equations with neural network terms for ion channel modelling

Source code associated with an [article in Frontiers of Physiology](.) by [Chon Lok Lei](https://chonlei.github.io/) and [Gary R. Mirams](https://www.maths.nottingham.ac.uk/plp/pmzgm/).


Requirements
------

To run the code within this repository requires [Python 3.5+](https://www.python.org/) with the following dependencies

- [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq)
- [`pints`](https://github.com/pints-team/pints)
- [`seaborn`](https://seaborn.pydata.org)

which can be installed via
```
$ pip install -r requirements.txt
```


Train the models
------

The following codes re-run the training for the models.

#### Synthetic data studies (no discrepancy)

- NN-f: [`train-s1.py`](./train-s1.py)
- NN-d: [`train-s2.py`](./train-s2.py)

#### Synthetic data studies (with discrepancy)

- Candidate model: [`train-d0.py`](./train-d0.py)
- NN-f: [`train-d1.py`](./train-d1.py)
- NN-d: [`train-d2.py`](./train-d2.py)

#### Experimental data

- NN-f: [`train-r1.py`](./train-r1.py)
- NN-d: [`train-r2.py`](./train-r2.py)

Their trained results are stored in directories `s1`, `s2`, `d1`, etc.


Main figures
------

To re-run and create the main figures, use:
- Figure 2: [`figure-1.py`](./figure-1.py)
- Figure 3: [`figure-2.py`](./figure-2.py)
- Figure 4: [`figure-3.py`](./figure-3.py)
- Figure 5: [`figure-4.py`](./figure-4.py)
- Figure 6: [`figure-5.py`](./figure-5.py)
- Figure 7: [`figure-6.py`](./figure-6.py).

These generate figures in directories `figure-1`, `figure-2`, etc.


Supplementary figures
------

To re-run and create the supplementary figures, use:
- Figure S2: [`figure-0-s.py`](./figure-0-s.py)
- Figure S3: [`figure-2-s.py`](./figure-2-s.py)
- Figure S4: [`figure-3-s.py`](./figure-3-s.py)
- Figure S5: [`figure-4-s.py`](./figure-4-s.py)
- Figure S6: [`figure-1-s2.py`](./figure-1-s2.py)
- Figure S7: [`figure-1-s1.py`](./figure-1-s1.py)

These generate figures in directories `figure-2-s`, `figure-3-s`, etc.
