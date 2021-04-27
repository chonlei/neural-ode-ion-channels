# Differential equations with neural networkterms for ion channel modelling

Code associated with a [Frontiers of Physiology paper](.) by Chon Lok Lei and Gary R. Mirams.


Requirements
------

To run the code within this repository requires

- Python 3.5+
- torchdiffeq
- pints
- seaborn

which can be installed via
```
$ pip install -r requirements.txt
```


Train the models
------

The following codes re-run the training for the models.

#### Synthetic data studies (no discrepancy)

- NN-f: [`s1.py`](./s1.py)
- NN-d: [`s2.py`](./s2.py)

#### Synthetic data studies (with discrepancy)

- candidate model: [`d0-m.py`](./d0-m.py)
- NN-f: [`d1.py`](./d1.py)
- NN-d: [`d2.py`](./d2.py)

#### Experimental data

- NN-f: [`r1.py`](./r1.py)
- NN-d: [`r2.py`](./r2.py)


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
- Figure S2: [`figure-2-s.py`](./figure-2-s.py)
- Figure S3: [`figure-3-s.py`](./figure-3-s.py)
- Figure S4: [`figure-4-s.py`](./figure-4-s.py)
- Figure S5: [`figure-1-s2.py`](./figure-1-s2.py)
- Figure S6: [`figure-1-s1.py`](./figure-1-s1.py)

These generate figures in directories `figure-2-s`, `figure-3-s`, etc.
