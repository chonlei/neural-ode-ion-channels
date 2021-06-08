# Neural network differential equations for ion channel modelling

Source code associated with an [article in Frontiers of Physiology](.) by [Chon Lok Lei](https://chonlei.github.io/) and [Gary R. Mirams](https://www.maths.nottingham.ac.uk/plp/pmzgm/).

![Model structures used in this repository](https://raw.githubusercontent.com/chonlei/neural-ode-ion-channels/main/model-structure/model-structure-2.svg?token=AGL42DUH3JXYLYLGOWOKL23AUXFOK)

From _left_ to _right_ shows the original Hodgkin-Huxley model (candidate model), the activation modelled using a neural network (NN-f), the activation with a neural network discrepancy term (NN-d), and the activation modelled with a three-state model (ground truth used in synthetic data studies with discrepancy).


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


Main figures and tables
------

To re-run and create the main figures and tables, use:
- Figure 2: [`figure-1.py`](./figure-1.py)
- Figure 3: [`figure-2.py`](./figure-2.py)
- Figure 4: [`figure-3.py`](./figure-3.py)
- Figure 5: [`figure-4.py`](./figure-4.py)
- Figure 6: [`figure-5.py`](./figure-5.py)
- Figure 7: [`figure-6.py`](./figure-6.py)
- Figure 8: [`figure-7.py`](./figure-7.py).
- Table 1: [`table-1.py`](./table-1.py).

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


Others
------

- `data`: Contains the experimental data from [Beattie et al. 2018](https://doi.org/10.1113/JP276068).
- `model-structure`: Contains Markov diagrams/schematics for the models.
- `test-protocols`: Contains time series files for various voltage-clamp protocols from [Beattie et al. 2018](https://doi.org/10.1113/JP276068) and [Lei et al. 2019a](https://doi.org/10.1016/j.bpj.2019.07.029) [& b](https://doi.org/10.1016/j.bpj.2019.07.030).


Acknowledging this work
------

If you publish any work based on the contents of this repository please cite ([CITATION file](.)):

[Place holder]
