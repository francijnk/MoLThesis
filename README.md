# Code for Master of Logic Thesis  
### *How Cultural Transmission and Individual Learning Mechanisms Shape the Emergence of Compositionality*

This repository contains the code, and a Jupyter Notebook to generate the plots used in the Master of Logic thesis *"How Cultural Transmission and Individual Learning Mechanisms Shape the Emergence of Compositionality."*  

---

### Experiments

| Experiment | Title |
|-------------|--------|
| 1 | Transmission Modes |
| 2 | Population Size |
| 3 | Social Network Structure |
| 4 | Rate of Replacement |
| 5 | Regulating the Impact of New Evidence |
| 6 | Prior Strength |
| 7 | Heterogeneous Population |
| 8 | Social Network Structure Revisited |
| 9 | Holistic Starting Language |
| 10 | Interaction Regulating Impact of New Evidence & Prior Strength |

---

## Running Simulations

A single simulation (for example, Experiment 1: *Transmission Modes*) can be executed using:

```bash
python sim_extra.py --mode "sample" --group_round 50 --turnover_round 0 --pop_size 25 --run_id 5 --network_type "fully-connected" --replace True --alpha 0.0
```

---

## Reproducing Experiments

All scripts within the folder can be used to replicate the full experiments.  
Parameter txt-files can be created using the Jupyter notebook:

```
get_params.ipynb
```
---



