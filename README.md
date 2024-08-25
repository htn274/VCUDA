# Variational Causal Discovery Unconstrained by Acyclicity (VCUDA)

This is the implementation for our paper: [Scalable Variational Causal Discovery
Unconstrained by Acyclicity](https://arxiv.org/pdf/2407.04992), accepted at ECAI 2024.

<p align="center" markdown="1">
    <img src="https://img.shields.io/badge/Python-3.8-green.svg" alt="Python Version" height="18">
    <a href="https://arxiv.org/pdf/2407.04992"><img src="https://img.shields.io/badge/arXiv-2307.07973-b31b1b.svg" alt="arXiv" height="18"></a>
</p>


<p align="center">
  <a href="#setup">Setup</a> •
  <a href="#structure">Structure</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#citation">Citation</a>
</p>

## Setup 

```bash
conda env create -n vcuda --file env.yml
conda activate vcuda
```

You can use `env-cpu.yml` if you don't have a gpu.

## Structure

- castle/: the modified [gcastle](https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle) package. 
- data/: synthetic and real datasets. Synthetic datasets will be generated when the experiment runs. 
- exp/: code for running the experiments. 
- src/: source code for VCUDA and baselines.

## Experiments

### Synthetic dataset notation

We denote a synthetic dataset as `{ER/SF}_d{nodes}_e{edges}_N{samples}_{noisetype/nonlinear functions}`.

For example: 

- ER_d10_e10_N1000_gauss: generated from the ER graph model of 10 nodes and 10 expected edges (degree of 1) with 1000 samples from the linear model with Gaussian noises.
- SF_d10_e10_N1000_gp: generated from the SF graph model of 10 nodes and 10 expected edges (degree of 1) with 1000 samples from the Gaussian Process. 

### DAG sampling 

The results reported in the paper are saved in `times.csv`. 

To generate Figure 2 in the paper:

```
python exp/dag_sampling/timing.py
```

The plot is saved in `exp/dag_sampling/plot.pdf`.

Add `--run` option to reproduce the result. 

To reproduce the results of differentiable DAG sampling optimization: 

```
python exp/dag_sampling/main.py --run
```

### DAG learning 

The results reported in the paper are saved in folder `exp/dag_learning/results/`. 

To reproduce the results, uncomment the `run: false` for a method that you want to rerun in the config files in `exp/dag_learning/configs/` and run:

```
python exp/dag_learning/run.py --cfg_file linear.yml
```

Use `nonlinear.yml` and `real.yml` to produce the results on nonlinear and real datasets, respectively.

To generate Figure 3, 4:

```
python exp/dag_learning/viz.py --cfg_file linear.yml 
```

The resultant plots are saved in `exp/dag_learning/plot/` following the format `linear/nonlinear-{d nodes}.pdf`

In case you got an error of "Tensors in different devices", refer to this [issue](https://github.com/pytorch/pytorch/issues/111573#issuecomment-1772414407).

## Citation

```
@inproceedings{hoang2024enabling,
      title={Scalable Variational Causal Discovery Unconstrained by Acyclicity}, 
      author={Hoang, Nu and Duong, Bao and Nguyen, Thin},
      year={2024},
      booktitle = {European Conference on Aritificial Intelligence (ECAI)},
}
```
