# Benchmarks for TensorWaves

The [`benchmark`](.) folder contains a few pre-generated amplitude modules in
the form of YAML recipe files, produced by the
[`expertsystem`](http://expertsystem.readthedocs.io/). They can be run wih the
benchmark [`run.py`](./run.py) script as follows:

```shell
python run.py Jpsi_f0_gamma.pi0.pi0_heli.yml
```

There is a default number of phasespace events and toy Monte Carlo data events
preset in the script, but you can also specify this with e.g. `--phsp=4e4` and
`--data=2000`, respectively.

Other amplitude model recipe files can be created with the
[`create_recipe.py`](./create_recipe.py) script.
