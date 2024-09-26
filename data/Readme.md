# Data Source

Original data downloaded from: https://en.eustat.eus/estadisticas/tema_37/opt_0/tipo_11/temas.html

The included `targets` workflow prepares the data for analysis.

Use the included `default.nix` to build the environment and run the `targets`
workflow in. From the project’s root:

```
cd data/ && nix-shell default.nix --run "Rscript -e 'targets::tar_make()'"
```
