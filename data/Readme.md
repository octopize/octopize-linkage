# Data Source

## PRA

Original data downloaded from: https://en.eustat.eus/estadisticas/tema_37/opt_0/tipo_11/temas.html

The included `targets` workflow prepares the data for analysis.

Use the included `default.nix` to build the environment and run the `targets`
workflow in. From the project’s root:

```
cd data/ && nix-shell default.nix --run "Rscript -e 'targets::tar_make()'"
```

## Adult (census)

Original data downloaded from: https://archive.ics.uci.edu/dataset/2/adult


## Student dropout

Original data downloaded from: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success


## Student performance

Original data downloaded from: https://archive.ics.uci.edu/dataset/320/student+performance


## Career change

Original data downloaded from: https://www.kaggle.com/datasets/jahnavipaliwal/field-of-study-vs-occupation


## Chess

Original data downloaded from: https://www.kaggle.com/datasets/datasnaek/chess

