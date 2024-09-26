library(targets)

tar_option_set(packages =
  c(
    "dplyr"
  , "readr"
  , "janitor"
  )
)

list(
  tar_target(
    paths_to_data,
    list.files(path = "Microdatos_PRA_2019_2024/Microdatos_PRA_2019_2024/",
               pattern = ".*2023.*.csv",
               recursive = TRUE,
               full.names = TRUE)
  )

, tar_target(
    raw_pra_2023,
    lapply(X = paths_to_data,
           FUN = read.csv,
           sep = ";") |>
    Reduce(f = rbind, x = _)
  )

, tar_target(
    english_col_names,
    readRDS("english_col_names.rds")
  )

, tar_target(
    pra_2023,
    setNames(raw_pra_2023, english_col_names) |>
    clean_names() |>
    filter(reference_quarter == 1) |>
    select(-level_of_studies_completed) # This variable is completely empty
  )

, tar_target(
    pra_2023_csv,
    write.csv(pra_2023,
              "pra_2023.csv",
              row.names = FALSE)
  )
)

