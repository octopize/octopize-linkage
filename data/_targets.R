library(targets)

tar_option_set(packages =
  c(
    "dplyr"
  , "readr"
  , "stringr"
  , "tibble"
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
    c(
      "household_number" = "NUMH",
      "survey_year" = "AENC",
      "reference_quarter" = "TENC",
      "province" = "TERH",
      "capital" = "MUNI",
      "sex" = "SEXO",
      "place_birth" = "LNAC",
      "age" = "EDAD",
      "nationality" = "NACI",
      "education_level" = "LEST",
      "formal_educ_system" = "ENRE",
      "professional_training" = "FOCU",
      "retirement_status" = "SJUB",
      "household_duties" = "SILH",
      "part-time_empl" = "EMPTP",
      "reason_reduced_work" = "JRED",
      "search_work" = "BUSQ",
      "search_reason" = "RBUSQ",
      "search_steps" = "GBUSQ",
      "search_method" = "FBUSQ",
      "search_months" = "MSBUSQ1",
      "availability" = "DISP",
      "relation_to_activity1" = "PRA1",
      "relation_to_activity2" = "PRA2",
      "main_occupation" = "PROF",
      "main_activity" = "RACT",
      "main_prof_situation" = "SPRO",
      "main_sector" = "SECT",
      "contract_type" = "CONTR",
      "hours" = "HTRA",
      "relationship" = "PARE",
      "elevator" = "ELEV"
    )
  )

 , tar_target(
    pra_2023,
    rename(raw_pra_2023, all_of(english_col_names)) |>
    filter(reference_quarter == 1) |>
    select(-education_level) |> # This variable is completely empty
    mutate(elevator = str_replace_all(elevator, ",", ".")) |>
    rowid_to_column() |>
    mutate(matricule = paste0(rowid, "_", household_number))
   )

, tar_target(
    pra_2023_csv,
    write.csv(pra_2023,
              "pra_2023.csv",
              row.names = FALSE)
  )

, tar_target(
    shared_columns,
    c(
      "age",
      "matricule",
      "nationality",
      "place_birth",
      "sex",
      "province"
    )
  )

, tar_target(
    pra_A,
    select(
      pra_2023,
      all_of(shared_columns),
      household_duties,
      relation_to_activity1,
      relation_to_activity2,
      )
  )


, tar_target(
    pra_B,
    select(
      pra_2023,
      all_of(shared_columns),
      relationship,
      main_occupation,
      availability,
      search_work,
      search_reason,
      search_steps,
      search_method,
      search_months,
      main_activity,
      main_prof_situation,
      main_sector,
      contract_type,
      hours
    )
  )

, tar_target(
    pra_A_csv,
    write.csv(pra_A,
              "pra_A.csv",
              row.names = FALSE)
  )

, tar_target(
    pra_B_csv,
    write.csv(pra_B,
              "pra_B.csv",
              row.names = FALSE)
  )

)

