#!/bin/bash
source activate paper2022
models=("/mlodata1/hokarami/fari/bread/best_models/f10_f10/2023-10-29 20:35:20" "/mlodata1/hokarami/fari/bread/best_models/f10_f10/2023-10-29 22:09:08")
algos=("hungarian" "percentage_hungarian")

# Define the config string
config="config_test_all_f10_f10"

# Loop through each model and algorithm combination and run the command
for model in "${models[@]}"
do
    for algo in "${algos[@]}"
    do
        python /mlodata1/hokarami/fari/bread/notebooks/testing_nicoles/test_p.py --config "$config" --algo "$algo" --model "$model"
    done
done
