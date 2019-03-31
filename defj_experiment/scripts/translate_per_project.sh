#!/usr/bin/env bash
for project in Chart Time Math Lang Closure Mockito;do
    echo $project
    python defj_experiment/full_translate_individual_project.py BR_ALL 20 200 $project
done
