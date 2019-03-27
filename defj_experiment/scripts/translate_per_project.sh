#!/usr/bin/env bash
for project in Char Time Math Lang Closure;do
    python defj_experiment/full_translate_individual_project.py BR_ALL 20 200 $project
done