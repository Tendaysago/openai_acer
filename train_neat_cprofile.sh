#!/bin/bash

PYTHONPATH=. python -m cProfile -o rogueprof.prof -s cumtime baselines/acer/run_rogue_Neat.py -f cfg_rogue_Neat_default.cfg --config config-neat-simple $*
