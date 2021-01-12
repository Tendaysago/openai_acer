#!/bin/bash

PYTHONPATH=. python baselines/acer/run_rogue_Neat.py -f cfg_rogue_Neat_default.cfg --config config-neat-simple $*
