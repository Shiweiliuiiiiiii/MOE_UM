# Upcycling-MoE


## Parallel Experts
Parallel Experts is a requirement for this codebase. To get it in your virtual env follow these steps:
1. Clone https://github.com/UMass-Foundation-Model/Mod-Squad.git
2. cd parallel_linear
3. pip install .
Note, you need timm==0.9.5 pytorch==1.13.1 for this to work

4. --num_round: control how many round will be run.
5. --moe_epoch control how many epochs will be runed for training moe.
6. --finetinue_epoch control how many epochs will be runed for running dense.
   
