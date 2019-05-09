This code accompanies the paper "Stein Point Markov Chain Monte Carlo".
It was written in MATLAB R2017a and also tested in MATLAB R2016a.

There are two directories:

"lib" is a library that contains basic functions associated with SP-MCMC
      and the various other algorithms being implemented.

"ppr" are scripts that specifically reproduce the experiments performed
      in both the main text and the supplement of the paper.

To reproduce an experiment, navigate to "src/ppr" and run the relevant script
e.g. "cmp_igarch.m" will run the IGARCH experiment from the main text and
assess performance using KSD and energy distance.
