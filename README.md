LiNNA
=====
**LiNNA (*Linear Neural Network Abstraction*)** is a Python library for abstracting feed-fordward
neural networks. The abstraction can then be used to speed up verification or
to gain insights about redundancies in the network.
The idea is to replace neurons by a linear combination of neurons
and thereby reduce the overall size of the network.

In addition, **LiNNA** implements the recently proposed *bisimulation* for neural
networks by Prabhakar (see [Bisimulations for Neural Network Reduction](https://link.springer.com/chapter/10.1007/978-3-030-94583-1_14))
and the previously proposed **DeepAbstract** (see Publications for details).

For more information, we refer to the [LiNNA website]().

Getting Started
---------------

You can install LiNNA by running `python setup.py install`. 
Please note that it will install Pytorch with CUDA as default. If you don't have a GPU, please run `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu` first.

If you want to use the L2-coefficient-finder, please also install Gurobi (e.g. `pip3 install gurobipy`) and make sure you have an active license ([Gurobi](https://www.gurobi.com/academia/academic-program-and-licenses/)).

Citing LiNNA
------------

Experimental Results
--------------------

Authors
-------
*LiNNA* is developed and maintained by [Calvin Chau](https://cxlvinchau.github.io/), [Jan Křetı́nský](https://www7.in.tum.de/~kretinsk/) and [Stefanie Mohr](https://www7.in.tum.de/~mohr/)
at the [Technical University of Munich](https://www.in.tum.de/en/in/cover-page/).

Publications
------------
**DeepAbstract: Neural Network Abstraction for Accelerating Verification**
*Pranav Ashok, Vahid Hashemi, Jan Křetínský and Stefanie Mohr*
([Paper](https://link.springer.com/chapter/10.1007/978-3-030-59152-6_5))
