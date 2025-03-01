# Modus operandi

Goal: The main.py must start a MNIST environment with 10 MNIST agents and train them

Current problems:
    - Can't properly divide between neural net and evolution algos
    - Can't define exactly what is the agent.py supposed to include
        - Probably all of the CONFIG
        - Setting up the MNIST and brain for MNIST
        - Functions to call from the Brain and Evolution
    - Where is the data going to be fetched from?
        - MNIST environment?

1. Evolution and Brain must be two classes.
    - Possibly base classes, or just abstract classes if each type of environment are not too different from each other
    - Then the config can be passed to them