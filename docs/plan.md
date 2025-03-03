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

## Questions

- Are activation functions the only thing that gives us non-linearities in a standard FFNN?
- Is maintenance/stability emergent or should it be implemented?
- Sure, making the network bigger is always better, but what is the constraint/limitation for this?
  - Think of competition
- What is a base/primordial model?
- What separates system 1 and system 2 in humans?
  - System 1: What we do without learning
  - System 2: What we must learn
  - 'easy/hard tasks'? Moravec's paradox?
- Revisit Nash equilibrium? Achieve Open-ended evolution? Never-ending algorithm? Exploration vs exploitation?
  - "To find Nash equilibrium, have every player in a non-cooperative game reveal their strategies to one another. If no player changes their strategy after knowing all others' strategies, a Nash equilibrium exists.”
- Any benefit pairing it with Reinforcement Learning? Q*?
- Competitive Coevolution = Adversarial Training | Cooperative Coevolution = ?
- Node creation and destruction?
  - New degree of freedom along with weight updates.
  - Solve for vanishing and exploding gradients naturally
- What would be akin to faith?
- How to train agents through observation? (senses)
- How to generate big bang sized complexity?
  - Must try algorithms. Particle generator c++
- Self-replication-only at first?
  - Are chromosomes an emergent feature for adaptation?
- To what extent must a neural net predict the future -> optimize? (Baby vs. adult)
- RL Agent needs periodic 'trials and tribulations.' What would the nature of these trials be? Creative? Cyclic? Monte-Carlo like?
  - It will be emergent through competition with other agents. Maybe not useful at the beginning, as the environment must be populated first, then competed against (as agents become increasingly different in neural gene).
