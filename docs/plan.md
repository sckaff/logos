# Modus operandi

- Everything should be done through the environment. Meaning, agents must be controlled through the environment.
- Add different loss functions
- Time to implement evolution:
  - Able to read NN layout
  - Probabilities to change each thing in a neural network structure
    - Those being: TBD
- Create test cases
- Training is taking way too long

## Questions

- Is it possible to create a different neural network but still pass weights from similar networks?
- Is maintenance/stability emergent or should it be implemented?
- Sure, making the network bigger is always better, but what is the constraint/limitation for this?
  - Think of competition
- What separates system 1 and system 2 in humans?
  - System 1: What we do without learning
  - System 2: What we must learn
  - 'easy/hard tasks'? Moravec's paradox?
- Revisit Nash equilibrium? Achieve Open-ended evolution? Never-ending algorithm? Exploration vs exploitation?
  - "To find Nash equilibrium, have every player in a non-cooperative game reveal their strategies to one another. If no player changes their strategy after knowing all others' strategies, a Nash equilibrium exists.”
- Any benefit pairing it with Reinforcement Learning? Q*?
- Competitive Coevolution = Adversarial Training | Cooperative Coevolution = ?
- Is it possible to implement node creation and destruction on PyTorch?
  - New degree of freedom along with weight updates.
  - Solve for vanishing and exploding gradients naturally
  - May have to implement my own NN lib
- What would be akin to faith?
- How to train agents through observation? (senses)
- How to generate big bang sized complexity?
  - Must try algorithms. Particle generator c++
- To what extent must a neural net predict the future -> optimize? (Baby vs. adult)
- RL Agent needs periodic 'trials and tribulations.' What would the nature of these trials be? Creative? Cyclic? Monte-Carlo like?
  - It will be emergent through competition with other agents. Maybe not useful at the beginning, as the environment must be populated first, then competed against (as agents become increasingly different in neural gene).
