# Documentation

Short answer: Incomplete

plan.md: The way I want to build this, scattered notes I will arrange
resources.md: Some resources I come back to when I want to iterate through some ideas

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

## QA

1. TWO APPROACHES: From environment simulations, or pre-built genetics. How to merge both?
  
    Both.

2. Agent needs periodic 'trials and tribulations.' What would the nature of these trials be? Creative? Cyclic? Monte-Carlo like (RL trials)?
  
    It will be emergent through competition with other agents. Maybe not useful at the beginning, as the environment must be populated first, then competed against (as agents become increasingly different in neural gene).
