# Evolutionary Computation

## Possible parameters

### EC

- Population
- Mutation Rate
- Crossover Rate
- Selection Mechanism
- Fitness Function
- Speciation
- Genetic Encoding
- Learning Rate
  - Hybrid method (gradient)
- Noise (stochastic handling)

### NEURAL

- Increase/decrease neurons in a layer
- Increase/decrease layer

## Notes

- Swarm intelligence? Agents each w/ different purposes
- Think of debbuging when creating vizualization
- Fully-connected network foor each input
  - "Areas of the brain"
  - Better than auxiliary networks?
- How to allow "junk DNA"? (Instead of allowing errors)
- Create more neurons in saturated areas
- Reproduce b/w species?
- Softmax is always used
- Ability to create new input?
  - Like creating new organs
  - Ideally we want to evolve towards that (new input)
- Encoder = Genotype | Decoder = Phenotype
- String-matching genes is also more efficient
- Weight-mutation? Based on gaussian if so
- When to add a  new layer?
  - If adding new noe, fully connected
  - When to change structure and when to change weights
- "Cap value" to avoid overfitting (?)

## Takeaways

- Best ways to combine EC + RL

1. Grow the nodes and layers much faster than NEAT  
2. Do NOT pre-assign weights
   - Reason to do this:  
     - We must combat backprop by having multiple points of entry  
   - This is a therapy:  
     - Not assigning weights may create "stable" arcs  
3. EC will happen less often, and the main learning will happen through RL (efficiency)
4. Must work on different fields
   - More landong
   - Knapsack
   - Snake
   - MNIST (vision)
