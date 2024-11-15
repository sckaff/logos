# Evolutionary Computation

## Possible parameters
- Population
- Mutation Rate
- Crossover Rate
- Selection Mechanism
- Fitness Function
- Speciation
- Genetic Encoding
- Learning Rate
	-  Hybrid method (gradient)
- Noise (stochastic handling)

### OBS
- Show initial state/best model
- Figure out good encoding mech
	- For initial state?
	- Including weights file with it

## Notes
- Swarm intelligence? Agents each w/ differennt purposes
- Think of debbuging whenn creating vizualization
- Fully-connected network foor each input
	- "Areas of the brain"
	- Better than auxiliary networks?
- How to allow "junk DNA"? (Instead of allowing errors)
- Create more neurons in saturated areas
- Reproduce b/w species?
- Softmax is always used
- Ability too create new input?
	- Like creating new organns
	- Ideally we want to evolve towards that (new input)
- Encoder = Genotype | Decoder = Phenotype
- String-matching genes is also more efficient
- Weight-mutation? Based on gaussian if so
- When to add a  new layer?
	- If adding new noe, fully connected
	- When to change structure and when to change weights
- "Cap value" to avoid overfitting (?)

### Choices
- Do ont use libs
- Standard weight init?
- No biases?

## Takeaways
- Best ways to combine EC + RL

1. Grow the nodes andn layers much faster than NEAT
2. Do NOT pre-assign weights
	- Reason to do this:
		- We must combat backprop by having multiple points of entry
	- This is a therapy:
		- Not assigning weights may create "stable" arc's 
3. EC will happen less often, and the main learning will happenn through RL (efficiency)
4. Must work on different fields
	- More landong
	- Knapsack
	- Snake
	- MNIST (vision)
