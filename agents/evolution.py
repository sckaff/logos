# Evolution loop
population = [NeuralNetwork() for _ in range(population_size)]

for gen in range(generations):
    # Evaluate fitness
    fitness_scores = [evaluate(nn, X_train, y_train) for nn in population]
    
    # Select the top-k networks
    sorted_indices = np.argsort(fitness_scores)[::-1]  # Sort by highest accuracy
    best_networks = [population[i] for i in sorted_indices[:top_k]]
    
    print(f"Generation {gen+1}, Best Accuracy: {fitness_scores[sorted_indices[0]]:.4f}")

    # Create new population: keep top-k and mutate
    new_population = best_networks[:]
    while len(new_population) < population_size:
        parent = np.random.choice(best_networks)  # Pick a random elite
        child = mutate(parent, mutation_rate)
        new_population.append(child)

    population = new_population  # Replace old population

# Final evaluation on test set
best_nn = best_networks[0]
test_accuracy = evaluate(best_nn, X_test, y_test)
print(f"Final Test Accuracy: {test_accuracy:.4f}")