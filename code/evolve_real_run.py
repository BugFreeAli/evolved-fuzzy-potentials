"""
EF-PBRS Phase 4 -- the real evolutionary search.

"""

import json

import cma

from ef_pbrs_common import evaluate_theta_fitness, INITIAL_THETA, THETA_BOUNDS

ENV_ID = "PandaPush-v3"
TRAINING_BUDGET = 30_000
N_EVAL_EPISODES = 15
POPSIZE = 6
GENERATIONS = 10

es = cma.CMAEvolutionStrategy(
    INITIAL_THETA, 0.05,
    {"popsize": POPSIZE, "bounds": list(THETA_BOUNDS), "seed": 1}
)

generation_log = []
best_fitness_ever = float("-inf")
best_theta_ever = list(INITIAL_THETA)

for gen in range(GENERATIONS):
    print(f"\n=== Generation {gen}/{GENERATIONS - 1} ===")
    solutions = es.ask()
    fitnesses = []
    successes_list = []
    distances_list = []

    for i, theta in enumerate(solutions):
        seed = 5000 + gen * POPSIZE + i
        fitness, succ, dist = evaluate_theta_fitness(
            theta, ENV_ID, TRAINING_BUDGET, N_EVAL_EPISODES, seed
        )
        fitnesses.append(fitness)
        successes_list.append(succ)
        distances_list.append(dist)
        print(f"  candidate {i}: fitness={fitness:.3f}  (success={succ:.0%}, mean_final_x2={dist:.3f})")

        if fitness > best_fitness_ever:
            best_fitness_ever = fitness
            best_theta_ever = list(theta)
            print(f"    -> new best-ever fitness ({fitness:.3f})")

    es.tell(solutions, [-f for f in fitnesses])

    gen_record = {
        "generation": gen,
        "best_fitness": max(fitnesses),
        "mean_fitness": sum(fitnesses) / len(fitnesses),
        "best_success_rate": successes_list[fitnesses.index(max(fitnesses))],
        "best_theta_this_gen": list(solutions[fitnesses.index(max(fitnesses))]),
        "best_fitness_ever": best_fitness_ever,
        "best_theta_ever": best_theta_ever,
    }
    generation_log.append(gen_record)

    
    
    with open("evolution_log.json", "w") as f:
        json.dump(generation_log, f, indent=2)
    with open("best_theta.json", "w") as f:
        json.dump({"theta": best_theta_ever, "fitness": best_fitness_ever}, f, indent=2)

    print(f"Generation {gen} done: best={max(fitnesses):.3f}, mean={sum(fitnesses)/len(fitnesses):.3f}, "
          f"best-ever={best_fitness_ever:.3f}  (checkpoint saved)")

print("\n=== Evolution complete ===")
print(f"Best fitness found: {best_fitness_ever:.4f}")
print(f"Best theta saved to best_theta.json")
print(f"Full generation-by-generation log saved to evolution_log.json")
