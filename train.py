import os
import pickle
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid blocking
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from env import DriftingEnv

TRACKS_FILE = "tracks/track_cache.pkl"

def load_cached_tracks():
    """Load pre-generated tracks"""
    with open(TRACKS_FILE, 'rb') as f:
        tracks = pickle.load(f)
    return tracks

MODEL_DIR = "models"
POPULATION_SIZE = 100
TOURNAMENT_SIZE = 7
ELITE_PERCENTAGE = 0.15
CROSSOVER_PROBABILITY = 1.0
MUTATION_RATE = 0.15
MUTATION_SCALE = 0.5
MUTATION_PROBABILITY = 0.8
NUM_WORKERS = 8  # Number of parallel workers for M1 Pro
GENERATIONS = 500
EVALUATION_EPISODES = 20

INPUT_SIZE = 12
HIDDEN_SIZE = 64
OUTPUT_SIZE = 2

# ============================================================================
# NEURAL NETWORK
# ============================================================================

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights randomly
        self.w1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.random.randn(hidden_size) * 0.5
        self.w2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.random.randn(output_size) * 0.5
    
    def forward(self, x):
        """Forward pass through network"""
        h = np.tanh(np.dot(x, self.w1) + self.b1)
        output = np.tanh(np.dot(h, self.w2) + self.b2)
        return output
    
    def get_weights(self):
        """Get all weights as a flat array for GA operations"""
        return np.concatenate([
            self.w1.flatten(),
            self.b1.flatten(),
            self.w2.flatten(),
            self.b2.flatten()
        ])
    
    def set_weights(self, weights):
        """Set weights from a flat array"""
        idx = 0
        
        w1_size = self.input_size * self.hidden_size
        self.w1 = weights[idx:idx+w1_size].reshape(self.input_size, self.hidden_size)
        idx += w1_size
        
        self.b1 = weights[idx:idx+self.hidden_size]
        idx += self.hidden_size
        
        w2_size = self.hidden_size * self.output_size
        self.w2 = weights[idx:idx+w2_size].reshape(self.hidden_size, self.output_size)
        idx += w2_size
        
        self.b2 = weights[idx:idx+self.output_size]

def evaluate_network_parallel(args):
    """Global function for parallel evaluation - needed for pickling"""
    network, tracks = args
    total_fitness = 0
    
    # Limit episodes to number of available tracks, use each track at most once
    num_episodes = min(EVALUATION_EPISODES, len(tracks))

    for episode in range(num_episodes):
        # Use each track once, in order
        track_idx = episode
        env = DriftingEnv(render_mode=None, track_data=tracks[track_idx])
        obs = env.reset()

        total_reward = 0
        done = False

        while not done:
            action = network.forward(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        fitness = total_reward
        total_fitness += fitness

    return total_fitness / num_episodes

# ============================================================================
# GENETIC ALGORITHM
# ============================================================================

class GeneticAlgorithm:
    def __init__(self):
        self.population_size = POPULATION_SIZE
        os.makedirs(MODEL_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(MODEL_DIR, timestamp)
        suffix = 1
        while os.path.exists(run_dir):
            run_dir = os.path.join(MODEL_DIR, f"{timestamp}_{suffix:02d}")
            suffix += 1
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        
        # Initialize population
        self.population = [
            NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
            for _ in range(self.population_size)
        ]
        
        self.fitness_scores = np.zeros(self.population_size)
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def evaluate_individual(self, network, tracks, render=False, num_episodes=EVALUATION_EPISODES):
        """Evaluate fitness of a single neural network across multiple episodes"""
        total_fitness = 0
        
        # Limit episodes to number of available tracks, use each track at most once
        actual_episodes = min(num_episodes, len(tracks))
        
        for episode in range(actual_episodes):
            # Use each track once, in order
            track_idx = episode
            env = DriftingEnv(render_mode="human" if render else None, track_data=tracks[track_idx])
            obs = env.reset()
            
            total_reward = 0
            done = False
            
            while not done:
                action = network.forward(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                
                if render:
                    env.render()
                    # env.render_visual()  # Uncomment to see visual debugging
                
                done = terminated or truncated
            
            # Fitness = reward + bonuses for progress
            fitness = total_reward
            total_fitness += fitness
        
        # Return average fitness across all episodes
        return total_fitness / actual_episodes
    
    def evaluate_population(self, tracks):
        """Evaluate fitness of entire population in parallel"""
        with Pool(processes=NUM_WORKERS) as pool:
            args = [(network, tracks) for network in self.population]
            self.fitness_scores = np.array(pool.map(evaluate_network_parallel, args))
    
    def selection_tournament(self):
        """Tournament selection - pick best from random group"""
        # Randomly select individuals for tournament
        indices = np.random.choice(self.population_size, TOURNAMENT_SIZE, replace=False)
        
        # Return the best one from tournament
        winner_idx = indices[np.argmax(self.fitness_scores[indices])]
        return self.population[winner_idx]
    
    def crossover(self, parent1, parent2):
        """Uniform crossover - randomly mix parent genes"""
        child = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        
        weights1 = parent1.get_weights()
        weights2 = parent2.get_weights()
        
        # For each weight, randomly choose from parent1 or parent2
        mask = np.random.rand(len(weights1)) < 0.5
        child_weights = np.where(mask, weights1, weights2)
        
        child.set_weights(child_weights)
        return child
    
    def mutate(self, network):
        """Add random noise to network weights"""
        weights = network.get_weights()
        
        # For each weight, randomly decide if it mutates
        mutation_mask = np.random.rand(len(weights)) < MUTATION_RATE
        
        # Add Gaussian noise to selected weights
        mutations = np.random.randn(len(weights)) * MUTATION_SCALE
        weights += mutation_mask * mutations
        
        network.set_weights(weights)
    
    def evolve(self):
        """Create next generation through selection, crossover, and mutation"""
        # Sort population by fitness (best first)
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        
        # ELITISM: Keep top individuals unchanged
        elite_size = max(1, int(self.population_size * ELITE_PERCENTAGE))
        new_population = [self.population[i] for i in sorted_indices[:elite_size]]
        
        # Fill rest of population with offspring
        while len(new_population) < self.population_size:
            # SELECTION: Pick two parents
            parent1 = self.selection_tournament()
            parent2 = self.selection_tournament()
            
            # CROSSOVER: Combine parents to create child
            if np.random.rand() < CROSSOVER_PROBABILITY:
                child = self.crossover(parent1, parent2)
            else:
                # Clone one parent
                child = self.crossover(parent1, parent1)
            
            # MUTATION: Randomly modify child
            if np.random.rand() < MUTATION_PROBABILITY:
                self.mutate(child)
            
            new_population.append(child)
        
        self.population = new_population
    
    def train(self, tracks):
        """Main training loop"""
        print(f"\n{'='*70}")
        print(f"GENETIC ALGORITHM TRAINING")
        print(f"{'='*70}")
        print(f"Population size: {POPULATION_SIZE}")
        print(f"Generations: {GENERATIONS}")
        print(f"Elite percentage: {ELITE_PERCENTAGE}")
        print(f"Tournament size: {TOURNAMENT_SIZE}")
        print(f"Mutation rate: {MUTATION_RATE}")
        print(f"Mutation scale: {MUTATION_SCALE}")
        print(f"Mutation probability: {MUTATION_PROBABILITY}")
        print(f"{'='*70}\n")
        
        for gen in range(GENERATIONS):
            # Evaluate all individuals
            self.evaluate_population(tracks)
            
            # Track statistics
            best_fitness = np.max(self.fitness_scores)
            avg_fitness = np.mean(self.fitness_scores)
            
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            
            print(f"Generation {gen+1}/{GENERATIONS}: Best = {best_fitness:.2f}, Average = {avg_fitness:.2f}")
            
            # Save best model this generation
            best_idx = np.argmax(self.fitness_scores)
            gen_model_path = os.path.join(self.run_dir, f'{gen+1}.pkl')
            with open(gen_model_path, 'wb') as f:
                pickle.dump(self.population[best_idx], f)
            
            # Evolve to next generation (skip on last generation)
            if gen < GENERATIONS - 1:
                self.evolve()
        
        # Save final plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(self.best_fitness_history) + 1), self.best_fitness_history, label='Best Fitness', linewidth=2)
        ax.plot(range(1, len(self.avg_fitness_history) + 1), self.avg_fitness_history, label='Average Fitness', linewidth=2, alpha=0.7)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.run_dir, 'training_progress.png'))
        plt.close(fig)
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"{'='*70}")
        
        # Return best individual
        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx]

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Load cached tracks once
    tracks = load_cached_tracks()
    print(f"Loaded {len(tracks)} cached tracks for training")
    
    # Create and train GA
    ga = GeneticAlgorithm()
    best_network = ga.train(tracks)
    
    # Save final best model
    final_model_path = os.path.join(ga.run_dir, 'final.pkl')
    with open(final_model_path, 'wb') as f:
        pickle.dump(best_network, f)
    
    print("\nTesting best model on random tracks...")
    final_fitness = ga.evaluate_individual(best_network, tracks, render=True, num_episodes=EVALUATION_EPISODES)
    print(f"Final best fitness (avg over {EVALUATION_EPISODES} tracks): {final_fitness:.2f}")
    
    # Print improvement over training
    print(f"\nImprovement: {ga.best_fitness_history[0]:.2f} → {ga.best_fitness_history[-1]:.2f}")
    print(f"Total gain: {ga.best_fitness_history[-1] - ga.best_fitness_history[0]:.2f}")