# AI Racing Agent - Genetic Algorithm Training

![Demo Video](demo.gif)

A racing game environment where an AI agent learns to drive using a Genetic Algorithm (GA) to evolve neural networks. The agent learns to navigate various procedurally generated tracks, complete laps, and maximize rewards through checkpoint-based progression.

## Features

- **Genetic Algorithm Training**: Evolves neural networks using tournament selection, crossover, and mutation
- **Physics-Based Simulation**: Close to realistic car physics with friction, and speed-dependent steering
- **Visual Testing**: Pygame-based visualization for watching trained models race
- **Manual Control**: Test tracks manually with keyboard controls
- **Model Comparison**: Compare performance across different trained models

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
Racing/
├── env.py              # Racing environment (DriftingEnv)
├── train.py            # Genetic Algorithm training script
├── gen.py              # Track generation and selection tool
├── test.py             # Model testing and visualization
├── record_video.py     # Video recording script
├── demo.gif            # Demo video of trained model (animated GIF)
├── models/             # Saved trained models (timestamped directories)
└── tracks/             # Generated tracks and cache
    ├── track_cache.pkl # Cached track data
    └── track_*.png     # Track visualization images
```

## Quick Start

### 1. Generate Tracks

First, generate and select tracks for training (if don't want to use existing ones):

```bash
python gen.py
```

This will:
- Generate 20 procedurally generated tracks
- Open a GUI window showing all tracks
- Click tracks to toggle selection (green = selected)
- Click "Save Selected" to cache tracks for training/testing
- Tracks are saved to `tracks/track_cache.pkl`

### 2. Train a Model

Train an AI agent using the Genetic Algorithm:

```bash
python train.py
```

**Training Configuration** (edit in `train.py`):
- `POPULATION_SIZE = 100` - Number of neural networks per generation
- `GENERATIONS = 500` - Number of generations to evolve
- `TOURNAMENT_SIZE = 7` - Tournament selection size
- `ELITE_PERCENTAGE = 0.15` - Top 15% preserved unchanged
- `CROSSOVER_PROBABILITY = 1.0` - Probability of crossover vs cloning
- `MUTATION_RATE = 0.15` - Probability of mutating each weight
- `MUTATION_SCALE = 0.5` - Magnitude of mutations
- `MUTATION_PROBABILITY = 0.8` - Probability of applying mutation to child
- `EVALUATION_EPISODES = 20` - Episodes per fitness evaluation
- `NUM_WORKERS = 8` - Parallel evaluation workers

Training will:
- Create a timestamped directory in `models/` (e.g., `models/20251112_144741/`)
- Save the best model from each generation as `{generation}.pkl`
- Save the final best model as `final.pkl`
- Generate a training progress plot (`training_progress.png`)

### 3. Test Pre-trained Model

Test the latest trained model with visualization:

```bash
python test.py
```

Or test a specific model:

```bash
python test.py model models/20251112_144741/final.pkl
```

**Controls:**
- `ESC` - Exit
- `SPACE` - Reset to next track

The visualization shows:
- Track boundaries (gray lines)
- Centerline (dark gray)
- Checkpoints (green circles, yellow = current)
- Car (red rectangle with yellow direction indicator)
- HUD with speed, checkpoint progress, laps, and reward

### 4. Manual Control Testing

Test the environment manually with keyboard controls:

```bash
python test.py manual
```

**Controls:**
- `UP Arrow` - Accelerate
- `DOWN Arrow` - Brake
- `LEFT Arrow` - Steer left
- `RIGHT Arrow` - Steer right
- `SPACE` - Reset to next track
- `ESC` - Exit

### 5. Compare Models

Compare performance of all saved models:

```bash
python test.py compare
```

This evaluates each model on all cached tracks and reports average rewards.

## Scripts Overview

### `gen.py` - Track Generation

Generates procedurally created racing tracks with various procedurally generated styles using multiple generation strategies.

**Usage:**
```bash
python gen.py
```

**Output:**
- `tracks/track_cache.pkl` - Pickled track data
- `tracks/track_*.png` - Visualization images

### `train.py` - Training Script

Trains neural networks using a Genetic Algorithm.

**Neural Network Architecture:**
- Input: 12 features (lateral position, track edges at 4 distances, velocity, angular velocity, checkpoint progress)
- Hidden: 64 neurons with tanh activation
- Output: 2 actions (acceleration, steering)

**Genetic Algorithm:**
- Tournament selection
- Uniform crossover
- Gaussian mutation
- Elitism (top 15% preserved)

**Usage:**
```bash
python train.py
```

**Output:**
- `models/{timestamp}/` - Training run directory
  - `{generation}.pkl` - Best model per generation
  - `final.pkl` - Final best model
  - `training_progress.png` - Fitness over time plot

### `test.py` - Testing Script

Three testing modes:

#### 1. Test Trained Model (Default)
```bash
python test.py
# or
python test.py model [model_path]
```

Tests a trained model with pygame visualization. If no path is provided, uses the latest model from `models/`.

#### 2. Manual Control
```bash
python test.py manual
```

Allows manual keyboard control for testing tracks.

#### 3. Compare Models
```bash
python test.py compare
```

Evaluates all saved models and reports average performance.

### `env.py` - Racing Environment

The `DriftingEnv` class implements the racing environment:

**Observation Space (12 features):**
1. Lateral position on track (-1 to 1)
2-9. Distances to track edges at 4 lookahead distances (20, 40, 60, 80 units)
10. Normalized velocity
11. Normalized angular velocity
12. Forward progress to next checkpoint

**Action Space (2 continuous):**
- `[acceleration, steering]`
- Acceleration: -1 (brake) to 1 (accelerate)
- Steering: -1 (left) to 1 (right)

**Reward System:**
- +1 for each checkpoint passed
- +40 for completing a lap
- Episode terminates if car goes off-track

**Physics:**
- Friction: 0.98 (velocity decay)
- Acceleration: 0.3
- Turn speed: 0.08 (scaled by velocity)
- Max velocity: 15 units/step
- Speed-dependent steering effectiveness

## Testing Pre-trained Models

The repository includes pre-trained models in `models/20251112_144741/`. To test them:

### Test Latest Model
```bash
python test.py
```

### Test Specific Generation
```bash
python test.py model models/20251112_144741/50.pkl
```

### Test Final Model
```bash
python test.py model models/20251112_144741/final.pkl
```

### Compare All Models
```bash
python test.py compare
```

## Configuration

### Training Parameters (`train.py`)

```python
POPULATION_SIZE = 100        # Population size
GENERATIONS = 500           # Training generations
TOURNAMENT_SIZE = 7         # Tournament selection size
ELITE_PERCENTAGE = 0.15     # Elite preservation
CROSSOVER_PROBABILITY = 1.0 # Probability of crossover vs cloning
MUTATION_RATE = 0.15        # Probability of mutating each weight
MUTATION_SCALE = 0.5        # Mutation magnitude
MUTATION_PROBABILITY = 0.8  # Probability of applying mutation to child
EVALUATION_EPISODES = 20    # Episodes per evaluation
NUM_WORKERS = 8             # Parallel workers
```

### Network Architecture (`train.py`)

```python
INPUT_SIZE = 12             # Observation features
HIDDEN_SIZE = 64           # Hidden layer neurons
OUTPUT_SIZE = 2            # Actions (accel, steer)
```

### Environment Parameters (`env.py`)

```python
max_velocity = 15           # Maximum speed
max_angular_velocity = 5   # Maximum turn rate
friction = 0.98             # Velocity decay
acceleration = 0.3         # Acceleration power
turn_speed = 0.08          # Base steering sensitivity
max_steps = 400            # Max steps per episode
```

### Track Generation (`gen.py`)

```python
NUM_TRACKS = 20            # Tracks to generate
POINTS_PER_TRACK = 800     # Centerline points
BASE_SCALE = 400           # Track size scale
TRACK_WIDTH = 40           # Track width
```

## Tips

1. **Track Selection**: Use `gen.py` to generate diverse tracks. More variety improves generalization.

2. **Training Time**: Training 500 generations with 100 population takes a couple of hours. Reduce `GENERATIONS` or `POPULATION_SIZE` for faster experiments.

3. **Model Selection**: Check `training_progress.png` to see if fitness is still improving. You may want to use a model from a later generation rather than `final.pkl`.

4. **Performance**: Adjust `NUM_WORKERS` in `train.py` to match your CPU cores for optimal parallel evaluation.

5. **Visualization**: The pygame window runs at 60 FPS. Close it with ESC or close the window.

## License

This project is open source and available for educational and research purposes.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional track generation algorithms
- Alternative neural network architectures
- Different evolutionary strategies
- Reward function improvements
- Performance optimizations

