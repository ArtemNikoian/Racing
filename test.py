import os
import sys
import numpy as np
import pickle
from env import DriftingEnv
from train import NeuralNetwork as _NeuralNetwork

sys.modules[__name__].NeuralNetwork = _NeuralNetwork

TRACKS_FILE = "tracks/track_cache.pkl"

def load_cached_tracks():
    """Load pre-generated tracks"""
    with open(TRACKS_FILE, 'rb') as f:
        return pickle.load(f)

MODEL_DIR = "models"

def _list_run_dirs():
    if not os.path.isdir(MODEL_DIR):
        return []
    run_dirs = [
        os.path.join(MODEL_DIR, name)
        for name in os.listdir(MODEL_DIR)
        if os.path.isdir(os.path.join(MODEL_DIR, name))
    ]
    return sorted(run_dirs)


def _latest_run_dir():
    run_dirs = _list_run_dirs()
    return run_dirs[-1] if run_dirs else None


def _parse_generation(path):
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(name)
    except ValueError:
        return float("inf")


def _latest_model_in_dir(directory):
    if not os.path.isdir(directory):
        return None
    candidates = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.endswith(".pkl")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (_parse_generation(p), os.path.basename(p)))
    return candidates[-1]


def _resolve_model_path(model_path):
    if model_path is None:
        latest_run = _latest_run_dir()
        if latest_run is None:
            raise FileNotFoundError("No saved model runs found in models directory.")
        resolved = os.path.join(latest_run, "final.pkl")
        if not os.path.exists(resolved):
            resolved = _latest_model_in_dir(latest_run)
        if resolved is None:
            raise FileNotFoundError(f"No model files found in latest run directory: {latest_run}")
        return resolved

    resolved = model_path if os.path.isabs(model_path) else os.path.join(MODEL_DIR, model_path)
    if os.path.isdir(resolved):
        candidate = os.path.join(resolved, "final.pkl")
        if os.path.exists(candidate):
            resolved = candidate
        else:
            latest_in_dir = _latest_model_in_dir(resolved)
            if latest_in_dir is None:
                raise FileNotFoundError(f"No model files found in provided directory: {resolved}")
            resolved = latest_in_dir
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Model file does not exist: {resolved}")
    return resolved


def test_trained_model(model_path=None):
    """Test a trained model with pygame visualization"""
    import pygame

    # Load model
    resolved_path = _resolve_model_path(model_path)

    with open(resolved_path, 'rb') as f:
        network = pickle.load(f)

    print("Testing trained model with visualization...")
    print("Press ESC to stop, SPACE to reset\n")

    # Load cached tracks
    tracks = load_cached_tracks()
    print(f"Loaded {len(tracks)} cached tracks for testing")

    # Initialize pygame
    pygame.init()
    width, height = 800, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Drifting Game - AI Agent")
    clock = pygame.time.Clock()

    # Start with first track
    track_idx = 0
    env = DriftingEnv(render_mode="human", track_data=tracks[track_idx])

    # Camera offset to center view
    camera_x, camera_y = width // 2, height // 2

    running = True
    obs = env.reset()
    total_reward = 0

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_SPACE:
                        # Cycle to next track
                        track_idx = (track_idx + 1) % len(tracks)
                        env = DriftingEnv(render_mode="human", track_data=tracks[track_idx])
                        obs = env.reset()
                        total_reward = 0

            # Get action from network
            action = network.forward(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # Reset if done
            if terminated or truncated:
                print(f"Episode finished! Reward: {total_reward:.2f}, "
                      f"Checkpoints: {env.checkpoint_index}, Laps: {env.laps_completed}")
                # Cycle to next track
                track_idx = (track_idx + 1) % len(tracks)
                env = DriftingEnv(render_mode="human", track_data=tracks[track_idx])
                obs = env.reset()
                total_reward = 0

            # Draw
            screen.fill((50, 50, 50))

            # Draw track
            # Get bounds for scaling
            all_x = [p[0] for p in env.centerline_points]
            all_y = [p[1] for p in env.centerline_points]
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)

            span_x = max_x - min_x
            span_y = max_y - min_y
            max_span = max(span_x, span_y, 1.0)
            # Maintain consistent track width, scale everything proportionally if needed
            reference_track_width_pixels = 40  # Desired visual width in pixels
            scale_width = reference_track_width_pixels / env.track_width
            scale_fit = 0.7 * min(width, height) / max_span
            # Use scale_width for consistent track width, but scale down proportionally if needed
            scale = min(scale_width, scale_fit)  # Maintains relative scale between track and car
            center_x = (max_x + min_x) / 2
            center_y = (max_y + min_y) / 2
            points = np.array(env.centerline_points)
            num_points = len(points)

            # Draw track boundaries - matching actual collision detection

            for boundary in env.track_boundaries:

                p1, p2 = boundary['p1'], boundary['p2']

                perp = boundary['perp'] * env.track_width

                

                # Inner boundary

                inner_p1 = (int(camera_x + (p1[0] - perp[0] - center_x) * scale),

                           int(camera_y + (p1[1] - perp[1] - center_y) * scale))

                inner_p2 = (int(camera_x + (p2[0] - perp[0] - center_x) * scale),

                           int(camera_y + (p2[1] - perp[1] - center_y) * scale))

                

                # Outer boundary

                outer_p1 = (int(camera_x + (p1[0] + perp[0] - center_x) * scale),

                           int(camera_y + (p1[1] + perp[1] - center_y) * scale))

                outer_p2 = (int(camera_x + (p2[0] + perp[0] - center_x) * scale),

                           int(camera_y + (p2[1] + perp[1] - center_y) * scale))

                

                pygame.draw.line(screen, (150, 150, 150), inner_p1, inner_p2, 2)

                pygame.draw.line(screen, (150, 150, 150), outer_p1, outer_p2, 2)


            # Draw centerline
            centerline_screen = []
            for p in env.centerline_points:
                screen_x = int(camera_x + (p[0] - center_x) * scale)
                screen_y = int(camera_y + (p[1] - center_y) * scale)
                centerline_screen.append((screen_x, screen_y))

            if len(centerline_screen) > 2:
                pygame.draw.lines(screen, (100, 100, 100), True, centerline_screen, 1)

            # Draw checkpoints - scale checkpoint size proportionally
            checkpoint_scale = scale / scale_width if scale_width > 0 else 1.0
            checkpoint_radius = max(2, int(6 * checkpoint_scale))
            for i, segment in enumerate(env.track_segments):
                x = int(camera_x + (segment['x'] - center_x) * scale)
                y = int(camera_y + (segment['y'] - center_y) * scale)

                color = (255, 255, 0) if i == env.checkpoint_index else (100, 200, 100)
                pygame.draw.circle(screen, color, (x, y), checkpoint_radius)

                if i % 5 == 0:  # Only show every 5th checkpoint number
                    font = pygame.font.Font(None, 18)
                    text = font.render(str(i), True, (200, 200, 200))
                    screen.blit(text, (x + 10, y - 10))

            # Draw car
            car_screen_x = int(camera_x + (env.car_x - center_x) * scale)
            car_screen_y = int(camera_y + (env.car_y - center_y) * scale)

            # Car body - scale car proportionally with track
            car_length_base = 15
            car_width_base = 8
            # Scale car by the same factor as track to maintain relative scale
            car_scale = scale / scale_width if scale_width > 0 else 1.0
            car_length = car_length_base * car_scale
            car_width = car_width_base * car_scale

            # Calculate car corners
            cos_a = np.cos(env.car_angle)
            sin_a = np.sin(env.car_angle)

            front_x = car_screen_x + car_length * cos_a
            front_y = car_screen_y + car_length * sin_a
            back_x = car_screen_x - car_length * cos_a
            back_y = car_screen_y - car_length * sin_a

            left_x = -car_width * sin_a
            left_y = car_width * cos_a
            right_x = car_width * sin_a
            right_y = -car_width * cos_a

            points = [
                (front_x + right_x, front_y + right_y),
                (front_x + left_x, front_y + left_y),
                (back_x + left_x, back_y + left_y),
                (back_x + right_x, back_y + right_y)
            ]

            pygame.draw.polygon(screen, (255, 0, 0), points)

            # Direction indicator - scale line width proportionally
            line_width = max(1, int(3 * car_scale))
            pygame.draw.line(screen, (255, 255, 0),
                           (car_screen_x, car_screen_y),
                           (front_x, front_y), line_width)

            # Draw HUD
            font = pygame.font.Font(None, 30)
            hud_texts = [
                f"Speed: {env.car_velocity:.1f}",
                f"Checkpoint: {env.checkpoint_index}/{len(env.track_segments)}",
                f"Laps: {env.laps_completed}",
                f"Reward: {total_reward:.1f}",
                f"Steps: {env.steps}"
            ]

            for i, text in enumerate(hud_texts):
                surface = font.render(text, True, (255, 255, 255))
                screen.blit(surface, (10, 10 + i * 30))

            pygame.display.flip()
            clock.tick(60)

    except KeyboardInterrupt:
        print("\nTesting stopped by user")
    finally:
        pygame.quit()

def test_manual_control():
    """Test environment with manual keyboard control"""
    import pygame

    print("Testing environment manually with keyboard...")
    print("Arrow keys: UP=accelerate, DOWN=brake, LEFT/RIGHT=steer")
    print("Press ESC to quit, SPACE to reset\n")

    # Load cached tracks
    tracks = load_cached_tracks()
    print(f"Loaded {len(tracks)} cached tracks for testing")

    pygame.init()
    width, height = 800, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Drifting Game - Manual Control")
    clock = pygame.time.Clock()

    # Start with first track
    track_idx = 0
    env = DriftingEnv(render_mode="human", track_data=tracks[track_idx])
    obs = env.reset()

    camera_x, camera_y = width // 2, height // 2
    running = True
    total_reward = 0

    while running:
        action = [0.0, 0.0]  # [acceleration, steering]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    # Cycle to next track
                    track_idx = (track_idx + 1) % len(tracks)
                    env = DriftingEnv(render_mode="human", track_data=tracks[track_idx])
                    obs = env.reset()
                    total_reward = 0

        # Get keyboard input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1.0
        if keys[pygame.K_DOWN]:
            action[0] = -0.5
        if keys[pygame.K_LEFT]:
            action[1] = -1.0
        if keys[pygame.K_RIGHT]:
            action[1] = 1.0

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode ended! Reward: {total_reward:.2f}")
            # Cycle to next track
            track_idx = (track_idx + 1) % len(tracks)
            env = DriftingEnv(render_mode="human", track_data=tracks[track_idx])
            obs = env.reset()
            total_reward = 0

        # Draw
        screen.fill((50, 50, 50))

        # Draw track
        # Get bounds for scaling
        all_x = [p[0] for p in env.centerline_points]
        all_y = [p[1] for p in env.centerline_points]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        span_x = max_x - min_x
        span_y = max_y - min_y
        max_span = max(span_x, span_y, 1.0)
        # Maintain consistent track width, scale everything proportionally if needed
        reference_track_width_pixels = 40  # Desired visual width in pixels
        scale_width = reference_track_width_pixels / env.track_width
        scale_fit = 0.7 * min(width, height) / max_span
        # Use scale_width for consistent track width, but scale down proportionally if needed
        scale = min(scale_width, scale_fit)  # Maintains relative scale between track and car
        center_x = (max_x + min_x) / 2
        center_y = (max_y + min_y) / 2
        
        # Draw track boundaries - matching actual collision detection

        for boundary in env.track_boundaries:

            p1, p2 = boundary['p1'], boundary['p2']

            perp = boundary['perp'] * env.track_width

            

            # Inner boundary

            inner_p1 = (int(camera_x + (p1[0] - perp[0] - center_x) * scale),

                       int(camera_y + (p1[1] - perp[1] - center_y) * scale))

            inner_p2 = (int(camera_x + (p2[0] - perp[0] - center_x) * scale),

                       int(camera_y + (p2[1] - perp[1] - center_y) * scale))

            

            # Outer boundary

            outer_p1 = (int(camera_x + (p1[0] + perp[0] - center_x) * scale),

                       int(camera_y + (p1[1] + perp[1] - center_y) * scale))

            outer_p2 = (int(camera_x + (p2[0] + perp[0] - center_x) * scale),

                       int(camera_y + (p2[1] + perp[1] - center_y) * scale))

            

            pygame.draw.line(screen, (150, 150, 150), inner_p1, inner_p2, 2)

            pygame.draw.line(screen, (150, 150, 150), outer_p1, outer_p2, 2)

        # Draw centerline
        centerline_screen = []
        for p in env.centerline_points:
            screen_x = int(camera_x + (p[0] - center_x) * scale)
            screen_y = int(camera_y + (p[1] - center_y) * scale)
            centerline_screen.append((screen_x, screen_y))

        if len(centerline_screen) > 2:
            pygame.draw.lines(screen, (100, 100, 100), True, centerline_screen, 1)

        # Scale checkpoint size proportionally
        checkpoint_scale = scale / scale_width if scale_width > 0 else 1.0
        checkpoint_radius = max(2, int(6 * checkpoint_scale))
        for i, segment in enumerate(env.track_segments):
            x = int(camera_x + (segment['x'] - center_x) * scale)
            y = int(camera_y + (segment['y'] - center_y) * scale)

            color = (255, 255, 0) if i == env.checkpoint_index else (100, 200, 100)
            pygame.draw.circle(screen, color, (x, y), checkpoint_radius)

            if i % 5 == 0:
                font = pygame.font.Font(None, 18)
                text = font.render(str(i), True, (200, 200, 200))
                screen.blit(text, (x + 10, y - 10))

        car_screen_x = int(camera_x + (env.car_x - center_x) * scale)
        car_screen_y = int(camera_y + (env.car_y - center_y) * scale)

        # Car body - scale car proportionally with track
        car_length_base = 15
        car_width_base = 8
        # Scale car by the same factor as track to maintain relative scale
        car_scale = scale / scale_width if scale_width > 0 else 1.0
        car_length = car_length_base * car_scale
        car_width = car_width_base * car_scale
        cos_a = np.cos(env.car_angle)
        sin_a = np.sin(env.car_angle)

        front_x = car_screen_x + car_length * cos_a
        front_y = car_screen_y + car_length * sin_a
        back_x = car_screen_x - car_length * cos_a
        back_y = car_screen_y - car_length * sin_a

        left_x = -car_width * sin_a
        left_y = car_width * cos_a
        right_x = car_width * sin_a
        right_y = -car_width * cos_a

        points = [
            (front_x + right_x, front_y + right_y),
            (front_x + left_x, front_y + left_y),
            (back_x + left_x, back_y + left_y),
            (back_x + right_x, back_y + right_y)
        ]

        pygame.draw.polygon(screen, (255, 0, 0), points)
        # Direction indicator - scale line width proportionally
        line_width = max(1, int(3 * car_scale))
        pygame.draw.line(screen, (255, 255, 0),
                       (car_screen_x, car_screen_y),
                       (front_x, front_y), line_width)

        font = pygame.font.Font(None, 30)
        hud_texts = [
            f"Speed: {env.car_velocity:.1f}",
            f"Checkpoint: {env.checkpoint_index}/{len(env.track_segments)}",
            f"Laps: {env.laps_completed}",
            f"Reward: {total_reward:.1f}"
        ]

        for i, text in enumerate(hud_texts):
            surface = font.render(text, True, (255, 255, 255))
            screen.blit(surface, (10, 10 + i * 30))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

def compare_models():
    """Compare different saved models"""
    model_files = []
    for run_dir in _list_run_dirs():
        for entry in os.listdir(run_dir):
            if entry.endswith(".pkl"):
                model_files.append(os.path.join(run_dir, entry))
    
    if not model_files:
        print("No saved models found!")
        return
    
    model_files.sort(key=lambda p: (
        os.path.basename(os.path.dirname(p)),
        _parse_generation(p),
        os.path.basename(p)
    ))

    print(f"Found {len(model_files)} models\n")
    
    results = []
    
    # Load cached tracks
    tracks = load_cached_tracks()
    print(f"Loaded {len(tracks)} cached tracks for comparison")
    
    for model_file in model_files:
        with open(model_file, 'rb') as f:
            network = pickle.load(f)
        
        total_rewards = []
        
        # Test on all tracks
        for track_idx, track in enumerate(tracks):
            env = DriftingEnv(track_data=track)
            obs = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = network.forward(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            
            total_rewards.append(total_reward)
        
        avg_reward = np.mean(total_rewards)
        results.append((model_file, avg_reward))
        print(f"{model_file}: Avg Reward = {avg_reward:.2f}")
    
    print(f"\nBest model: {max(results, key=lambda x: x[1])[0]}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "manual":
            test_manual_control()
        elif sys.argv[1] == "compare":
            compare_models()
        elif sys.argv[1] == "model":
            if len(sys.argv) > 2:
                model_path = sys.argv[2]
            else:
                model_path = None
            test_trained_model(model_path)
    else:
        # Default: test the final trained model
        test_trained_model()