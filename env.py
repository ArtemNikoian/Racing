import numpy as np
import math
from scipy.spatial import KDTree

class DriftingEnv:
    def __init__(self, render_mode=None, track_data=None):
        self.render_mode = render_mode
        
        # Car properties
        self.car_x = 0
        self.car_y = 0
        self.car_angle = 0
        self.car_velocity = 0
        self.car_angular_velocity = 0
        self.max_velocity = 15
        self.max_angular_velocity = 5
        
        # Track properties
        self.track_width = 20
        self.track_segments = []
        self.centerline_points = []
        self.track_boundaries = []
        self.checkpoint_index = 0
        self.laps_completed = 0
        self.last_checkpoint_side = None
        self.last_checkpoint_inside = False
        self.prev_car_x = 0
        self.prev_car_y = 0
        self._visual_initialized = False
        self._visual_fig = None
        self._visual_ax = None
        
        # NEW: Store track data
        self.track_data = track_data
        
        # Physics
        self.friction = 0.98
        self.acceleration = 0.3
        self.turn_speed = 0.08
        self.brake_power = 0.2
        
        # Episode
        self.steps = 0
        self.max_steps = 400
        
        # Observation configuration
        self.lookahead_distances = [20, 40, 60, 80]  # Distances ahead to sample track
        
        if self.track_data is not None:
            # Load track from provided data
            self._load_track_from_data(self.track_data)
        else:
            # Generate random track
            self.generate_track()
        self.reset()
        
    def _load_track_from_data(self, track_data):
        """Load track from pre-generated track data"""
        centerline = track_data['centerline']
        inner_boundary = track_data['inner_boundary']
        outer_boundary = track_data['outer_boundary']
        
        # Store full centerline points for rendering
        self.centerline_points = centerline
        
        # NEW: Build KD-tree for fast nearest neighbor lookup
        self.centerline_kdtree = KDTree(centerline)
        
        # Calculate total track length along centerline
        total_length = 0
        lengths = []
        for i in range(len(centerline)):
            next_idx = (i + 1) % len(centerline)
            length = np.linalg.norm(centerline[next_idx] - centerline[i])
            lengths.append(length)
            total_length += length
        
        # Place 64 checkpoints evenly along track
        num_checkpoints = 64
        self.track_segments = []
        target_spacing = total_length / num_checkpoints
        
        current_length = 0
        centerline_idx = 0
        for checkpoint_idx in range(num_checkpoints):
            target_length = checkpoint_idx * target_spacing
            
            # Find which centerline segment this checkpoint is on
            while centerline_idx < len(centerline) and current_length + lengths[centerline_idx] < target_length:
                current_length += lengths[centerline_idx]
                centerline_idx += 1
            
            # Wrap around if needed
            if centerline_idx >= len(centerline):
                centerline_idx = centerline_idx % len(centerline)
                current_length = 0
            
            # Interpolate position within segment
            remaining = target_length - current_length
            segment_length = lengths[centerline_idx] if lengths[centerline_idx] > 0 else 1
            t = remaining / segment_length if segment_length > 0 else 0
            
            p1 = centerline[centerline_idx]
            next_idx = (centerline_idx + 1) % len(centerline)
            p2 = centerline[next_idx]
            
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            
            # Calculate direction from tangent
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            direction = np.arctan2(dy, dx)
            
            self.track_segments.append({
                'x': x,
                'y': y,
                'direction': direction,
                'checkpoint': checkpoint_idx,
                'perp': np.array([-np.sin(direction), np.cos(direction)])
            })
        
        # Pre-compute track boundaries from dense boundaries
        self.track_boundaries = []
        for i in range(len(centerline)):
            p1 = centerline[i]
            next_idx = (i + 1) % len(centerline)
            p2 = centerline[next_idx]
            
            # Calculate perpendicular direction
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = np.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                perp_x = -dy / length
                perp_y = dx / length
            else:
                perp_x, perp_y = 0, 0
            
            # Use average width at this point
            inner_p1 = inner_boundary[i]
            outer_p1 = outer_boundary[i]
            width = np.linalg.norm(outer_p1 - inner_p1)
            
            # Store segment with boundary offsets
            self.track_boundaries.append({
                'p1': p1,
                'p2': p2,
                'perp': np.array([perp_x, perp_y])
            })
        
        # Update track width to average width
        if 'widths' in track_data:
            self.track_width = np.mean(track_data['widths'])
        else:
            # Calculate from boundaries
            widths = []
            for i in range(len(inner_boundary)):
                width = np.linalg.norm(outer_boundary[i] - inner_boundary[i])
                widths.append(width)
            self.track_width = np.mean(widths) if widths else 20
        
        # Store dense boundaries for collision detection
        self.inner_boundary = inner_boundary
        self.outer_boundary = outer_boundary
    
    def generate_track(self):
        """Generate a track from 8 random points connected in order"""
        num_points = 8
        points = np.random.rand(num_points, 2)
        
        # Scale to reasonable size (centered around origin)
        scale = 300
        points = (points - 0.5) * scale
        
        # Sort points by angle from centroid to avoid self-intersection
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        sorted_indices = np.argsort(angles)
        points = points[sorted_indices]
        
        # Store the centerline points
        self.centerline_points = points
        
        # Build KD-tree for fast nearest neighbor lookup
        self.centerline_kdtree = KDTree(points)
        
        # Calculate total track length
        total_length = 0
        segment_lengths = []
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            length = np.linalg.norm(p2 - p1)
            segment_lengths.append(length)
            total_length += length
        
        # Place 64 checkpoints evenly along track
        num_checkpoints = 64
        self.track_segments = []
        target_spacing = total_length / num_checkpoints
        
        current_length = 0
        segment_idx = 0
        for checkpoint_idx in range(num_checkpoints):
            target_length = checkpoint_idx * target_spacing
            
            # Find which segment this checkpoint is on
            while segment_idx < len(points) and current_length + segment_lengths[segment_idx] < target_length:
                current_length += segment_lengths[segment_idx]
                segment_idx += 1
            
            # Interpolate position within segment
            if segment_idx >= len(points):
                segment_idx = len(points) - 1
            
            remaining = target_length - current_length
            segment_length = segment_lengths[segment_idx] if segment_idx < len(segment_lengths) else 1
            t = remaining / segment_length if segment_length > 0 else 0
            
            p1 = points[segment_idx]
            p2 = points[(segment_idx + 1) % len(points)]
            
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            
            # Calculate direction
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            direction = np.arctan2(dy, dx)
            
            self.track_segments.append({
                'x': x,
                'y': y,
                'direction': direction,
                'checkpoint': checkpoint_idx,
                'perp': np.array([-np.sin(direction), np.cos(direction)])
            })
        
        # Pre-compute track boundaries for fast collision detection
        self.track_boundaries = []
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            
            # Calculate perpendicular direction
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = np.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                perp_x = -dy / length
                perp_y = dx / length
            else:
                perp_x, perp_y = 0, 0
            
            # Store segment with boundary offsets
            self.track_boundaries.append({
                'p1': p1,
                'p2': p2,
                'perp': np.array([perp_x, perp_y])
            })
        
    def reset(self, seed=None, options=None):
        # Ensure track is loaded
        if len(self.track_segments) == 0:
            if self.track_data is not None:
                self._load_track_from_data(self.track_data)
            else:
                self.generate_track()
        
        # Start at first segment
        self.car_x = self.track_segments[0]['x']
        self.car_y = self.track_segments[0]['y']
        self.car_angle = self.track_segments[0]['direction']
        self.car_velocity = 0
        self.car_angular_velocity = 0

        self.prev_car_x = self.car_x - np.cos(self.car_angle) * 1e-3
        self.prev_car_y = self.car_y - np.sin(self.car_angle) * 1e-3
        self.checkpoint_index = 0
        self.laps_completed = 0
        self.steps = 0
        self.last_checkpoint_side = -1
        self.last_checkpoint_inside = False
        return self.get_observation()
    
    def get_observation(self):
        """Get observation with lateral position and forward track samples"""
        obs = []
        
        # 1. Lateral position on track (normalized -1 to 1)
        lateral_pos = self._get_lateral_position()
        obs.append(lateral_pos)
        
        # 2. Forward track samples at multiple distances
        for distance in self.lookahead_distances:
            left_dist, right_dist = self._get_track_edges_at_distance(distance)
            obs.append(left_dist)
            obs.append(right_dist)
        
        # 3. Velocity (normalized)
        obs.append(self.car_velocity / self.max_velocity)
        
        # 4. Angular velocity (normalized)
        obs.append(self.car_angular_velocity / self.max_angular_velocity)
        
        # 5. Forward progress to checkpoint
        next_checkpoint = self.track_segments[self.checkpoint_index]
        to_checkpoint = np.array([
            next_checkpoint['x'] - self.car_x,
            next_checkpoint['y'] - self.car_y
        ])
        forward_progress = np.dot(to_checkpoint, 
                                  [np.cos(self.car_angle), np.sin(self.car_angle)])
        obs.append(forward_progress / 100.0)  # Normalize
        
        return np.array(obs, dtype=np.float32)
    
    def _get_lateral_position(self):
        """Get normalized lateral position on track (-1 = left edge, +1 = right edge)"""
        # Find nearest centerline point
        _, nearest_idx = self.centerline_kdtree.query([self.car_x, self.car_y])
        
        centerline_point = self.centerline_points[nearest_idx]
        next_idx = (nearest_idx + 1) % len(self.centerline_points)
        next_point = self.centerline_points[next_idx]
        
        # Calculate perpendicular direction
        dx = next_point[0] - centerline_point[0]
        dy = next_point[1] - centerline_point[1]
        length = np.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            perp_x = -dy / length
            perp_y = dx / length
        else:
            return 0.0
        
        # Calculate lateral offset
        to_car_x = self.car_x - centerline_point[0]
        to_car_y = self.car_y - centerline_point[1]
        lateral_offset = to_car_x * perp_x + to_car_y * perp_y
        
        # Normalize to -1 to 1
        return np.clip(lateral_offset / (self.track_width / 2), -1.0, 1.0)
    
    def _get_track_edges_at_distance(self, distance):
        """Get distances to left and right track edges at a point ahead
        Returns (left_distance, right_distance)"""
        
        # Point ahead in car's direction
        look_x = self.car_x + distance * np.cos(self.car_angle)
        look_y = self.car_y + distance * np.sin(self.car_angle)
        
        # Find nearest centerline point to lookahead point
        _, nearest_idx = self.centerline_kdtree.query([look_x, look_y])
        
        centerline_point = self.centerline_points[nearest_idx]
        next_idx = (nearest_idx + 1) % len(self.centerline_points)
        next_point = self.centerline_points[next_idx]
        
        # Calculate perpendicular direction (left side of track)
        dx = next_point[0] - centerline_point[0]
        dy = next_point[1] - centerline_point[1]
        length = np.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            perp_x = -dy / length
            perp_y = dx / length
        else:
            # Fallback if points are coincident
            return self.track_width / 2, self.track_width / 2
        
        # Calculate left and right edge positions
        half_width = self.track_width / 2
        left_edge = centerline_point + np.array([perp_x * half_width, perp_y * half_width])
        right_edge = centerline_point - np.array([perp_x * half_width, perp_y * half_width])
        
        # Distance from lookahead point to edges
        left_dist = np.linalg.norm(left_edge - [look_x, look_y])
        right_dist = np.linalg.norm(right_edge - [look_x, look_y])
        
        # Normalize by max expected distance
        max_dist = self.track_width * 2
        left_dist = left_dist / max_dist
        right_dist = right_dist / max_dist
        
        return left_dist, right_dist
    
    def is_on_track(self, x, y, is_car=False):
        """Check if a point is within track boundaries using KD-tree"""
        # Use KD-tree for cached tracks
        if hasattr(self, 'centerline_kdtree'):
            point = np.array([x, y])
            
            # Find closest point on centerline using KD-tree (FAST!)
            dist_to_center, closest_idx = self.centerline_kdtree.query(point)
            
            # Get boundaries at closest point
            inner_pt = self.inner_boundary[closest_idx]
            outer_pt = self.outer_boundary[closest_idx]
            
            # Calculate track width at this point
            track_width_here = np.linalg.norm(outer_pt - inner_pt)
            
            # Point is on track if it's within track width
            if dist_to_center < track_width_here * 0.7:
                dist_to_inner = np.linalg.norm(point - inner_pt)
                dist_to_outer = np.linalg.norm(point - outer_pt)
                if dist_to_inner < track_width_here * 1.2 and dist_to_outer < track_width_here * 1.2:
                    return True
            return False
        
        # Fallback to original method (for randomly generated tracks without KD-tree)
        for boundary in self.track_boundaries:
            p1 = boundary['p1']
            p2 = boundary['p2']
            
            # Find closest point on segment to (x, y)
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            if dx == 0 and dy == 0:
                closest = p1
            else:
                t = ((x - p1[0]) * dx + (y - p1[1]) * dy) / (dx*dx + dy*dy)
                t = max(0, min(1, t))
                closest = p1 + t * (p2 - p1)
            
            # Check distance to closest point on this segment
            dist = np.sqrt((x - closest[0])**2 + (y - closest[1])**2)
            if dist <= self.track_width:
                return True
        
        return False

    def cast_ray_to_track_edge(self, angle):
        """Cast a ray and return distance to track edge using binary search"""
        max_ray_distance = 100
        
        # Binary search for track edge
        left, right = 0, max_ray_distance
        
        # First check if max distance is on track (if so, return max)
        x_far = self.car_x + max_ray_distance * math.cos(angle)
        y_far = self.car_y + max_ray_distance * math.sin(angle)
        if self.is_on_track(x_far, y_far):
            return max_ray_distance
        
        # Binary search for the edge
        while right - left > 1:
            mid = (left + right) / 2
            x = self.car_x + mid * math.cos(angle)
            y = self.car_y + mid * math.sin(angle)
            
            if self.is_on_track(x, y):
                left = mid
            else:
                right = mid
        
        return left
    
    def step(self, action):
        """
        action: [acceleration, steering]
        acceleration: -1 to 1 (brake to accelerate)
        steering: -1 to 1 (left to right)
        """
        acceleration, steering = action
        
        # Update velocity with different brake/acceleration power
        if acceleration > 0:
            self.car_velocity += acceleration * self.acceleration
        else:
            self.car_velocity += acceleration * self.brake_power  # Braking is less effective

        prev_x = self.car_x
        prev_y = self.car_y
        self.car_velocity *= self.friction
        self.car_velocity = np.clip(self.car_velocity, -self.max_velocity, self.max_velocity)
        
        # Update angular velocity (drifting mechanics)
        # Reduced turn effectiveness at high speeds for realism (like real racing)
        speed_ratio = abs(self.car_velocity) / self.max_velocity
        turn_effectiveness = max(0.3, 1.0 - speed_ratio * 0.7)  # Scales from 1.0 at rest to 0.3 at max speed
        self.car_angular_velocity = steering * self.turn_speed * turn_effectiveness
        
        # Update angle and position
        self.car_angle += self.car_angular_velocity
        self.car_angle = self.car_angle % (2 * math.pi)
        
        self.car_x += self.car_velocity * math.cos(self.car_angle)
        self.car_y += self.car_velocity * math.sin(self.car_angle)
        self.prev_car_x = prev_x
        self.prev_car_y = prev_y
        
        # Calculate reward
        reward = self.calculate_reward()
        
        # Check termination
        self.steps += 1
        terminated = False
        truncated = self.steps >= self.max_steps
        
        if not self.is_on_track(self.car_x, self.car_y, is_car=True):
            terminated = True
        
        return self.get_observation(), reward, terminated, truncated, {}
    
    def calculate_reward(self):
        """Calculate reward based on progress"""
        reward = 0
        
        # Get current and next checkpoint
        next_checkpoint = self.track_segments[self.checkpoint_index]
        
        prev_side = self._checkpoint_side(next_checkpoint, self.prev_car_x, self.prev_car_y)
        side = self._checkpoint_side(next_checkpoint, self.car_x, self.car_y)
        was_inside = self.last_checkpoint_inside
        inside_box = self._is_within_checkpoint_box(next_checkpoint, self.car_x, self.car_y)
        self.last_checkpoint_inside = inside_box
        
        # Vector from checkpoint to car for distance checks
        to_car_x = self.car_x - next_checkpoint['x']
        to_car_y = self.car_y - next_checkpoint['y']
        
        crossed = (prev_side * side < 0)
        entered_box = inside_box and not was_inside
        
        if crossed or entered_box:
            # Validate car is roughly going in track direction (within 90 degrees)
            car_direction = self.car_angle
            track_direction = next_checkpoint['direction']
            
            angle_diff = abs(car_direction - track_direction)
            angle_diff = min(angle_diff, 2*np.pi - angle_diff)  # Shortest angle
            
            if angle_diff < np.pi/2:  # Within 90 degrees
                # Also check if car is within track bounds at checkpoint
                dist_from_checkpoint = np.sqrt(to_car_x**2 + to_car_y**2)
                
                if dist_from_checkpoint < self.track_width * 1.5 or inside_box:
                    # Valid checkpoint pass!
                    reward += 1
                    self.checkpoint_index = (self.checkpoint_index + 1) % len(self.track_segments)
                    self.last_checkpoint_inside = False
                    
                    # Calculate which side of the NEW checkpoint we're on
                    new_checkpoint = self.track_segments[self.checkpoint_index]
                    new_side = self._checkpoint_side(new_checkpoint, self.car_x, self.car_y)
                    if new_side == 0:
                        offset_x = self.car_x + np.cos(self.car_angle) * 1e-3
                        offset_y = self.car_y + np.sin(self.car_angle) * 1e-3
                        new_side = self._checkpoint_side(new_checkpoint, offset_x, offset_y)
                    self.last_checkpoint_side = new_side if new_side != 0 else -1
                    
                    if self.checkpoint_index == 0:
                        self.laps_completed += 1
                        reward += 40
                else:
                    # Too far from checkpoint center when crossing
                    self.last_checkpoint_side = side if side != 0 else prev_side
            else:
                # Wrong direction
                self.last_checkpoint_side = side if side != 0 else prev_side
        else:
            self.last_checkpoint_side = side if side != 0 else prev_side
        
        return reward

    def _is_within_checkpoint_box(self, checkpoint, x, y):
        dir_vec = np.array([np.cos(checkpoint['direction']), np.sin(checkpoint['direction'])])
        rel = np.array([x - checkpoint['x'], y - checkpoint['y']])
        along = np.dot(rel, dir_vec)
        perp = np.dot(rel, checkpoint['perp'])
        half_length = self.track_width * 0.5
        half_width = self.track_width
        return abs(along) <= half_length and abs(perp) <= half_width

    def _checkpoint_side(self, checkpoint, x, y, epsilon=1e-6):
        perp = checkpoint['perp']
        dot = (x - checkpoint['x']) * perp[0] + (y - checkpoint['y']) * perp[1]
        if dot > epsilon:
            return 1
        if dot < -epsilon:
            return -1
        return 0

    def render_visual(self):
        """Visual rendering with matplotlib for debugging"""
        import matplotlib.pyplot as plt
        
        if not self._visual_initialized:
            plt.ion()
            self._visual_fig, self._visual_ax = plt.subplots(figsize=(8, 8))
            self._visual_initialized = True
        else:
            plt.figure(self._visual_fig.number)
            self._visual_ax.clear()
        
        ax = self._visual_ax
        
        # Draw track boundaries
        for boundary in self.track_boundaries:
            p1, p2 = boundary['p1'], boundary['p2']
            perp = boundary['perp'] * self.track_width
            
            # Inner boundary
            ax.plot([p1[0] - perp[0], p2[0] - perp[0]], 
                    [p1[1] - perp[1], p2[1] - perp[1]], 'k-', linewidth=1)
            # Outer boundary
            ax.plot([p1[0] + perp[0], p2[0] + perp[0]], 
                    [p1[1] + perp[1], p2[1] + perp[1]], 'k-', linewidth=1)
        
        # Draw checkpoint lines
        for i, checkpoint in enumerate(self.track_segments):
            perp = checkpoint['perp'] * self.track_width
            x, y = checkpoint['x'], checkpoint['y']
            
            color = 'g' if i == self.checkpoint_index else 'b'
            alpha = 1.0 if i == self.checkpoint_index else 0.3
            
            ax.plot([x - perp[0], x + perp[0]], 
                    [y - perp[1], y + perp[1]], color=color, linewidth=2, alpha=alpha)
        
        # Draw car
        ax.plot(self.car_x, self.car_y, 'ro', markersize=10)
        
        # Draw car direction
        car_front_x = self.car_x + 10 * np.cos(self.car_angle)
        car_front_y = self.car_y + 10 * np.sin(self.car_angle)
        ax.arrow(self.car_x, self.car_y, car_front_x - self.car_x, car_front_y - self.car_y, 
                 head_width=5, head_length=5, fc='r', ec='r', length_includes_head=True)
        
        ax.set_aspect('equal', 'box')
        ax.set_title(f'Checkpoint: {self.checkpoint_index}/{len(self.track_segments)}, Vel: {self.car_velocity:.1f}')
        plt.draw()
        plt.pause(0.01)
    
    def render(self):
        """Simple text-based rendering for debugging"""
        if self.render_mode == "visual":
            self.render_visual()
        elif self.render_mode == "human":
            print(f"Pos: ({self.car_x:.1f}, {self.car_y:.1f}), "
                  f"Vel: {self.car_velocity:.1f}, "
                  f"Angle: {math.degrees(self.car_angle):.1f}°, "
                  f"Checkpoint: {self.checkpoint_index}/{len(self.track_segments)}, "
                  f"Laps: {self.laps_completed}")