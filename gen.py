import numpy as np
import pickle
import os
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button

# Configuration
NUM_TRACKS = 20
TRACKS_DIR = "tracks"
TRACKS_FILE = os.path.join(TRACKS_DIR, "track_cache.pkl")

# Track generation parameters
POINTS_PER_TRACK = 800
BASE_SCALE = 400
TRACK_WIDTH = 40
SCALE_FACTOR = BASE_SCALE / 200  # Auto-scale based on base scale

class TrackGenerator:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
    
    def smooth_waypoints(self, waypoints, sigma=3):
        """Apply Gaussian smoothing"""
        x_smooth = gaussian_filter1d(waypoints[:, 0], sigma=sigma, mode='wrap')
        y_smooth = gaussian_filter1d(waypoints[:, 1], sigma=sigma, mode='wrap')
        return np.column_stack([x_smooth, y_smooth])
    
    def generate_figure_eight(self):
        """Generate a figure-8 style track"""
        t = np.linspace(0, 2*np.pi, 100)
        scale = BASE_SCALE * np.random.uniform(0.8, 1.2)
        
        # Lissajous curve variant
        x = scale * np.sin(t)
        y = scale * 0.5 * np.sin(2*t + np.random.uniform(0, 0.5))
        
        waypoints = np.column_stack([x, y])
        return waypoints
    
    def generate_stadium(self):
        """Generate a stadium/oval with complex infield"""
        waypoints = []
        
        # Long straight 1
        for i in range(30):
            waypoints.append([i * 5 * SCALE_FACTOR, 0])
        
        # Tight hairpin
        radius = np.random.uniform(30, 45) * SCALE_FACTOR
        angles = np.linspace(0, np.pi, 20)
        for a in angles:
            x = 150 * SCALE_FACTOR + radius * np.cos(a)
            y = radius * np.sin(a)
            waypoints.append([x, y])
        
        # Infield section with chicanes
        waypoints.append([140 * SCALE_FACTOR, radius + 20 * SCALE_FACTOR])
        waypoints.append([120 * SCALE_FACTOR, radius + 35 * SCALE_FACTOR])
        waypoints.append([100 * SCALE_FACTOR, radius + 30 * SCALE_FACTOR])
        waypoints.append([80 * SCALE_FACTOR, radius + 40 * SCALE_FACTOR])
        waypoints.append([60 * SCALE_FACTOR, radius + 35 * SCALE_FACTOR])
        waypoints.append([40 * SCALE_FACTOR, radius + 30 * SCALE_FACTOR])
        waypoints.append([20 * SCALE_FACTOR, radius + 25 * SCALE_FACTOR])
        
        # Another hairpin to close
        angles = np.linspace(np.pi, 2*np.pi, 20)
        for a in angles:
            x = 10 * SCALE_FACTOR + radius * np.cos(a)
            y = radius + 25 * SCALE_FACTOR + radius * np.sin(a)
            waypoints.append([x, y])
        
        return np.array(waypoints)
    
    def generate_street_circuit(self):
        """Generate 90-degree street circuit style"""
        waypoints = [[0, 0]]
        current_pos = np.array([0.0, 0.0])
        current_direction = np.array([1.0, 0.0])
        
        num_blocks = np.random.randint(8, 14)
        
        for i in range(num_blocks):
            # Straight section
            straight_length = np.random.uniform(30, 80) * SCALE_FACTOR
            current_pos = current_pos + current_direction * straight_length
            waypoints.append(current_pos.tolist())
            
            # 90-degree corner (or sometimes 45)
            if np.random.rand() < 0.7:
                angle = np.pi/2 * np.random.choice([-1, 1])
            else:
                angle = np.pi/4 * np.random.choice([-1, 1])
            
            # Rotation matrix
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            current_direction = rotation @ current_direction
            
            # Corner arc
            radius = np.random.uniform(25, 40) * SCALE_FACTOR
            arc_angles = np.linspace(0, abs(angle), 8)
            
            for a in arc_angles[1:]:
                turn_matrix = np.array([[np.cos(a * np.sign(angle)), -np.sin(a * np.sign(angle))],
                                       [np.sin(a * np.sign(angle)), np.cos(a * np.sign(angle))]])
                offset = turn_matrix @ (np.array([radius, 0]) * np.sign(angle))
                waypoints.append((current_pos + offset).tolist())
            
            current_pos = waypoints[-1]
        
        return np.array(waypoints)
    
    def generate_mountain_road(self):
        """Generate a winding mountain road style"""
        waypoints = []
        
        # Create a path that winds back and forth
        x = 0
        y = 0
        direction = 1
        
        for i in range(12):
            # Straight climb
            length = np.random.uniform(40, 70) * SCALE_FACTOR
            for j in range(15):
                t = j / 14
                waypoints.append([x + t * length, y])
            
            x += length
            
            # Hairpin turn
            radius = np.random.uniform(30, 50) * SCALE_FACTOR
            y_offset = radius * 2 * direction
            
            angles = np.linspace(0, np.pi, 15)
            for a in angles:
                waypoints.append([
                    x + radius * (1 - np.cos(a)),
                    y + y_offset * (1 - np.cos(a)) / 2
                ])
            
            y += y_offset
            direction *= -1
        
        return np.array(waypoints)
    
    def generate_technical_complex(self):
        """Generate a technical section with many direction changes"""
        waypoints = [[0, 0]]
        current_angle = 0
        current_pos = np.array([0.0, 0.0])
        
        num_features = np.random.randint(20, 30)
        
        for i in range(num_features):
            # Random feature
            feature = np.random.choice(['short_straight', 'quick_left', 'quick_right', 
                                       'medium_corner', 'double_apex'])
            
            if feature == 'short_straight':
                length = np.random.uniform(20, 40) * SCALE_FACTOR
                current_pos = current_pos + np.array([np.cos(current_angle), 
                                                     np.sin(current_angle)]) * length
                waypoints.append(current_pos.tolist())
            
            elif feature == 'quick_left':
                angle_change = np.random.uniform(np.pi/6, np.pi/3)
                current_angle += angle_change
                
                radius = np.random.uniform(25, 40) * SCALE_FACTOR
                arc = np.linspace(0, angle_change, 8)
                for a in arc:
                    offset = radius * np.array([np.sin(a), 1 - np.cos(a)])
                    # Rotate offset
                    rotated = np.array([
                        offset[0] * np.cos(current_angle - angle_change) - offset[1] * np.sin(current_angle - angle_change),
                        offset[0] * np.sin(current_angle - angle_change) + offset[1] * np.cos(current_angle - angle_change)
                    ])
                    waypoints.append((current_pos + rotated).tolist())
                
                current_pos = np.array(waypoints[-1])
            
            elif feature == 'quick_right':
                angle_change = -np.random.uniform(np.pi/6, np.pi/3)
                current_angle += angle_change
                
                radius = np.random.uniform(25, 40) * SCALE_FACTOR
                arc = np.linspace(0, abs(angle_change), 8)
                for a in arc:
                    offset = radius * np.array([np.sin(a), -(1 - np.cos(a))])
                    rotated = np.array([
                        offset[0] * np.cos(current_angle - angle_change) - offset[1] * np.sin(current_angle - angle_change),
                        offset[0] * np.sin(current_angle - angle_change) + offset[1] * np.cos(current_angle - angle_change)
                    ])
                    waypoints.append((current_pos + rotated).tolist())
                
                current_pos = np.array(waypoints[-1])
            
            elif feature == 'medium_corner':
                angle_change = np.random.uniform(-np.pi/2, np.pi/2)
                current_angle += angle_change
                
                radius = np.random.uniform(35, 55) * SCALE_FACTOR
                arc = np.linspace(0, abs(angle_change), 12)
                for a in arc:
                    sign = np.sign(angle_change) if angle_change != 0 else 1
                    offset = radius * np.array([np.sin(a), sign * (1 - np.cos(a))])
                    rotated = np.array([
                        offset[0] * np.cos(current_angle - angle_change) - offset[1] * np.sin(current_angle - angle_change),
                        offset[0] * np.sin(current_angle - angle_change) + offset[1] * np.cos(current_angle - angle_change)
                    ])
                    waypoints.append((current_pos + rotated).tolist())
                
                current_pos = np.array(waypoints[-1])
            
            else:  # double_apex
                for apex in range(2):
                    angle_change = np.random.uniform(-np.pi/4, np.pi/4)
                    current_angle += angle_change
                    
                    radius = np.random.uniform(30, 45) * SCALE_FACTOR
                    arc = np.linspace(0, abs(angle_change), 6)
                    for a in arc:
                        sign = np.sign(angle_change) if angle_change != 0 else 1
                        offset = radius * np.array([np.sin(a), sign * (1 - np.cos(a))])
                        rotated = np.array([
                            offset[0] * np.cos(current_angle - angle_change) - offset[1] * np.sin(current_angle - angle_change),
                            offset[0] * np.sin(current_angle - angle_change) + offset[1] * np.cos(current_angle - angle_change)
                        ])
                        waypoints.append((current_pos + rotated).tolist())
                    
                    current_pos = np.array(waypoints[-1])
        
        return np.array(waypoints)
    
    def generate_mixed_speed(self):
        """Generate track with distinct fast and slow sections"""
        waypoints = []
        
        # Fast section - long straights and sweeping corners
        for i in range(3):
            # Long straight
            for j in range(20):
                waypoints.append([(i * 150 + j * 5) * SCALE_FACTOR, 0])
            
            # Fast sweeper
            radius = np.random.uniform(70, 100) * SCALE_FACTOR
            angles = np.linspace(0, np.pi/3, 12)
            direction = (-1) ** i
            
            for a in angles:
                x = ((i + 1) * 150 - 20) * SCALE_FACTOR + radius * np.sin(a)
                y = radius * (1 - np.cos(a)) * direction
                waypoints.append([x, y])
        
        # Transition
        waypoints.append([400 * SCALE_FACTOR, 50 * SCALE_FACTOR])
        
        # Slow technical section
        x, y = 400 * SCALE_FACTOR, 50 * SCALE_FACTOR
        for i in range(8):
            # Tight chicane
            x += 15 * SCALE_FACTOR
            y += 20 * SCALE_FACTOR * ((-1) ** i)
            waypoints.append([x, y])
            
            x += 15 * SCALE_FACTOR
            y += 15 * SCALE_FACTOR * ((-1) ** i)
            waypoints.append([x, y])
        
        # Return sweep
        angles = np.linspace(0, np.pi, 25)
        radius = 80 * SCALE_FACTOR
        for a in angles:
            x = 500 * SCALE_FACTOR + radius * np.cos(a)
            y = 100 * SCALE_FACTOR + radius * np.sin(a)
            waypoints.append([x, y])
        
        return np.array(waypoints)
    
    def close_track_loop(self, waypoints):
        """Force closure with smooth bridge"""
        gap = waypoints[0] - waypoints[-1]
        gap_dist = np.linalg.norm(gap)
        
        if gap_dist > 20 * SCALE_FACTOR:
            num_bridge = max(20, int(gap_dist / 3))
            bridge = np.linspace(waypoints[-1], waypoints[0], num_bridge)
            waypoints = np.vstack([waypoints, bridge[1:]])
        
        waypoints = np.vstack([waypoints, waypoints[0:1]])
        return waypoints
    
    def interpolate_smooth(self, waypoints):
        """Interpolate with moderate smoothing to preserve character"""
        waypoints = self.close_track_loop(waypoints)
        waypoints = self.smooth_waypoints(waypoints, sigma=4)  # Less smoothing
        
        try:
            tck, u = splprep([waypoints[:, 0], waypoints[:, 1]], 
                            s=len(waypoints) * 1.5,  # Less smoothing
                            per=True, 
                            k=3)
            
            u_new = np.linspace(0, 1, POINTS_PER_TRACK, endpoint=False)
            x_new, y_new = splev(u_new, tck)
            
            smooth_track = np.column_stack([x_new, y_new])
            smooth_track = self.smooth_waypoints(smooth_track, sigma=2)  # Light final pass
            
            return smooth_track
            
        except Exception:
            return None
    
    def calculate_curvature(self, points):
        """Calculate radius of curvature at each point"""
        n = len(points)
        curvatures = []
        
        for i in range(n):
            p1 = points[(i - 10) % n]
            p2 = points[i]
            p3 = points[(i + 10) % n]
            
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p3 - p2)
            c = np.linalg.norm(p1 - p3)
            
            s = (a + b + c) / 2
            area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
            
            if area > 0.01:
                radius = (a * b * c) / (4 * area)
            else:
                radius = 1000
            
            curvatures.append(radius)
        
        return np.array(curvatures)
    
    def validate_centerline_curvature(self, centerline):
        """Ensure safe curvature"""
        curvatures = self.calculate_curvature(centerline)
        min_radius = np.min(curvatures)
        return min_radius > TRACK_WIDTH * 0.8  # More lenient
    
    def check_self_intersection(self, points, min_separation=12):
        """Check for self-intersection"""
        n = len(points)
        check_interval = 5
        
        for i in range(0, n, check_interval):
            p1 = points[i]
            p2 = points[(i + 1) % n]
            
            for j in range(i + 50, min(i + n - 50, n), check_interval):
                p3 = points[j % n]
                p4 = points[(j + 1) % n]
                
                if self.segments_intersect(p1, p2, p3, p4):
                    return True
                
                if j - i > 100:
                    dist = self.point_to_segment_distance(p1, p3, p4)
                    if dist < min_separation:
                        return True
        
        return False
    
    def segments_intersect(self, p1, p2, p3, p4):
        """Check if segments intersect"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def point_to_segment_distance(self, point, seg_start, seg_end):
        """Distance from point to segment"""
        seg_vec = seg_end - seg_start
        point_vec = point - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)
        
        if seg_len_sq == 0:
            return np.linalg.norm(point - seg_start)
        
        t = max(0, min(1, np.dot(point_vec, seg_vec) / seg_len_sq))
        projection = seg_start + t * seg_vec
        
        return np.linalg.norm(point - projection)
    
    def generate_boundaries_constant_width(self, centerline):
        """Generate boundaries with constant width"""
        n = len(centerline)
        
        # Determine winding
        signed_area = 0
        for i in range(n):
            j = (i + 1) % n
            signed_area += (centerline[j, 0] - centerline[i, 0]) * (centerline[j, 1] + centerline[i, 1])
        
        flip_direction = 1 if signed_area > 0 else -1
        
        inner = []
        outer = []
        half_width = TRACK_WIDTH / 2
        
        for i in range(n):
            next_idx = (i + 8) % n
            prev_idx = (i - 8) % n
            
            tangent = centerline[next_idx] - centerline[prev_idx]
            tangent_norm = np.linalg.norm(tangent)
            
            if tangent_norm > 0:
                tangent = tangent / tangent_norm
                perp = np.array([-tangent[1], tangent[0]]) * flip_direction
                
                inner.append(centerline[i] + perp * half_width)
                outer.append(centerline[i] - perp * half_width)
            else:
                inner.append(centerline[i])
                outer.append(centerline[i])
        
        inner = np.array(inner)
        outer = np.array(outer)
        
        # Moderate smoothing
        inner = self.smooth_waypoints(inner, sigma=4)
        outer = self.smooth_waypoints(outer, sigma=4)
        
        # Closure
        inner[0] = inner[-1] = (inner[0] + inner[-1]) / 2
        outer[0] = outer[-1] = (outer[0] + outer[-1]) / 2
        
        return inner, outer
    
    def validate_boundaries(self, inner, outer, centerline):
        """Ensure correct orientation"""
        centroid = np.mean(centerline, axis=0)
        
        sample_indices = np.linspace(0, len(centerline)-1, 30, dtype=int)
        
        avg_inner = np.mean([np.linalg.norm(inner[i] - centroid) for i in sample_indices])
        avg_outer = np.mean([np.linalg.norm(outer[i] - centroid) for i in sample_indices])
        
        if avg_inner > avg_outer:
            return outer, inner
        
        return inner, outer
    
    def generate_track(self, track_id):
        """Generate a varied track"""
        strategies = [
            self.generate_figure_eight,
            self.generate_stadium,
            self.generate_street_circuit,
            self.generate_mountain_road,
            self.generate_technical_complex,
            self.generate_mixed_speed,
        ]
        
        max_attempts = 400
        
        for attempt in range(max_attempts):
            try:
                strategy = np.random.choice(strategies)
                waypoints = strategy()
                
                centerline = self.interpolate_smooth(waypoints)
                
                if centerline is None:
                    continue
                
                if not self.validate_centerline_curvature(centerline):
                    continue
                
                if not self.check_self_intersection(centerline, min_separation=TRACK_WIDTH * 1.5):
                    inner, outer = self.generate_boundaries_constant_width(centerline)
                    inner, outer = self.validate_boundaries(inner, outer, centerline)
                    
                    inner_valid = not self.check_self_intersection(inner, min_separation=5)
                    outer_valid = not self.check_self_intersection(outer, min_separation=5)
                    
                    if inner_valid and outer_valid:
                        return {
                            'id': track_id,
                            'centerline': centerline,
                            'inner_boundary': inner,
                            'outer_boundary': outer,
                            'widths': np.full(len(centerline), TRACK_WIDTH)
                        }
            
            except Exception:
                continue
        
        raise RuntimeError(f"Failed after {max_attempts} attempts")

# [Keep TrackSelector exactly the same as before]
class TrackSelector:
    def __init__(self, tracks):
        self.tracks = tracks
        self.selected = [True] * len(tracks)
        
        n_tracks = len(tracks)
        self.cols = int(np.ceil(np.sqrt(n_tracks)))
        self.rows = int(np.ceil(n_tracks / self.cols))
        
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.canvas.manager.set_window_title('Track Selection')
        self.fig.suptitle('Click on tracks to toggle selection (Green = Selected)', fontsize=14, fontweight='bold')
        
        self.axes = []
        self.patches = []
        
        for i, track in enumerate(tracks):
            ax = self.fig.add_subplot(self.rows, self.cols, i + 1)
            self.axes.append(ax)
            
            centerline = track['centerline']
            inner = track['inner_boundary']
            outer = track['outer_boundary']
            
            ax.plot(inner[:, 0], inner[:, 1], 'r-', linewidth=1.5, alpha=0.8)
            ax.plot(outer[:, 0], outer[:, 1], 'b-', linewidth=1.5, alpha=0.8)
            ax.plot(centerline[:, 0], centerline[:, 1], 'k--', linewidth=0.5, alpha=0.3)
            
            ax.set_aspect('equal')
            ax.set_title(f'Track {i}', fontsize=10)
            ax.axis('off')
            
            patch = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                            fill=False, edgecolor='green', linewidth=3)
            ax.add_patch(patch)
            self.patches.append(patch)
            ax.set_picker(True)
        
        self.setup_buttons()
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        plt.tight_layout()
    
    def setup_buttons(self):
        ax_save = plt.axes([0.35, 0.02, 0.12, 0.04])
        self.btn_save = Button(ax_save, 'Save Selected', color='lightgreen')
        self.btn_save.on_clicked(self.save_selected)
        
        ax_all = plt.axes([0.48, 0.02, 0.12, 0.04])
        self.btn_all = Button(ax_all, 'Select All', color='lightblue')
        self.btn_all.on_clicked(self.select_all)
        
        ax_none = plt.axes([0.61, 0.02, 0.12, 0.04])
        self.btn_none = Button(ax_none, 'Deselect All', color='lightcoral')
        self.btn_none.on_clicked(self.deselect_all)
    
    def on_pick(self, event):
        ax = event.artist
        if ax in self.axes:
            idx = self.axes.index(ax)
            self.selected[idx] = not self.selected[idx]
            self.patches[idx].set_edgecolor('green' if self.selected[idx] else 'red')
            self.patches[idx].set_linewidth(3)
            self.fig.canvas.draw()
    
    def select_all(self, event):
        for i in range(len(self.selected)):
            self.selected[i] = True
            self.patches[i].set_edgecolor('green')
            self.patches[i].set_linewidth(3)
        self.fig.canvas.draw()
    
    def deselect_all(self, event):
        for i in range(len(self.selected)):
            self.selected[i] = False
            self.patches[i].set_edgecolor('red')
            self.patches[i].set_linewidth(3)
        self.fig.canvas.draw()
    
    def save_selected(self, event):
        selected_tracks = [track for i, track in enumerate(self.tracks) if self.selected[i]]
        
        if len(selected_tracks) == 0:
            print("No tracks selected!")
            return
        
        for i, track in enumerate(selected_tracks):
            track['id'] = i
        
        os.makedirs(TRACKS_DIR, exist_ok=True)
        
        with open(TRACKS_FILE, 'wb') as f:
            pickle.dump(selected_tracks, f)
        
        for track in selected_tracks:
            fig, ax = plt.subplots(figsize=(10, 10))
            
            centerline = track['centerline']
            inner = track['inner_boundary']
            outer = track['outer_boundary']
            
            ax.plot(centerline[:, 0], centerline[:, 1], 'b-', linewidth=1, label='Centerline', alpha=0.5)
            ax.plot(inner[:, 0], inner[:, 1], 'r-', linewidth=2, label='Inner Boundary')
            ax.plot(outer[:, 0], outer[:, 1], 'g-', linewidth=2, label='Outer Boundary')
            
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_title(f"Track {track['id']}")
            
            viz_path = os.path.join(TRACKS_DIR, f"track_{track['id']:03d}.png")
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"\n{'='*60}")
        print(f"✓ Saved {len(selected_tracks)} tracks to {TRACKS_FILE}")
        print(f"{'='*60}\n")
        
        plt.close(self.fig)
    
    def show(self):
        plt.show()

def generate_and_select_tracks():
    generator = TrackGenerator()
    tracks = []
    
    print(f"Generating {NUM_TRACKS} tracks...\n")
    
    for i in range(NUM_TRACKS):
        try:
            track = generator.generate_track(track_id=i)
            tracks.append(track)
            print(f"✓ Track {i+1}/{NUM_TRACKS}")
        except RuntimeError as e:
            print(f"✗ Track {i+1}/{NUM_TRACKS} - {e}")
    
    print(f"\nGenerated {len(tracks)} valid tracks")
    print("\nOpening track selector...")
    
    selector = TrackSelector(tracks)
    selector.show()

def load_tracks():
    if not os.path.exists(TRACKS_FILE):
        raise FileNotFoundError(f"Track cache not found.")
    
    with open(TRACKS_FILE, 'rb') as f:
        tracks = pickle.load(f)
    
    print(f"Loaded {len(tracks)} tracks from cache")
    return tracks

if __name__ == "__main__":
    generate_and_select_tracks()