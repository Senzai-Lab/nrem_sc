import numpy as np
import pynapple as nap
import pygfx as gfx
from rendercanvas.glfw import RenderCanvas, loop
import matplotlib.pyplot as plt


def _extract_xyz(data: nap.TsdFrame):
    """Extract x, y, z arrays from a TsdFrame as float32."""
    if 'x' in data.columns and 'y' in data.columns and 'z' in data.columns:
        x = data['x'].values.astype("float32")
        y = data['y'].values.astype("float32")
        z = data['z'].values.astype("float32")
    elif data.values.shape[1] >= 3:
        x = data.values[:, 0].astype("float32")
        y = data.values[:, 1].astype("float32")
        z = data.values[:, 2].astype("float32")
    else:
        # Fallback to 2D with z=0
        x = data.values[:, 0].astype("float32")
        y = data.values[:, 1].astype("float32")
        z = np.zeros_like(x)
    return x, y, z


def _ensure_rgba(colors: np.ndarray) -> np.ndarray:
    """Ensure colors are float32 RGBA (N, 4). Adds alpha=1 if RGB."""
    colors = np.asarray(colors, dtype="float32")
    if colors.ndim == 2 and colors.shape[1] == 3:
        alpha = np.ones((len(colors), 1), dtype="float32")
        colors = np.hstack([colors, alpha])
    return colors


class FirstPersonController:
    """
    First-person camera controller with WASD movement and mouse look.
    
    Controls:
        WASD: Move forward/left/backward/right
        Q/E: Move up/down
        Right-click + drag: Look around
        Scroll wheel: Adjust movement speed
    """
    
    def __init__(self, camera, canvas, move_speed=0.5, look_sensitivity=0.003):
        self.camera = camera
        self.canvas = canvas
        self.move_speed = move_speed
        self.look_sensitivity = look_sensitivity
        
        # Camera orientation (Euler angles)
        self._yaw = 0.0    # Rotation around Y axis (left/right)
        self._pitch = 0.0  # Rotation around X axis (up/down)
        self._roll = 0.0   # Rotation around Z axis (tilt)
        
        # Smooth look targets
        self._target_yaw = 0.0
        self._target_pitch = 0.0
        self._target_roll = 0.0
        self._look_smoothing = 0.3  # Lower = smoother but more lag
        
        # Movement state
        self._keys_pressed = set()
        self._mouse_down = False
        self._last_mouse_pos = None
        
        # Register events on canvas
        canvas.add_event_handler(self._on_key_down, "key_down")
        canvas.add_event_handler(self._on_key_up, "key_up")
        canvas.add_event_handler(self._on_pointer_down, "pointer_down")
        canvas.add_event_handler(self._on_pointer_up, "pointer_up")
        canvas.add_event_handler(self._on_pointer_move, "pointer_move")
        canvas.add_event_handler(self._on_wheel, "wheel")
    
    def _on_key_down(self, event):
        self._keys_pressed.add(event["key"].lower())
    
    def _on_key_up(self, event):
        self._keys_pressed.discard(event["key"].lower())
    
    def _on_pointer_down(self, event):
        # Left-click for mouse look
        if event["button"] == 1:
            self._mouse_down = True
            self._last_mouse_pos = (event["x"], event["y"])
    
    def _on_pointer_up(self, event):
        if event["button"] == 1:
            self._mouse_down = False
            self._last_mouse_pos = None
    
    def _on_pointer_move(self, event):
        if self._mouse_down and self._last_mouse_pos is not None:
            dx = event["x"] - self._last_mouse_pos[0]
            dy = event["y"] - self._last_mouse_pos[1]
            
            # Update target yaw and pitch (will be smoothed in update())
            self._target_yaw -= dx * self.look_sensitivity
            self._target_pitch -= dy * self.look_sensitivity
            
            # Clamp pitch to avoid gimbal lock
            self._target_pitch = np.clip(self._target_pitch, -np.pi / 2 + 0.1, np.pi / 2 - 0.1)
            
            self._last_mouse_pos = (event["x"], event["y"])
    
    def _on_wheel(self, event):
        # Adjust movement speed with scroll wheel
        if event["dy"] > 0:
            self.move_speed *= 0.9
        else:
            self.move_speed *= 1.1
        self.move_speed = np.clip(self.move_speed, 0.01, 10.0)
    
    def _update_camera_rotation(self):
        """Update camera rotation from yaw, pitch, and roll."""
        # Create rotation quaternion from Euler angles
        # Order: yaw (Y) * pitch (X) * roll (Z)
        cy, sy = np.cos(self._yaw / 2), np.sin(self._yaw / 2)
        cp, sp = np.cos(self._pitch / 2), np.sin(self._pitch / 2)
        cr, sr = np.cos(self._roll / 2), np.sin(self._roll / 2)
        
        # Quaternion multiplication: yaw * pitch * roll
        qw = cy * cp * cr + sy * sp * sr
        qx = cy * sp * cr + sy * cp * sr
        qy = sy * cp * cr - cy * sp * sr
        qz = cy * cp * sr - sy * sp * cr
        
        self.camera.local.rotation = (qx, qy, qz, qw)
    
    def get_forward_vector(self):
        """Get the camera's forward direction vector."""
        # Forward is -Z in camera space, rotated by yaw and pitch
        forward = np.array([
            -np.sin(self._yaw) * np.cos(self._pitch),
            np.sin(self._pitch),
            -np.cos(self._yaw) * np.cos(self._pitch)
        ])
        return forward
    
    def get_right_vector(self):
        """Get the camera's right direction vector."""
        right = np.array([
            np.cos(self._yaw),
            0,
            -np.sin(self._yaw)
        ])
        return right
    
    def update(self):
        """Update camera position based on keys pressed. Call this each frame."""
        moved = False
        
        # Smooth look interpolation
        if (abs(self._yaw - self._target_yaw) > 0.0001 or 
            abs(self._pitch - self._target_pitch) > 0.0001 or
            abs(self._roll - self._target_roll) > 0.0001):
            self._yaw += (self._target_yaw - self._yaw) * self._look_smoothing
            self._pitch += (self._target_pitch - self._pitch) * self._look_smoothing
            self._roll += (self._target_roll - self._roll) * self._look_smoothing
            self._update_camera_rotation()
            moved = True
        
        # Q/E for roll rotation (tilt left/right)
        if "q" in self._keys_pressed:
            self._target_roll += 0.03
            moved = True
        if "e" in self._keys_pressed:
            self._target_roll -= 0.03
            moved = True
        
        if not self._keys_pressed:
            return moved
        
        pos = np.array(self.camera.local.position)
        forward = self.get_forward_vector()
        right = self.get_right_vector()
        up = np.array([0, 1, 0])
        
        if "w" in self._keys_pressed:
            pos += forward * self.move_speed
            moved = True
        if "s" in self._keys_pressed:
            pos -= forward * self.move_speed
            moved = True
        if "a" in self._keys_pressed:
            pos -= right * self.move_speed
            moved = True
        if "d" in self._keys_pressed:
            pos += right * self.move_speed
            moved = True
        if "z" in self._keys_pressed:
            pos += up * self.move_speed
            moved = True
        if "x" in self._keys_pressed:
            pos -= up * self.move_speed
            moved = True
        
        if moved:
            self.camera.local.position = tuple(pos)
        
        return moved
    
    def look_at(self, target):
        """Orient the camera to look at a target position."""
        pos = np.array(self.camera.local.position)
        target = np.array(target)
        
        direction = target - pos
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        
        # Calculate yaw and pitch from direction
        self._yaw = np.arctan2(-direction[0], -direction[2])
        self._pitch = np.arcsin(np.clip(direction[1], -1, 1))
        self._roll = 0.0  # Reset roll when looking at target
        
        # Set targets to current (no smoothing needed for look_at)
        self._target_yaw = self._yaw
        self._target_pitch = self._pitch
        self._target_roll = self._roll
        
        self._update_camera_rotation()


class TrajectoryViewer3D:
    """
    A 3D trajectory viewer with first-person controls using pygfx.
    
    The viewer is constructed in steps:
    1. Create viewer with basic settings (canvas, background)
    2. Add scatter layers with add_scatter() - can add multiple
    3. Add trajectory for animation with add_trajectory()
    4. Call show() to display
    
    Parameters
    ----------
    background : str or tuple
        Background color.
    title : str
        Window title.
    show_time_overlay : bool
        Whether to show time/frame info overlay.
    size : tuple
        Canvas size in pixels (width, height).
    fov : float
        Field of view for perspective camera (degrees).
    """
    
    def __init__(
        self,
        background: str = "#0a0a0a",
        title: str = "3D Trajectory Viewer",
        show_time_overlay: bool = True,
        size: tuple = (1024, 768),
        fov: float = 60,
    ):
        self._show_time_overlay = show_time_overlay
        self._scatters = []
        self._trajectory_data = None
        self._trajectory_setup = False
        
        # Create canvas and renderer
        self.canvas = RenderCanvas(title=title, size=size)
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        
        # Create scene and 3D perspective camera
        self.scene = gfx.Scene()
        self.camera = gfx.PerspectiveCamera(fov=fov)
        
        # Set background
        self.scene.add(gfx.Background.from_color(background))
        
        # Add ambient light for better visibility
        self.scene.add(gfx.AmbientLight(intensity=1.0))
        
        # Build overlay scene for time display
        self._build_overlay()
        
        # Setup first-person controller
        self._fps_controller = FirstPersonController(
            self.camera, self.canvas, move_speed=0.5
        )
        
        # Animation state
        self._playing = False
        self._play_speed = 1.0
        self._frame_position = 0.0
        self._frame_index = 0
        self._trail_visible = True
        self._trail_start_frame = 0
        
        # Camera follow mode
        self._follow_mode = False
        self._follow_distance = 5.0  # Distance behind the trajectory point
        self._follow_height = 2.0    # Height above the trajectory point
        self._follow_smoothing = 0.1  # Camera smoothing factor (0-1, lower = smoother)
        
        # Connect events
        self.canvas.add_event_handler(self._on_key, "key_down")
        self.canvas.request_draw(self._animate)
    
    def add_scatter(
        self,
        data: nap.TsdFrame,
        colors: np.ndarray = None,
        cmap: str = None,
        point_size: float = 5.0,
        opacity: float = 0.8,
        name: str = None,
    ) -> gfx.Points:
        """
        Add a 3D scatter layer to the scene.
        
        Parameters
        ----------
        data : nap.TsdFrame
            TsdFrame with 'x', 'y', 'z' columns or 3 columns for 3D coordinates.
        colors : np.ndarray, optional
            RGBA colors for each point (N, 4). If None, uses cmap or gray.
        cmap : str, optional
            Matplotlib colormap name. If provided, colors points by time.
        point_size : float
            Size of points.
        opacity : float
            Opacity of points (0-1).
        name : str, optional
            Name for this scatter object.
        
        Returns
        -------
        gfx.Points
            The created Points object.
        """
        x, y, z = _extract_xyz(data)
        n_points = len(x)
        
        # Setup colors
        if colors is None:
            if cmap is not None:
                cmap_func = plt.get_cmap(cmap)
                t_norm = np.linspace(0, 1, n_points)
                point_colors = cmap_func(t_norm)[:, :4].astype("float32")
            else:
                point_colors = np.ones((n_points, 4), dtype="float32")
                point_colors[:, :3] = 0.8
        else:
            point_colors = _ensure_rgba(colors)
        
        # Apply opacity
        point_colors[:, 3] *= opacity
        
        # Create positions
        positions = np.column_stack([x, y, z]).astype("float32")
        
        # Create points object
        points = gfx.Points(
            gfx.Geometry(positions=positions, colors=point_colors),
            gfx.PointsMaterial(
                size=point_size,
                color_mode="vertex",
            ),
        )
        self.scene.add(points)
        
        # Store reference
        scatter_info = {
            'points': points,
            'data': data,
            'x': x,
            'y': y,
            'z': z,
            'colors': point_colors,
            'name': name or f"scatter_{len(self._scatters)}",
        }
        self._scatters.append(scatter_info)
        
        # Update camera view to fit data
        self._update_view_bounds()
        
        return points
    
    def add_trajectory(
        self,
        data: nap.TsdFrame,
        trail_length: int = 100,
        marker_color: str = "red",
        marker_size: float = None,
        trail_thickness: float = 3.0,
    ):
        """
        Add the trajectory animation (time marker and trail).
        
        Parameters
        ----------
        data : nap.TsdFrame
            TsdFrame with 'x', 'y', 'z' columns. The animation follows this data.
        trail_length : int
            Number of frames to show in the trail.
        marker_color : str
            Color of the time marker.
        marker_size : float, optional
            Size of marker. If None, uses 3x the first scatter's size.
        trail_thickness : float
            Thickness of the trail line.
        """
        self._traj_x, self._traj_y, self._traj_z = _extract_xyz(data)
        
        self._trajectory_data = data
        self._times = data.times()
        self._trail_length = trail_length
        self._marker_color = gfx.Color(marker_color) # Store for trail
        
        # Determine marker size
        if marker_size is None:
            if self._scatters:
                marker_size = self._scatters[0]['points'].material.size * 3
            else:
                marker_size = 15.0
        
        # Create time marker
        marker_pos = np.array([[self._traj_x[0], self._traj_y[0], self._traj_z[0]]], dtype="float32")
        self.time_marker = gfx.Points(
            gfx.Geometry(positions=marker_pos),
            gfx.PointsMaterial(size=marker_size, color=marker_color),
        )
        self.scene.add(self.time_marker)
        
        # Create trail
        trail_pos = np.zeros((trail_length, 3), dtype="float32")
        trail_colors = np.zeros((trail_length, 4), dtype="float32")
        self.trail = gfx.Line(
            gfx.Geometry(positions=trail_pos, colors=trail_colors),
            gfx.LineMaterial(thickness=trail_thickness, color_mode="vertex"),
        )
        self.scene.add(self.trail)
        
        self._trajectory_setup = True
        self._update_marker(0)
    
    def _update_view_bounds(self):
        """Position camera to view all point clouds."""
        if not self._scatters:
            return
        
        # Collect all coordinates
        all_x = np.concatenate([pc['x'] for pc in self._scatters])
        all_y = np.concatenate([pc['y'] for pc in self._scatters])
        all_z = np.concatenate([pc['z'] for pc in self._scatters])
        
        # Calculate center and extent
        center = np.array([all_x.mean(), all_y.mean(), all_z.mean()])
        extent = max(np.ptp(all_x), np.ptp(all_y), np.ptp(all_z))
        
        # Position camera looking at center from a distance
        distance = extent * 1.5
        self.camera.local.position = (center[0], center[1], center[2] + distance)
        
        # Update FPS controller to look at center
        self._fps_controller.look_at(center)
        self._fps_controller.move_speed = extent * 0.02
    
    def _build_overlay(self):
        """Build the overlay scene for time display."""
        self._overlay_scene = gfx.Scene()
        self._overlay_camera = gfx.ScreenCoordsCamera(invert_y=True)
        
        # Time display text
        self._time_text = gfx.Text(
            text="t = 0.000 s",
            font_size=16,
            anchor="top-left",
            screen_space=True,
            material=gfx.TextMaterial(
                color="#62b686",
                outline_color="#000000",
                outline_thickness=0.15,
                aa=True,
            ),
        )
        self._time_text.local.position = (10, 10, 0)
        self._overlay_scene.add(self._time_text)
        
        # Frame info text
        self._frame_text = gfx.Text(
            text="frame 0 / 0",
            font_size=12,
            anchor="top-left",
            screen_space=True,
            material=gfx.TextMaterial(
                color="#aaaaaa",
                outline_color="#000000",
                outline_thickness=0.1,
                aa=True,
            ),
        )
        self._frame_text.local.position = (10, 30, 0)
        self._overlay_scene.add(self._frame_text)
        
        # Controls hint
        self._controls_text = gfx.Text(
            text="WASD: move | Z/X: up/down | Q/E: turn | Drag: look | Scroll: speed",
            font_size=11,
            anchor="bottom-left",
            screen_space=True,
            material=gfx.TextMaterial(
                color="#666666",
                outline_color="#000000",
                outline_thickness=0.1,
                aa=True,
            ),
        )
        self._controls_text.local.position = (10, -10, 0)
        self._overlay_scene.add(self._controls_text)
    
    def _get_interpolated_position(self, frame_position: float):
        """Get linearly interpolated x, y, z position at fractional frame index."""
        if not self._trajectory_setup:
            return 0.0, 0.0, 0.0
        
        n_frames = len(self._traj_x)
        frame_position = np.clip(frame_position, 0, n_frames - 1)
        
        idx = int(frame_position)
        frac = frame_position - idx
        
        if idx >= n_frames - 1:
            return float(self._traj_x[-1]), float(self._traj_y[-1]), float(self._traj_z[-1])
        
        x = self._traj_x[idx] + frac * (self._traj_x[idx + 1] - self._traj_x[idx])
        y = self._traj_y[idx] + frac * (self._traj_y[idx + 1] - self._traj_y[idx])
        z = self._traj_z[idx] + frac * (self._traj_z[idx + 1] - self._traj_z[idx])
        
        return float(x), float(y), float(z)
    
    def _update_marker(self, frame_position: float):
        """Update the time marker position with interpolation support."""
        if not self._trajectory_setup:
            return
        
        n_frames = len(self._traj_x)
        self._frame_position = frame_position % n_frames
        self._frame_index = int(self._frame_position)
        
        # Get interpolated position
        interp_x, interp_y, interp_z = self._get_interpolated_position(self._frame_position)
        
        # Update marker position
        pos = self.time_marker.geometry.positions.data
        pos[0, 0] = interp_x
        pos[0, 1] = interp_y
        pos[0, 2] = interp_z
        self.time_marker.geometry.positions.update_full()
        
        # Update trail
        trail_pos = self.trail.geometry.positions.data
        trail_colors = self.trail.geometry.colors.data
        
        if self._trail_visible and self._trail_length > 0:
            effective_start = max(self._trail_start_frame, self._frame_index - self._trail_length + 2)
            start_idx = max(0, effective_start)
            end_idx = min(self._frame_index + 1, len(self._traj_x))
            trail_len = end_idx - start_idx
            
            if trail_len > 0:
                trail_pos[:trail_len, 0] = self._traj_x[start_idx:end_idx]
                trail_pos[:trail_len, 1] = self._traj_y[start_idx:end_idx]
                trail_pos[:trail_len, 2] = self._traj_z[start_idx:end_idx]
                
                # Add interpolated position
                trail_pos[trail_len, 0] = interp_x
                trail_pos[trail_len, 1] = interp_y
                trail_pos[trail_len, 2] = interp_z
                trail_len += 1
                
                # Fade trail
                alpha = np.linspace(0.1, 1.0, trail_len)
                
                # Use marker color for trail
                r, g, b, a = self._marker_color
                trail_colors[:trail_len, 0] = r
                trail_colors[:trail_len, 1] = g
                trail_colors[:trail_len, 2] = b
                trail_colors[:trail_len, 3] = alpha * a
            
            trail_pos[trail_len:] = np.nan
        else:
            trail_pos[:] = np.nan
        
        self.trail.geometry.positions.update_full()
        self.trail.geometry.colors.update_full()
        
        # Update overlay text
        if self._show_time_overlay:
            idx = self._frame_index
            if idx < len(self._times) - 1:
                frac = self._frame_position - idx
                t0 = self._times[idx]
                t1 = self._times[idx + 1]
                current_time = t0 + frac * (t1 - t0)
            else:
                current_time = self._times[-1]
            self._time_text.set_text(f"t = {current_time:.4f} s")
            self._frame_text.set_text(f"frame {self._frame_index} / {len(self._traj_x) - 1}")
    
    def go_to_time(self, target_time: float):
        """Jump to a specific time."""
        if not self._trajectory_setup:
            return
        
        idx = np.searchsorted(self._times, target_time, side="right") - 1
        idx = np.clip(idx, 0, len(self._times) - 1)
        
        if idx < len(self._times) - 1:
            t0 = self._times[idx]
            t1 = self._times[idx + 1]
            if t1 > t0:
                frac = (target_time - t0) / (t1 - t0)
                frac = np.clip(frac, 0, 1)
                frame_position = idx + frac
            else:
                frame_position = float(idx)
        else:
            frame_position = float(idx)
        
        self._update_marker(frame_position)
        self.canvas.request_draw()
    
    def _reset_trail(self):
        """Reset trail to start from current position."""
        if not self._trajectory_setup:
            return
        self._trail_start_frame = self._frame_index
        trail_pos = self.trail.geometry.positions.data
        trail_pos[:] = np.nan
        self.trail.geometry.positions.update_full()
    
    def _toggle_trail(self):
        """Toggle trail visibility."""
        if not self._trajectory_setup:
            return
        self._trail_visible = not self._trail_visible
        if not self._trail_visible:
            trail_pos = self.trail.geometry.positions.data
            trail_pos[:] = np.nan
            self.trail.geometry.positions.update_full()
    
    def fly_to_marker(self):
        """Move camera to current marker position."""
        if not self._trajectory_setup:
            return
        
        x, y, z = self._get_interpolated_position(self._frame_position)
        
        # Position camera slightly behind and above the marker
        forward = self._fps_controller.get_forward_vector()
        offset = 5.0 * self._fps_controller.move_speed * 10
        
        new_pos = np.array([x, y, z]) - forward * offset
        self.camera.local.position = tuple(new_pos)
        self._fps_controller.look_at([x, y, z])
    
    def toggle_follow_mode(self):
        """Toggle camera follow mode."""
        self._follow_mode = not self._follow_mode
        if self._follow_mode:
            # Initialize follow distance based on current camera settings
            self._follow_distance = self._fps_controller.move_speed * 50
            self._follow_height = self._fps_controller.move_speed * 20
    
    def _get_trajectory_direction(self, frame_position: float):
        """Get the direction of movement along the trajectory."""
        if not self._trajectory_setup:
            return np.array([0, 0, -1])
        
        n_frames = len(self._traj_x)
        idx = int(frame_position)
        
        # Get direction from current to next point (or previous if at end)
        if idx < n_frames - 1:
            curr = np.array([self._traj_x[idx], self._traj_y[idx], self._traj_z[idx]])
            next_pt = np.array([self._traj_x[idx + 1], self._traj_y[idx + 1], self._traj_z[idx + 1]])
            direction = next_pt - curr
        elif idx > 0:
            prev = np.array([self._traj_x[idx - 1], self._traj_y[idx - 1], self._traj_z[idx - 1]])
            curr = np.array([self._traj_x[idx], self._traj_y[idx], self._traj_z[idx]])
            direction = curr - prev
        else:
            direction = np.array([0, 0, -1])
        
        # Normalize
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm
        else:
            direction = np.array([0, 0, -1])
        
        return direction
    
    def _update_follow_camera(self):
        """Update camera position to follow the trajectory."""
        if not self._trajectory_setup or not self._follow_mode:
            return
        
        # Get current marker position
        marker_x, marker_y, marker_z = self._get_interpolated_position(self._frame_position)
        marker_pos = np.array([marker_x, marker_y, marker_z])
        
        # Get trajectory direction
        direction = self._get_trajectory_direction(self._frame_position)
        
        # Calculate target camera position: behind and above the marker
        # "Behind" means opposite to the direction of travel
        target_pos = marker_pos - direction * self._follow_distance
        target_pos[1] += self._follow_height  # Add height offset
        
        # Smoothly interpolate camera position
        current_pos = np.array(self.camera.local.position)
        new_pos = current_pos + (target_pos - current_pos) * self._follow_smoothing
        
        self.camera.local.position = tuple(new_pos)
        
        # Make camera look at the marker
        self._fps_controller.look_at(marker_pos)
    
    def _on_key(self, event):
        """Handle keyboard input (non-movement keys)."""
        key = event["key"]
        
        if key == " ":
            self._playing = not self._playing
            print(f"{'Playing' if self._playing else 'Paused'} at frame {self._frame_index}")
        
        elif key == "ArrowRight":
            self._update_marker(self._frame_position + 10)
        
        elif key == "ArrowLeft":
            self._update_marker(self._frame_position - 10)
        
        elif key == "ArrowUp":
            self._play_speed = min(self._play_speed * 2, 64)
            print(f"Speed: {self._play_speed}x")
        
        elif key == "ArrowDown":
            self._play_speed = max(self._play_speed / 2, 0.0625)
            print(f"Speed: {self._play_speed}x")
        
        elif key == "r":
            self._update_view_bounds()
            self._frame_position = 0.0
            self._frame_index = 0
            self._update_marker(0)
        
        elif key == "c":
            self._reset_trail()
            print(f"Trail reset at frame {self._frame_index}")
        
        elif key == "t":
            self._toggle_trail()
            print(f"Trail {'visible' if self._trail_visible else 'hidden'}")
        
        elif key == "f":
            # Fly to current marker position
            self.fly_to_marker()
            print("Flying to marker")
        
        elif key == "g":
            # Toggle follow mode
            self.toggle_follow_mode()
            print(f"Follow mode {'ON' if self._follow_mode else 'OFF'}")
        
        elif key == "Home":
            self._update_marker(0)
        
        elif key == "End":
            if self._trajectory_setup:
                self._update_marker(len(self._traj_x) - 1)
        
        self.canvas.request_draw()
    
    def _animate(self):
        """Animation loop."""
        # Update FPS controller (WASD movement) - only if not in follow mode
        if not self._follow_mode:
            self._fps_controller.update()
        
        # Update playback
        if self._playing and self._trajectory_setup:
            self._update_marker(self._frame_position + self._play_speed)
        
        # Update follow camera if in follow mode
        if self._follow_mode:
            self._update_follow_camera()
        
        # Render main scene
        self.renderer.render(self.scene, self.camera, flush=False)
        
        # Render overlay
        if self._show_time_overlay and self._trajectory_setup:
            self.renderer.render(self._overlay_scene, self._overlay_camera, flush=False)
        
        self.renderer.flush()
        self.canvas.request_draw(self._animate)
    
    def get_scatter(self, name_or_index=0):
        """Get a scatter object by name or index."""
        if isinstance(name_or_index, int):
            return self._scatters[name_or_index]
        for pc in self._scatters:
            if pc['name'] == name_or_index:
                return pc
        raise KeyError(f"Scatter '{name_or_index}' not found")
    
    def show(self):
        """Display the viewer."""
        print("=" * 60)
        print("3D Trajectory Viewer Controls:")
        print("=" * 60)
        print("  WASD: Move forward/left/backward/right")
        print("  Z/X: Move up/down")
        print("  Q/E: Turn left/right")
        print("  Mouse drag: Look around")
        print("  Scroll wheel: Adjust movement speed")
        print("-" * 60)
        print("  Space: Play/Pause animation")
        print("  Arrow Left/Right: Step backward/forward")
        print("  Arrow Up/Down: Change playback speed")
        print("  F: Fly to current marker position")
        print("  G: Toggle camera follow mode")
        print("  R: Reset view and position")
        print("  C: Reset trail")
        print("  T: Toggle trail visibility")
        print("  Home/End: Jump to start/end")
        print("=" * 60)
        loop.run()


def create_trajectory_viewer_3d(
    manifold: nap.TsdFrame,
    color_by: str = "time",
    cmap: str = "viridis",
    point_size: float = 4.0,
    trail_length: int = 100,
    show_time_overlay: bool = True,
) -> TrajectoryViewer3D:
    """
    Convenience function to create a 3D trajectory viewer with a single dataset.
    
    Parameters
    ----------
    manifold : nap.TsdFrame
        Manifold coordinates with 'x', 'y', 'z' columns (or 3 columns).
    color_by : str
        How to color points: "time" or None.
    cmap : str
        Matplotlib colormap name.
    point_size : float
        Size of scatter points.
    trail_length : int
        Number of frames in the trail.
    show_time_overlay : bool
        Whether to show time/frame overlay.
    
    Returns
    -------
    TrajectoryViewer3D
        The viewer instance.
    """

    viewer = TrajectoryViewer3D(show_time_overlay=show_time_overlay)

    use_cmap = cmap if color_by == "time" else None
    
    viewer.add_scatter(
        manifold,
        cmap=use_cmap,
        point_size=point_size,
        name="main",
    )
    
    viewer.add_trajectory(manifold, trail_length=trail_length)
    
    return viewer
