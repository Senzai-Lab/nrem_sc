import numpy as np
import pynapple as nap
import pygfx as gfx
from pygfx import Viewport
from rendercanvas.auto import RenderCanvas, loop


class ManifoldController:
    """
    A pynaviz-compatible controller for the ManifoldViewer.
    
    Implements the interface required by pynaviz's ControllerGroup:
    - sync(event): respond to time synchronization events
    - advance(delta): advance time by delta
    - controller_id: unique identifier for the controller
    - enabled: whether the controller is active
    """
    
    def __init__(self, viewer: "ManifoldViewer", renderer, controller_id: int = None):
        self.viewer = viewer
        self.renderer = renderer
        self._controller_id = controller_id
        self.enabled = True
        self._current_time = 0.0
        
        # For pynaviz compatibility
        self.renderer_handle_event = None
        if renderer:
            viewport = Viewport.from_viewport_or_renderer(renderer)
            self.renderer_handle_event = viewport.renderer.handle_event
    
    @property
    def controller_id(self):
        return self._controller_id
    
    @controller_id.setter
    def controller_id(self, value):
        if self._controller_id is not None:
            raise ValueError("Controller id can be set only once!")
        self._controller_id = value
    
    def sync(self, event):
        """
        Synchronize to a time from pynaviz's ControllerGroup.
        
        Parameters
        ----------
        event : SyncEvent
            Event containing current_time or cam_state with position.
        """
        if not self.enabled:
            return
            
        # Extract time from event
        if hasattr(event, 'kwargs'):
            if "cam_state" in event.kwargs:
                new_time = event.kwargs["cam_state"]["position"][0]
            elif "current_time" in event.kwargs:
                new_time = event.kwargs["current_time"]
            else:
                return
        else:
            return
        
        self._current_time = new_time
        self.viewer.go_to_time(new_time)
    
    def advance(self, delta: float = 0.025):
        """
        Advance the current time by delta.
        
        Parameters
        ----------
        delta : float
            Time increment in seconds.
        """
        self._current_time += delta
        self.viewer.go_to_time(self._current_time)
    
    def go_to(self, target_time: float):
        """Jump to a specific time."""
        self._current_time = target_time
        self.viewer.go_to_time(target_time)


class ManifoldViewer:
    """
    A modular 2D manifold viewer using pygfx.
    
    The viewer is constructed in steps:
    1. Create viewer with basic settings (canvas, background)
    2. Add point clouds with add_point_cloud() - can add multiple
    3. Setup trajectory for animation with setup_trajectory()
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
    
    Examples
    --------
    >>> viewer = ManifoldViewer(title="NREM vs Wake")
    >>> viewer.add_point_cloud(wake_data, cmap="coolwarm", opacity=0.2, z_offset=-0.1)
    >>> viewer.add_point_cloud(nrem_data, cmap="viridis")
    >>> viewer.setup_trajectory(nrem_data, trail_length=100)
    >>> viewer.show()
    """
    
    def __init__(
        self,
        background: str = "#0a0a0a",
        title: str = "Manifold Viewer",
        show_time_overlay: bool = True,
        size: tuple = (800, 800),
    ):
        self._show_time_overlay = show_time_overlay
        self._point_clouds = []  # List of added point clouds
        self._trajectory_data = None  # Data for trajectory animation
        self._trajectory_setup = False
        
        # Create canvas and renderer
        self.canvas = RenderCanvas(title=title, size=size)
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        
        # Create scene and camera
        self.scene = gfx.Scene()
        self.camera = gfx.OrthographicCamera()
        self.camera.maintain_aspect = True
        
        # Set background
        self.scene.add(gfx.Background.from_color(background))
        
        # Build overlay scene for time display
        self._build_overlay()
        
        # Setup interaction (basic pan/zoom for standalone use)
        self._pan_zoom = gfx.PanZoomController(self.camera, register_events=self.renderer)
        
        # Create pynaviz-compatible controller for synchronization
        self.controller = ManifoldController(self, self.renderer)
        
        # Animation state (initialized but not active until setup_trajectory)
        self._playing = False
        self._play_speed = 1.0
        self._frame_position = 0.0
        self._frame_index = 0
        self._trail_visible = True
        self._trail_start_frame = 0
        
        # Connect events
        self.canvas.add_event_handler(self._on_key, "key_down")
        self.canvas.request_draw(self._animate)
    
    def add_point_cloud(
        self,
        data: nap.TsdFrame,
        colors: np.ndarray = None,
        cmap: str = None,
        point_size: float = 5.0,
        opacity: float = 0.6,
        z_offset: float = 0.0,
        name: str = None,
    ) -> gfx.Points:
        """
        Add a point cloud to the scene.
        
        Parameters
        ----------
        data : nap.TsdFrame
            TsdFrame with 'x' and 'y' columns for 2D coordinates.
        colors : np.ndarray, optional
            RGBA colors for each point (N, 4). If None, uses cmap or gray.
        cmap : str, optional
            Matplotlib colormap name. If provided, colors points by time.
        point_size : float
            Size of points.
        opacity : float
            Opacity of points (0-1).
        z_offset : float
            Z position offset. Negative values render behind, positive in front.
        name : str, optional
            Name for this point cloud (for later reference).
        
        Returns
        -------
        gfx.Points
            The created Points object.
        """
        # Extract coordinates
        if 'x' in data.columns and 'y' in data.columns:
            x = data['x'].values.astype("float32")
            y = data['y'].values.astype("float32")
        else:
            x = data.values[:, 0].astype("float32")
            y = data.values[:, 1].astype("float32")
        
        n_points = len(x)
        
        # Setup colors
        if colors is None:
            if cmap is not None:
                import matplotlib.pyplot as plt
                cmap_func = plt.get_cmap(cmap)
                t_norm = np.linspace(0, 1, n_points)
                point_colors = cmap_func(t_norm)[:, :4].astype("float32")
            else:
                point_colors = np.ones((n_points, 4), dtype="float32")
                point_colors[:, :3] = 0.8  # Light gray
        else:
            point_colors = colors.astype("float32")
            if point_colors.shape[1] == 3:
                alpha = np.ones((n_points, 1), dtype="float32")
                point_colors = np.hstack([point_colors, alpha])
        
        # Create positions
        positions = np.zeros((n_points, 3), dtype="float32")
        positions[:, 0] = x
        positions[:, 1] = y
        positions[:, 2] = z_offset
        
        # Create points object
        points = gfx.Points(
            gfx.Geometry(positions=positions, colors=point_colors),
            gfx.PointsMaterial(
                size=point_size,
                color_mode="vertex",
                opacity=opacity,
            ),
        )
        self.scene.add(points)
        
        # Store reference
        cloud_info = {
            'points': points,
            'data': data,
            'x': x,
            'y': y,
            'colors': point_colors,
            'name': name or f"cloud_{len(self._point_clouds)}",
        }
        self._point_clouds.append(cloud_info)
        
        # Update camera view to fit all data
        self._update_view_bounds()
        
        return points
    
    def setup_trajectory(
        self,
        data: nap.TsdFrame,
        trail_length: int = 100,
        marker_color: str = "red",
        marker_size: float = None,
        trail_thickness: float = 2.0,
    ):
        """
        Setup the trajectory animation (time marker and trail).
        
        This defines which data the playback animation follows.
        
        Parameters
        ----------
        data : nap.TsdFrame
            TsdFrame with 'x' and 'y' columns. The animation will follow
            this data's timestamps and coordinates.
        trail_length : int
            Number of frames to show in the trail (default 100).
        marker_color : str
            Color of the time marker.
        marker_size : float, optional
            Size of marker. If None, uses 3x the first point cloud's size.
        trail_thickness : float
            Thickness of the trail line.
        """
        # Extract coordinates
        if 'x' in data.columns and 'y' in data.columns:
            self._traj_x = data['x'].values.astype("float32")
            self._traj_y = data['y'].values.astype("float32")
        else:
            self._traj_x = data.values[:, 0].astype("float32")
            self._traj_y = data.values[:, 1].astype("float32")
        
        self._trajectory_data = data
        self._times = data.times()
        self._trail_length = trail_length
        
        # Determine marker size
        if marker_size is None:
            if self._point_clouds:
                # Use 3x the first point cloud's size
                marker_size = self._point_clouds[0]['points'].material.size * 3
            else:
                marker_size = 15.0
        
        # Create time marker (larger, bright point)
        marker_pos = np.array([[self._traj_x[0], self._traj_y[0], 0.1]], dtype="float32")
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
        
        # Initialize marker position
        self._update_marker(0)
    
    def _update_view_bounds(self):
        """Update camera to fit all point clouds."""
        if not self._point_clouds:
            return
        
        # Collect all x, y coordinates
        all_x = np.concatenate([pc['x'] for pc in self._point_clouds])
        all_y = np.concatenate([pc['y'] for pc in self._point_clouds])
        
        margin = 0.1
        x_range = all_x.max() - all_x.min()
        y_range = all_y.max() - all_y.min()
        
        x_min = all_x.min() - margin * x_range
        x_max = all_x.max() + margin * x_range
        y_min = all_y.min() - margin * y_range
        y_max = all_y.max() + margin * y_range
        
        self.camera.show_rect(x_min, x_max, y_min, y_max)
    
    def _build_overlay(self):
        """Build the overlay scene for time display."""
        # Create overlay scene and camera for screen-space text
        # Use invert_y=True so Y=0 is at the top (like typical UI coordinates)
        self._overlay_scene = gfx.Scene()
        self._overlay_camera = gfx.ScreenCoordsCamera(invert_y=True)
        
        # Time display text (top-left corner)
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
        self._time_text.local.position = (10, 10, 0)  # 10px from top-left
        self._overlay_scene.add(self._time_text)
        
        # Frame info text (below time)
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

    def add_colorbar(
        self,
        cmap: str = "husl",
        label: str = "",
        vmin: float = 0,
        vmax: float = 360,
        n_colors: int = 256,
        width: int = 20,
        height: int = 200,
        position: str = "right",
        padding: int = 20,
    ):
        """
        Add a colorbar to the overlay.
        
        Parameters
        ----------
        cmap : str
            Matplotlib colormap name (e.g., 'husl', 'viridis', 'hsv').
        label : str
            Label for the colorbar.
        vmin, vmax : float
            Value range for the colorbar tick labels.
        n_colors : int
            Number of colors in the gradient.
        width : int
            Width of the colorbar in pixels.
        height : int
            Height of the colorbar in pixels.
        position : str
            Position: 'right', 'left', 'top', 'bottom'.
        padding : int
            Padding from the edge in pixels.
        """
        import matplotlib.pyplot as plt
        
        # Get colormap colors
        cmap_func = plt.get_cmap(cmap)
        colors = cmap_func(np.linspace(0, 1, n_colors))[:, :4].astype("float32")
        
        # Create a textured quad for the colorbar
        # Build color gradient as image data (RGBA)
        if position in ("right", "left"):
            # Vertical colorbar
            img_data = np.zeros((n_colors, 1, 4), dtype="float32")
            img_data[:, 0, :] = colors[::-1]  # Flip so high values at top
        else:
            # Horizontal colorbar
            img_data = np.zeros((1, n_colors, 4), dtype="float32")
            img_data[0, :, :] = colors
        
        # Create image texture
        texture = gfx.Texture(img_data, dim=2)
        
        # Create a plane mesh for the colorbar
        colorbar_mesh = gfx.Mesh(
            gfx.plane_geometry(width=width, height=height),
            gfx.MeshBasicMaterial(map=texture),
        )
        
        # Position the colorbar based on canvas size
        # Note: actual positioning happens in render loop or resize handler
        # For now, store the mesh and position preference
        colorbar_mesh.local.position = (0, 0, 0)  # Will be updated
        
        self._colorbar = {
            'mesh': colorbar_mesh,
            'width': width,
            'height': height,
            'position': position,
            'padding': padding,
            'vmin': vmin,
            'vmax': vmax,
            'label': label,
        }
        self._overlay_scene.add(colorbar_mesh)
        
        # Add tick labels (min and max)
        label_color = "#cccccc"
        
        # Determine text anchors based on position
        if position == "right":
            anchor = "middle-left"
        elif position == "left":
            anchor = "middle-right"
        else:  # top or bottom (horizontal)
            anchor = "top-center"
        
        # Max value label
        self._colorbar_max_text = gfx.Text(
            text=f"{vmax:.0f}",
            font_size=12,
            anchor=anchor,
            screen_space=True,
            material=gfx.TextMaterial(color=label_color, aa=True),
        )
        self._overlay_scene.add(self._colorbar_max_text)
        
        # Min value label
        self._colorbar_min_text = gfx.Text(
            text=f"{vmin:.0f}",
            font_size=12,
            anchor=anchor,
            screen_space=True,
            material=gfx.TextMaterial(color=label_color, aa=True),
        )
        self._overlay_scene.add(self._colorbar_min_text)
        
        # Label text
        if label:
            self._colorbar_label = gfx.Text(
                text=label,
                font_size=14,
                anchor="middle-center",
                screen_space=True,
                material=gfx.TextMaterial(color=label_color, aa=True),
            )
            self._overlay_scene.add(self._colorbar_label)
        
        # Update positions based on current canvas size
        self._update_colorbar_position()
        
        # Add resize handler to update colorbar position
        self.canvas.add_event_handler(self._on_resize, "resize")
    
    def _update_colorbar_position(self):
        """Update colorbar position based on canvas size."""
        if not hasattr(self, '_colorbar'):
            return
        
        w, h = self.canvas.get_logical_size()
        cb = self._colorbar
        padding = cb['padding']
        width = cb['width']
        height = cb['height']
        pos = cb['position']
        
        # Note: overlay uses invert_y=True, so Y=0 is at TOP of screen
        if pos == "right":
            x = w - padding - width / 2
            y = h / 2
        elif pos == "left":
            x = padding + width / 2
            y = h / 2
        elif pos == "bottom":
            x = padding + width / 2  # left-aligned
            y = h - padding - height / 2  # bottom of screen (high Y)
        elif pos == "top":
            x = padding + width / 2  # left-aligned
            y = padding + height / 2  # top of screen (low Y)
        else:
            x = w / 2
            y = h / 2
        
        cb['mesh'].local.position = (x, y, 0)
        
        # Update tick label positions based on orientation
        if pos in ("right", "left"):
            # Vertical colorbar: max at top, min at bottom
            if pos == "right":
                text_x = x + width / 2 + 5
            else:
                text_x = x - width / 2 - 5
            self._colorbar_max_text.local.position = (text_x, y - height / 2 + 5, 0)
            self._colorbar_min_text.local.position = (text_x, y + height / 2 - 5, 0)
            if hasattr(self, '_colorbar_label') and self._colorbar_label:
                self._colorbar_label.local.position = (x, y - height / 2 - 15, 0)
        else:
            # Horizontal colorbar: min at left, max at right
            text_y = y + height / 2 + 15
            self._colorbar_min_text.local.position = (x - width / 2, text_y, 0)
            self._colorbar_max_text.local.position = (x + width / 2, text_y, 0)
            if hasattr(self, '_colorbar_label') and self._colorbar_label:
                self._colorbar_label.local.position = (x, y - height / 2 - 10, 0)
    
    def _on_resize(self, event):
        """Handle canvas resize."""
        self._update_colorbar_position()

    def _get_interpolated_position(self, frame_position: float):
        """
        Get linearly interpolated x, y position at fractional frame index.
        
        Parameters
        ----------
        frame_position : float
            Fractional frame index (can be between integer frames).
        
        Returns
        -------
        tuple
            (x, y) interpolated coordinates.
        """
        if not self._trajectory_setup:
            return 0.0, 0.0
        
        n_frames = len(self._traj_x)
        # Clamp to valid range
        frame_position = np.clip(frame_position, 0, n_frames - 1)
        
        # Get integer frame and fractional part
        idx = int(frame_position)
        frac = frame_position - idx
        
        # Handle edge case at last frame
        if idx >= n_frames - 1:
            return float(self._traj_x[-1]), float(self._traj_y[-1])
        
        # Linear interpolation between adjacent frames
        x = self._traj_x[idx] + frac * (self._traj_x[idx + 1] - self._traj_x[idx])
        y = self._traj_y[idx] + frac * (self._traj_y[idx + 1] - self._traj_y[idx])
        
        return float(x), float(y)
    
    def _update_marker(self, frame_position: float):
        """Update the time marker position with interpolation support."""
        if not self._trajectory_setup:
            return
        
        # Handle wrapping
        n_frames = len(self._traj_x)
        self._frame_position = frame_position % n_frames
        self._frame_index = int(self._frame_position)
        
        # Get interpolated position for smooth movement
        interp_x, interp_y = self._get_interpolated_position(self._frame_position)
        
        # Update marker position
        pos = self.time_marker.geometry.positions.data
        pos[0, 0] = interp_x
        pos[0, 1] = interp_y
        self.time_marker.geometry.positions.update_full()
        
        # Update trail only if visible
        trail_pos = self.trail.geometry.positions.data
        trail_colors = self.trail.geometry.colors.data
        
        if self._trail_visible and self._trail_length > 0:
            # Trail includes frames up to current frame PLUS the interpolated marker position
            # This ensures the trail always connects to the marker
            # Respect _trail_start_frame to allow trail reset
            effective_start = max(self._trail_start_frame, self._frame_index - self._trail_length + 2)
            start_idx = max(0, effective_start)
            end_idx = min(self._frame_index + 1, len(self._traj_x))  # Include current frame
            trail_len = end_idx - start_idx
            
            if trail_len > 0:
                trail_pos[:trail_len, 0] = self._traj_x[start_idx:end_idx]
                trail_pos[:trail_len, 1] = self._traj_y[start_idx:end_idx]
                trail_pos[:trail_len, 2] = 0.05
                
                # Add the interpolated position as final point to connect to marker
                trail_pos[trail_len, 0] = interp_x
                trail_pos[trail_len, 1] = interp_y
                trail_pos[trail_len, 2] = 0.05
                trail_len += 1
                
                # Fade trail from transparent to opaque
                alpha = np.linspace(0.1, 1.0, trail_len)
                trail_colors[:trail_len, 0] = 1.0  # Red
                trail_colors[:trail_len, 1] = 0.3
                trail_colors[:trail_len, 2] = 0.3
                trail_colors[:trail_len, 3] = alpha
            
            # Hide unused trail points
            trail_pos[trail_len:] = np.nan
        else:
            # Trail hidden or disabled - set all to nan
            trail_pos[:] = np.nan
        
        self.trail.geometry.positions.update_full()
        self.trail.geometry.colors.update_full()
        
        # Update time overlay text with interpolated time
        if self._show_time_overlay:
            # Interpolate time to match the interpolated marker position
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
        """
        Jump to a specific time (for pynaviz synchronization).
        
        Parameters
        ----------
        target_time : float
            Target time in seconds.
        """
        if not self._trajectory_setup:
            return
        
        # Find the frame index for this time
        idx = np.searchsorted(self._times, target_time, side="right") - 1
        idx = np.clip(idx, 0, len(self._times) - 1)
        
        # For sub-frame interpolation, calculate fractional position
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
        
        # Set the trail start frame to current frame
        # This makes the trail "forget" its history and start fresh
        self._trail_start_frame = self._frame_index
        
        # Clear current trail display
        trail_pos = self.trail.geometry.positions.data
        trail_pos[:] = np.nan
        self.trail.geometry.positions.update_full()
    
    def _toggle_trail(self):
        """Toggle trail visibility."""
        if not self._trajectory_setup:
            return
        
        self._trail_visible = not self._trail_visible
        if not self._trail_visible:
            # Immediately hide trail
            trail_pos = self.trail.geometry.positions.data
            trail_pos[:] = np.nan
            self.trail.geometry.positions.update_full()
    
    def _on_key(self, event):
        """Handle keyboard input."""
        key = event["key"]
        
        if key == " ":
            # Toggle play/pause
            self._playing = not self._playing
            print(f"{'Playing' if self._playing else 'Paused'} at frame {self._frame_index} (speed: {self._play_speed}x)")
        
        elif key == "ArrowRight":
            self._update_marker(self._frame_position + 10)
        
        elif key == "ArrowLeft":
            self._update_marker(self._frame_position - 10)
        
        elif key == "ArrowUp":
            self._play_speed = min(self._play_speed * 2, 64)
            print(f"Speed: {self._play_speed}x")
        
        elif key == "ArrowDown":
            self._play_speed = max(self._play_speed / 2, 0.0625)  # Allow slower speeds
            print(f"Speed: {self._play_speed}x")
        
        elif key == "r":
            # Reset view
            self._update_view_bounds()
            self._frame_position = 0.0
            self._frame_index = 0
            self._update_marker(0)
        
        elif key == "c":
            # Reset trail (start fresh from current position)
            self._reset_trail()
            print(f"Trail reset at frame {self._frame_index}")
        
        elif key == "t":
            # Toggle trail visibility
            self._toggle_trail()
            print(f"Trail {'visible' if self._trail_visible else 'hidden'}")
        
        elif key == "Home":
            self._update_marker(0)
        
        elif key == "End":
            if self._trajectory_setup:
                self._update_marker(len(self._traj_x) - 1)
        
        self.canvas.request_draw()
    
    def _animate(self):
        """Animation loop."""
        if self._playing and self._trajectory_setup:
            # Use fractional frame position for smooth slow-motion
            self._update_marker(self._frame_position + self._play_speed)
        
        # Render main scene
        self.renderer.render(self.scene, self.camera, flush=False)
        
        # Render overlay (time display) on top
        if self._show_time_overlay and self._trajectory_setup:
            self.renderer.render(self._overlay_scene, self._overlay_camera, flush=False)
        
        # Flush to display
        self.renderer.flush()
        self.canvas.request_draw(self._animate)
    
    def get_point_cloud(self, name_or_index=0):
        """
        Get a point cloud by name or index.
        
        Parameters
        ----------
        name_or_index : str or int
            Name of the point cloud or its index in the list.
        
        Returns
        -------
        dict
            Point cloud info dict with 'points', 'data', 'x', 'y', 'colors', 'name'.
        """
        if isinstance(name_or_index, int):
            return self._point_clouds[name_or_index]
        else:
            for pc in self._point_clouds:
                if pc['name'] == name_or_index:
                    return pc
            raise KeyError(f"Point cloud '{name_or_index}' not found")
    
    def set_point_cloud_colors(self, name_or_index=0, colors=None, cmap=None, values=None, vmin=None, vmax=None):
        """
        Update colors of an existing point cloud.
        
        Parameters
        ----------
        name_or_index : str or int
            Which point cloud to update.
        colors : np.ndarray, optional
            Direct RGBA colors (N, 4).
        cmap : str, optional
            Colormap name. Used with values or to color by time.
        values : np.ndarray, optional
            Values to map to colors (e.g., head direction).
        vmin, vmax : float, optional
            Value range for colormap normalization.
        """
        import matplotlib.pyplot as plt
        
        pc = self.get_point_cloud(name_or_index)
        n_points = len(pc['x'])
        
        if colors is not None:
            new_colors = colors.astype("float32")
            if new_colors.shape[1] == 3:
                alpha = np.ones((n_points, 1), dtype="float32")
                new_colors = np.hstack([new_colors, alpha])
        elif cmap is not None:
            cmap_func = plt.get_cmap(cmap)
            if values is not None:
                if vmin is None:
                    vmin = np.nanmin(values)
                if vmax is None:
                    vmax = np.nanmax(values)
                v_norm = (values - vmin) / (vmax - vmin + 1e-10)
                v_norm = np.clip(v_norm, 0, 1)
            else:
                # Color by time (index)
                v_norm = np.linspace(0, 1, n_points)
            new_colors = cmap_func(v_norm)[:, :4].astype("float32")
        else:
            return  # Nothing to do
        
        pc['colors'] = new_colors
        pc['points'].geometry.colors.data[:] = new_colors
        pc['points'].geometry.colors.update_full()
    
    def show(self):
        """Display the viewer."""
        print("Controls:")
        print("  Space: Play/Pause")
        print("  Arrow Left/Right: Step backward/forward")
        print("  Arrow Up/Down: Change playback speed (supports slow-motion)")
        print("  R: Reset view and position")
        print("  C: Reset trail (forget history, start fresh)")
        print("  T: Toggle trail visibility")
        print("  Home/End: Jump to start/end")
        print("  Mouse: Pan and zoom")
        loop.run()


def create_manifold_viewer(
    manifold: nap.TsdFrame,
    color_by: str = "time",  # "time" or None
    cmap: str = "viridis",
    point_size: float = 4.0,
    trail_length: int = 100,
    show_time_overlay: bool = True,
) -> ManifoldViewer:
    """
    Convenience function to create a manifold viewer with a single dataset.
    
    For more complex setups (multiple point clouds, background data), use
    ManifoldViewer directly with add_point_cloud() and setup_trajectory().
    
    Parameters
    ----------
    manifold : nap.TsdFrame
        Manifold coordinates with 'x' and 'y' columns.
    color_by : str
        How to color points: "time" or None.
    cmap : str
        Matplotlib colormap name for coloring points (default "viridis").
        Used when color_by="time".
    point_size : float
        Size of scatter points.
    trail_length : int
        Number of frames to show in the trail (default 100).
        Set to 0 to disable trail. Smaller = faster fade.
    show_time_overlay : bool
        Whether to show the time/frame overlay text (default True).
    
    Returns
    -------
    ManifoldViewer
        The viewer instance.
    
    Examples
    --------
    >>> # Simple usage
    >>> viewer = create_manifold_viewer(nrem_data, color_by="time")
    >>> viewer.show()
    
    >>> # With background data (modular API)
    >>> viewer = ManifoldViewer(title="NREM vs Wake")
    >>> viewer.add_point_cloud(wake_data, cmap="coolwarm", opacity=0.2, z_offset=-0.1)
    >>> viewer.add_point_cloud(nrem_data, cmap="viridis", name="nrem")
    >>> viewer.setup_trajectory(nrem_data, trail_length=100)
    >>> viewer.show()
    """
    # Create viewer with basic settings
    viewer = ManifoldViewer(
        show_time_overlay=show_time_overlay,
    )
    
    # Determine colormap
    use_cmap = cmap if color_by == "time" else None
    
    # Add the point cloud
    viewer.add_point_cloud(
        manifold,
        cmap=use_cmap,
        point_size=point_size,
        name="main",
    )
    
    # Setup trajectory for animation
    viewer.setup_trajectory(manifold, trail_length=trail_length)
    
    return viewer


# === MAIN ===
if __name__ == "__main__":
    from src.constants import INTERIM_DATA_PATH, PROCESSED_DATA_PATH
    import matplotlib.colors as mcolors

    unit_id = "116b"
    data_dir = INTERIM_DATA_PATH / unit_id / "tmp"
    
    # Load data
    manifold_shifted = nap.load_file(data_dir / "manifold_shifted.npz")
    manifold_openfield = nap.load_file(PROCESSED_DATA_PATH / unit_id / "manifold_openfield2.npz")
    hd_angle_openfield = nap.load_file(PROCESSED_DATA_PATH / unit_id / "angle_openfield2.npz").to_numpy()
    
    # HD angle to RGBA colors
    # Normalize
    hsv_colors = np.ones((len(hd_angle_openfield), 3))
    hsv_colors[:, 0] = hd_angle_openfield / 360.0

    # HSV to RGBA
    rgb_colors = mcolors.hsv_to_rgb(hsv_colors)
    colors_rgba = np.column_stack((rgb_colors, np.ones(len(rgb_colors))))
    
    print(f"Loaded manifold: {len(manifold_openfield)} points")
    print(f"Loaded manifold: {len(manifold_shifted)} points")

    viewer = ManifoldViewer()

    # Add the point cloud
    viewer.add_point_cloud(
        data=manifold_openfield,
        colors=colors_rgba,
        point_size=2,
        name="wake")
    
    viewer.add_point_cloud(
        data=manifold_shifted,
        point_size=1,
        opacity=0.1,
        name="nrem")
    
    # Setup trajectory for animation
    viewer.setup_trajectory(manifold_shifted, trail_length=25)
    viewer.show()
    
    # === MODULAR USAGE EXAMPLE ===
    # For multiple point clouds (e.g., NREM + wake):
    #
    # viewer = ManifoldViewer(title="NREM vs Wake")
    # viewer.add_point_cloud(wake_data, cmap="coolwarm", opacity=0.2, z_offset=-0.1, name="wake")
    # viewer.add_point_cloud(nrem_data, cmap="viridis", name="nrem")
    # viewer.setup_trajectory(nrem_data, trail_length=100)
    # viewer.show()
    
    # === PYNAVIZ INTEGRATION EXAMPLE ===
    # To synchronize with other pynaviz plots:
    #
    # import pynaviz as viz
    # from pynaviz import ControllerGroup
    # 
    # # Create pynaviz plots
    # spike_widget = viz.TsGroupWidget(hd_spikes_shifted)
    # 
    # # Create manifold viewer
    # manifold_viewer = create_manifold_viewer(manifold_shifted)
    # 
    # # Create a ControllerGroup to synchronize them
    # cg = ControllerGroup(
    #     plots=[spike_widget],
    #     interval=(0, manifold_shifted.times()[-1])
    # )
    # 
    # # Add the manifold viewer to the group
    # cg.add(manifold_viewer, controller_id=1)
    # 
    # # Show widgets
    # spike_widget.show()
    # manifold_viewer.show()
