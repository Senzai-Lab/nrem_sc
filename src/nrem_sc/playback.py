from __future__ import annotations

import time
from typing import Callable, List, Optional

import math
import numpy as np
import pygfx as gfx
from pygfx.utils.viewport import Viewport


class PlaybackController:
    """Timeseries playback controller for pygfx.

    Features
    --------
    * Play / pause with configurable speed (time-based, not frame-rate-dependent).
    * Speed up / down in 2x increments.
    * Frame stepping (forward / backward, small and large).
    * Continuous stepping when holding step keys.
    * Jump to start / end.
    * Jump to next / previous event marker (optional).
    * Loops or stops at boundaries.

    Default key/mouse bindings (avoid FlyController's WASD + space + shift + QE)::

        Right Click             Play / pause
        Arrow Up / Down         Speed up / down  (2x)
        Arrow Right / Left      Step forward / backward
        Ctrl + Scroll Up/Down   Speed up / down  (2x)
        Home / End              Jump to start / end
        n  /  b         next / previous event marker

    Parameters
    ----------
    times : array-like
        1-D array of timestamps in seconds (e.g. ``data.times()``).
    speed : float
        Initial playback speed.  1.0 = real-time.
    step_size : int
        Frames per small step (default 1).
    loop : bool
        Wrap around at boundaries.
    events : array-like, optional
        Sorted event timestamps for ``jump_next_event`` / ``jump_prev_event``.
    register_events : Renderer | Viewport, optional
        If provided, registers ``key_down`` and ``before_render`` handlers.
    verbose : bool
        Print state changes to stdout (useful for debugging).
    """

    _default_key_bindings: dict[str, str] = {
        "p": "toggle_play",
        "ArrowUp": "speed_up",
        "ArrowDown": "speed_down",
        "ArrowRight": "step_forward",
        "ArrowLeft": "step_backward",
        "Home": "jump_start",
        "End": "jump_end",
        "n": "jump_next_event",
        "b": "jump_prev_event",
    }

    def __init__(
        self,
        times,
        *,
        speed: float = 1.0,
        step_size: int = 1,
        loop: bool = True,
        events=None,
        register_events=None,
        verbose: bool = False,
    ):
        self._times = np.asarray(times, dtype=np.float64)
        self._n_frames = len(self._times)
        if self._n_frames < 2:
            raise ValueError("Need at least 2 timestamps")

        self._speed = float(speed)
        self._step_size = int(step_size)
        self._loop = bool(loop)
        self._events = np.sort(np.asarray(events, dtype=np.float64)) if events is not None else None
        self._verbose = verbose

        # Playback state
        self._playing = False
        self._frame_position: float = 0.0
        self._current_time: float = float(self._times[0])

        # Wall-clock tracking
        self._last_tick_time: float = time.perf_counter()
        self._in_tick: bool = False  # re-entrancy guard

        # Callbacks
        self._handlers: List[Callable] = []
        self._held_step_actions: set = set()

        # Key bindings (copy so per-instance customization works)
        self.key_bindings: dict[str, str] = dict(self._default_key_bindings)

        # Viewport reference for request_draw
        self._viewport: Optional[Viewport] = None

        # pynaviz ControllerGroup compatibility
        self._controller_id: Optional[int] = None
        self.enabled: bool = True
        self.renderer_handle_event: Optional[Callable] = None

        if register_events is not None:
            self.register_events(register_events)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def times(self) -> np.ndarray:
        """The timestamp array."""
        return self._times

    @property
    def n_frames(self) -> int:
        """Total number of frames."""
        return self._n_frames

    @property
    def playing(self) -> bool:
        """Whether playback is running."""
        return self._playing

    @playing.setter
    def playing(self, value: bool):
        self._playing = bool(value)
        if self._playing:
            # Reset wall-clock so first tick doesn't produce a huge dt
            self._last_tick_time = time.perf_counter()

    @property
    def speed(self) -> float:
        """Playback speed multiplier (1.0 = real-time)."""
        return self._speed

    @speed.setter
    def speed(self, value: float):
        self._speed = float(value)

    @property
    def frame_index(self) -> int:
        """Current integer frame index."""
        return int(self._frame_position)

    @property
    def frame_position(self) -> float:
        """Current fractional frame position (for interpolation)."""
        return self._frame_position

    @property
    def current_time(self) -> float:
        """Current data time in seconds (interpolated)."""
        return self._current_time

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        return float(self._times[-1] - self._times[0])

    @property
    def progress(self) -> float:
        """Playback progress as a fraction in [0, 1]."""
        d = self.duration
        return 0.0 if d <= 0 else (self._current_time - self._times[0]) / d

    @property
    def loop(self) -> bool:
        return self._loop

    @loop.setter
    def loop(self, value: bool):
        self._loop = bool(value)

    @property
    def controller_id(self) -> Optional[int]:
        """Unique id used by pynaviz's ControllerGroup."""
        return self._controller_id

    @controller_id.setter
    def controller_id(self, value: int):
        if self._controller_id is not None:
            raise ValueError("Controller id can be set only once!")
        self._controller_id = value

    @property
    def controller(self) -> "PlaybackController":
        """Return *self* so ControllerGroup.add() finds ``.controller``."""
        return self

    @property
    def renderer(self):
        """The pygfx renderer (or ``None`` if events not registered).

        Exposed so ControllerGroup.add() can find ``.renderer``.
        """
        if self._viewport is not None:
            return self._viewport.renderer
        return None

    # ------------------------------------------------------------------
    # Event registration
    # ------------------------------------------------------------------

    def register_events(self, viewport_or_renderer):
        """Register ``key_down`` and ``before_render`` handlers."""
        self._viewport = Viewport.from_viewport_or_renderer(viewport_or_renderer)
        renderer = self._viewport.renderer
        renderer.add_event_handler(self._on_key_down, "key_down")
        renderer.add_event_handler(self._on_key_up, "key_up")
        renderer.add_event_handler(self._on_before_render, "before_render")
        renderer.add_event_handler(self._on_pointer_down, "pointer_down")
        renderer.add_event_handler(self._on_wheel, "wheel")
        # Expose for pynaviz ControllerGroup
        self.renderer_handle_event = renderer.handle_event

    def add_handler(self, callback: Callable):
        """Add a callback invoked on every time change.

        Signature: ``callback(controller: PlaybackController)``
        """
        self._handlers.append(callback)

    def remove_handler(self, callback: Callable):
        """Remove a previously added callback."""
        self._handlers.remove(callback)

    # ------------------------------------------------------------------
    # Actions (public API)
    # ------------------------------------------------------------------

    def toggle_play(self):
        """Toggle play / pause."""
        self.playing = not self._playing
        if self._verbose:
            state = "Playing" if self._playing else "Paused"
            print(f"{state}  (speed {self._speed:.4g}x, frame {self.frame_index})")

    def speed_up(self):
        """Double playback speed (max 64x)."""
        self._speed = min(self._speed * 2, 64.0)
        if self._verbose:
            print(f"Speed: {self._speed:.4g}x")

    def speed_down(self):
        """Halve playback speed (min 1/16x)."""
        self._speed = max(self._speed / 2, 1 / 16)
        if self._verbose:
            print(f"Speed: {self._speed:.4g}x")

    def step_forward(self, n: int | None = None):
        """Advance by *n* frames (default ``step_size``)."""
        n = n if n is not None else math.ceil(self._step_size*self._speed)
        self._set_frame_position(self._frame_position + n)

    def step_backward(self, n: int | None = None):
        """Rewind by *n* frames (default ``step_size``)."""
        n = n if n is not None else math.ceil(self._step_size*self._speed)
        self._set_frame_position(self._frame_position - n)

    def jump_start(self):
        """Jump to the first frame."""
        self._set_frame_position(0.0)

    def jump_end(self):
        """Jump to the last frame."""
        self._set_frame_position(float(self._n_frames - 1))

    def go_to(self, target_time: float):
        """Jump to a specific data time (seconds)."""
        target_time = float(np.clip(target_time, self._times[0], self._times[-1]))
        self._set_frame_position(self._time_to_frame(target_time))

    def go_to_frame(self, frame: int):
        """Jump to a specific frame index."""
        self._set_frame_position(float(frame))

    # ------------------------------------------------------------------
    # pynaviz ControllerGroup interface
    # ------------------------------------------------------------------

    def sync(self, event):
        """Synchronize to a time from pynaviz's ControllerGroup.

        Parameters
        ----------
        event : SyncEvent
            Event containing ``current_time`` or ``cam_state`` with position.
        """
        if not self.enabled:
            return

        if hasattr(event, 'kwargs'):
            if "cam_state" in event.kwargs:
                new_time = event.kwargs["cam_state"]["position"][0]
            elif "current_time" in event.kwargs:
                new_time = event.kwargs["current_time"]
            else:
                return
        else:
            return

        self.go_to(new_time)

    def advance(self, delta: float = 0.025):
        """Advance current time by *delta* seconds (pynaviz interface)."""
        self.go_to(self._current_time + delta)

    def jump_next_event(self):
        """Jump to the next event marker (no-op if none defined)."""
        if self._events is None or len(self._events) == 0:
            return
        idx = np.searchsorted(self._events, self._current_time, side="right")
        if idx < len(self._events):
            self.go_to(float(self._events[idx]))

    def jump_prev_event(self):
        """Jump to the previous event marker (no-op if none defined)."""
        if self._events is None or len(self._events) == 0:
            return
        idx = np.searchsorted(self._events, self._current_time, side="left") - 1
        if idx >= 0:
            self.go_to(float(self._events[idx]))

    # ------------------------------------------------------------------
    # Tick (time advancement)
    # ------------------------------------------------------------------

    def tick(self):
        """Advance playback by wall-clock *dt*.

        Called automatically every frame if ``register_events`` was used.
        Can also be called manually for custom animation loops.
        """
        if self._in_tick:
            return
        self._in_tick = True
        try:
            self._tick_inner()
        finally:
            self._in_tick = False

    def _tick_inner(self):
        now = time.perf_counter()
        if not self._playing:
            # Continuous stepping while step keys are held down
            if self._held_step_actions:
                for action in list(self._held_step_actions):
                    getattr(self, action, lambda: None)()
            self._last_tick_time = now
            return

        wall_dt = min(now - self._last_tick_time, 0.1)  # cap to avoid jumps
        self._last_tick_time = now

        data_dt = wall_dt * self._speed
        new_time = self._current_time + data_dt

        t_start = float(self._times[0])
        t_end = float(self._times[-1])
        duration = t_end - t_start

        if self._loop and duration > 0:
            new_time = t_start + (new_time - t_start) % duration
        else:
            if new_time >= t_end:
                new_time = t_end
                self._playing = False
            elif new_time <= t_start:
                new_time = t_start
                self._playing = False

        self._current_time = new_time
        self._frame_position = self._time_to_frame(new_time)
        self._fire_handlers()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_frame_position(self, frame_pos: float):
        """Set frame position with boundary / loop handling."""
        if self._loop:
            frame_pos = frame_pos % self._n_frames
        else:
            frame_pos = float(np.clip(frame_pos, 0, self._n_frames - 1))
            if frame_pos >= self._n_frames - 1:
                self._playing = False

        self._frame_position = frame_pos
        self._current_time = self._frame_to_time(frame_pos)
        # Reset wall-clock so the next tick() doesn't accumulate stale dt
        self._last_tick_time = time.perf_counter()
        self._fire_handlers()

    def _frame_to_time(self, frame_pos: float) -> float:
        """Fractional frame position → data time (linear interpolation)."""
        idx = int(frame_pos)
        idx = max(0, min(idx, self._n_frames - 2))
        frac = frame_pos - idx
        t0 = self._times[idx]
        t1 = self._times[min(idx + 1, self._n_frames - 1)]
        return float(t0 + frac * (t1 - t0))

    def _time_to_frame(self, t: float) -> float:
        """Data time → fractional frame position (linear interpolation)."""
        idx = int(np.searchsorted(self._times, t, side="right")) - 1
        idx = max(0, min(idx, self._n_frames - 2))
        t0 = self._times[idx]
        t1 = self._times[idx + 1]
        if t1 > t0:
            frac = float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))
        else:
            frac = 0.0
        return float(idx + frac)

    def _fire_handlers(self):
        for h in self._handlers:
            h(self)
        if self._viewport is not None:
            self._viewport.renderer.request_draw()

    def _on_before_render(self, event):
        self.tick()

    def _on_key_down(self, event):
        action_name = self.key_bindings.get(event.key)
        if action_name is not None:
            if action_name in ("step_forward", "step_backward"):
                self._held_step_actions.add(action_name)
            method = getattr(self, action_name, None)
            if method is not None:
                method()

    def _on_key_up(self, event):
        action_name = self.key_bindings.get(event.key)
        if action_name in ("step_forward", "step_backward"):
            self._held_step_actions.discard(action_name)

    def _on_wheel(self, event):
        """Adjust playback speed with Ctrl + scroll wheel."""
        if not event.modifiers or "Control" not in event.modifiers:
            return
        if event.dy < 0:
            self.speed_up()
        elif event.dy > 0:
            self.speed_down()

    def _on_pointer_down(self, event):
        """Toggle play on right-click."""
        if event.button == 2:
            self.toggle_play()