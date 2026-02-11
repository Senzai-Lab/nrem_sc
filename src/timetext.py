
# ---------------------------------------------------------------------------
# TimeText
# ---------------------------------------------------------------------------
import pygfx as gfx
from pygfx.utils.viewport import Viewport
from .playback import PlaybackController

class TimeText(gfx.Group):
    """Screen-space text showing current time.

    Follows the same usage pattern as ``gfx.Stats``::

        overlay = TimeText(viewport=renderer, position='bottom-left')

        def anim():
            renderer.render(scene, camera, flush=False)
            overlay.update(playback)
            overlay.render()

    Parameters
    ----------
    viewport : Renderer | Viewport
        Used for positioning and rendering.
    position : str
        Corner placement: ``"top-left"`` (default) or ``"bottom-left"``.
    font_size : int
        Font size in pixels.
    color : str
        Text color.
    """

    def __init__(
        self,
        viewport,
        position: str = "top-left"
    ):
        super().__init__()
        
        self._line_height = 16
        font_size: int = 14
        color: str = "#0f0"
        
        self._viewport = Viewport.from_viewport_or_renderer(viewport)
        self._position = position

        self.info_text = gfx.Text(
            material=gfx.TextMaterial(color=color),
            text="",
            screen_space=True,
            font_size=font_size,
            anchor="topleft",
        )
        self.add(self.info_text)

        self.camera = gfx.ScreenCoordsCamera()
        self._update_positions()
        self._viewport.renderer.add_event_handler(self._update_positions, "resize")

    # ------------------------------------------------------------------

    def _update_positions(self, event=None):
        _, height = self._viewport.logical_size
        pad = 10
        if "top" in self._position:
            y = height - pad
        else:
            y = pad + 60  # approximate 3-line text height
        self.info_text.local.position = (pad, y, 0)

    def update(self, playback: PlaybackController | None = None, **kwargs):
        """Refresh the overlay text.

        Can accept a :class:`PlaybackController` directly, or explicit
        keyword arguments ``time``, ``frame_index``, ``n_frames``,
        ``speed``, ``playing``.
        """
        if playback is not None:
            t = playback.current_time
            idx = playback.frame_index
            n = playback.n_frames
            spd = playback.speed
            playing = playback.playing
        else:
            t = kwargs.get("time", 0.0)
            idx = kwargs.get("frame_index", 0)
            n = kwargs.get("n_frames", 0)
            spd = kwargs.get("speed", 1.0)
            playing = kwargs.get("playing", False)

        state_icon = ">" if playing else "||"
        lines = [
            f"t = {t:.4f} s",
            f"frame {idx} / {n - 1}",
            f"{state_icon}  {spd:.4g}x",
        ]
        self.info_text.set_text("\n".join(lines))

    def render(self, flush=True):
        """Render the overlay.  Call after your main ``renderer.render(..., flush=False)``."""
        self._viewport.render(self, self.camera, flush=flush)
