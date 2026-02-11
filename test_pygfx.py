from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx
import numpy as np


canvas = RenderCanvas(max_fps=60, title="Pygfx Test")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#000"))

grid = gfx.Grid(
    None,
    gfx.GridMaterial(
        major_step=100,
        minor_step=10,
        thickness_space="world",
        major_thickness=2,
        minor_thickness=0.1,
        infinite=True,
    ),
    orientation="xz",
)
grid.local.y = -120
scene.add(grid)

# Create a bunch of points
n = 1000
positions = np.random.normal(0, 50, (n, 3)).astype(np.float32)
sizes = np.random.rand(n).astype(np.float32) * 50
colors = np.random.rand(n, 4).astype(np.float32)
geometry = gfx.Geometry(positions=positions, sizes=sizes, colors=colors)

material = gfx.PointsMaterial(
    color_mode="vertex",
    size_mode="vertex",
    size_space="world",
)
points = gfx.Points(geometry, material)
scene.add(points)

camera = gfx.PerspectiveCamera(70)
camera.show_object(scene)
controller = gfx.FlyController(camera, register_events=renderer)

stats = gfx.Stats(viewport=renderer)


def anim():
    with stats:
        renderer.render(scene, camera, flush=False)
    stats.render()
    canvas.request_draw()

if __name__ == "__main__":
    canvas.request_draw(anim)
    loop.run()