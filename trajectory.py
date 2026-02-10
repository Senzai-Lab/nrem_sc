import pandas as pd
import pynapple as nap
import pygfx as gfx
from rendercanvas.glfw import RenderCanvas, loop

from trajectoryViewer import TrajectoryViewer
from manifold_viewer_3d_fps import TrajectoryViewer3D

fname = r"R:\Basic_Sciences\Phys\SenzaiLab\Tuguldur\points_3d.csv"
data = pd.read_csv(fname, skiprows=7)
t = data['Time (Seconds)'].to_numpy()
xyz = data[['X.1', 'Y.1', 'Z.1']].to_numpy()
tsdf = nap.TsdFrame(t=t, d=xyz, columns=['x','y','z'])


grid = gfx.Grid(
    None,
    gfx.GridMaterial(
        major_step=100,
        minor_step=10,
        thickness_space="world",
        major_thickness=1,
        minor_thickness=0.1,
        infinite=True,
    ),
    orientation="xz",
)
grid.local.y = 200

v = TrajectoryViewer3D()
v.scene.add(grid)
v.add_trajectory(data=tsdf, trail_length=100)

if __name__ == "__main__":
    loop.run()