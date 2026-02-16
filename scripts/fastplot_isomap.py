import numpy as np
import pynapple as nap
import fastplotlib as fpl
from sklearn.manifold import SpectralEmbedding
from scipy.signal.windows import gaussian
import matplotlib.colors as mcolors

from src.constants import INTERIM_DATA_PATH, PROCESSED_DATA_PATH

# Load data
hd_spikes       = nap.load_file(PROCESSED_DATA_PATH / "hd_spikes_total.npz")
hd_angle        = nap.load_file(PROCESSED_DATA_PATH / "angle_openfield.npz")
active_wake     = nap.load_file(PROCESSED_DATA_PATH / "active_wake.npz")

# wake_embds = np.load(INTERIM_DATA_PATH / "umap_embds_3d.npy")
# hd_angle = np.load(INTERIM_DATA_PATH / "umap_embds_angle.npy")

# Remove neurons with low kappa values (von mises concentration parameter)
noisy_units = [14, 30, 25, 19, 17, 2, 11, 20, 38, 37, 1, 15, 16, 3, 0, 18]
labels = ['good']*len(hd_spikes)
for nu in noisy_units:
    labels[nu] = 'noisy'
hd_spikes.set_info({'labels': labels})
hd_spikes = hd_spikes.getby_category('labels')['good']
 
# Preprocess data
wake_bin = 0.4
kernel_small = gaussian(M=50, std=3)

hd_angle = hd_angle.bin_average(bin_size=wake_bin, ep=active_wake)
wake_binned = hd_spikes.count(bin_size=wake_bin, ep=active_wake)
wake_rate = np.sqrt(wake_binned.convolve(kernel_small))

model = SpectralEmbedding(n_neighbors=200, n_components=3, n_jobs=-1)
wake_embds = model.fit_transform(wake_rate)

model = SpectralEmbedding(n_neighbors=200, n_components=2, n_jobs=-1)
wake_embds_2d = model.fit_transform(wake_rate)

# Visualize with fastplot

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(wake_embds)
wake_embds = scaler.transform(wake_embds)

scaler.fit(wake_embds_2d)
wake_embds_2d = scaler.transform(wake_embds_2d)

# Normalize
hsv_colors = np.ones((len(hd_angle), 3))
hsv_colors[:, 0] = hd_angle / 360.0

# HSV to RGBA
rgb_colors = mcolors.hsv_to_rgb(hsv_colors)
colors_rgba = np.column_stack((rgb_colors, np.ones(len(rgb_colors))))


fig_gpu = fpl.Figure(shape=(1, 2), size=(1400, 500),
                     cameras=[['3d', '2d']],)

# 3D
scatter_img = fig_gpu[0, 0].add_scatter(
    data=wake_embds,
    colors=colors_rgba,
    sizes=5,
    alpha=0.85
)

# 2D
scatter_2d = fig_gpu[0, 1].add_scatter(
    data=wake_embds_2d,
    colors=colors_rgba,
    sizes=4,
    alpha=0.8
)

fig_gpu.show()

if __name__ == "__main__":
    fpl.loop.run()