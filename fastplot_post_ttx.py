import fastplotlib as fpl
import numpy as np

wake_embds = np.load("post_ttx_umap_euc.npy")

fig_gpu = fpl.Figure(shape=(1, 1), size=(1400, 500),
                     cameras=[['3d']],)

# 3D
scatter_img = fig_gpu[0, 0].add_scatter(
    data=wake_embds,
    cmap='viridis',
    sizes=5,
    alpha=0.85)

fig_gpu.show()

if __name__ == "__main__":
    fpl.loop.run()