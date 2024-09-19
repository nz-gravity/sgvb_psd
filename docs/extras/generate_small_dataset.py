import h5py
import numpy as np

data = []
CHANNELS = ["X", "Y", "Z"]
for i, channel in enumerate(CHANNELS):
    with h5py.File(f"{channel}_ETnoise_GP.hdf5", "r") as f:
        data.append(f[f"E{i+1}:STRAIN"][:])
time = np.linspace(0, 2000, len(data[0]))
data = np.array(data)
data = data[:, : 2**19]
time = time[: 2**19]
with h5py.File("et_data.h5", "w") as f:
    # Create datasets for X, Y, Z
    f.create_dataset("X", data=data[0])
    f.create_dataset("Y", data=data[1])
    f.create_dataset("Z", data=data[2])
    f.create_dataset("time", data=time)
