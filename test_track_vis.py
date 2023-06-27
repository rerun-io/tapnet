import numpy as np
import rerun as rr

t = np.linspace(0, 5, 1000)

rr.init("track test")

xys = np.stack((np.cos(t), np.sin(t)), axis=-1)
print(xys.shape)

rr.init("track_vis", spawn=True)
for t, xy in zip(t, xys):
  rr.set_time_seconds("time", t)
  rr.log_point("current_point", xy, radius=0.01, color=[0.9, 0.2, 0.2])
  rr.log_point("track", xy, radius=0.005, color=[0.9, 0.2, 0.2])


# good plan seems to be
#   log all points as one entity
#   only log current
