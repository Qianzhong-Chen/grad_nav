# Using a Custom Scene with grad_nav

How to train on your own nerfstudio 3DGS scene instead of the provided maps.
The common failure (see issue #8) is a coordinate frame mismatch that makes path
planning fail with `RuntimeError: Could not infer dtype of NoneType`. This guide
explains how to avoid it.

## 1. Train and export

Train a `splatfacto` model in nerfstudio, then export a point cloud:

```bash
ns-train splatfacto --data ./processed/
ns-export pointcloud --load-config outputs/<scene>/splatfacto/<timestamp>/config.yml --output-dir ./exports/
```

Drop the `.ply` in `envs/assets/point_cloud/` and add it to the `maps` dict in
`envs/drone_long_traj.py`.

## 2. The coordinate frame problem

nerfstudio normalizes the scene, so `(0, 0, 0)` is near the camera centroid, not
a physical landmark. grad_nav's planner works in this point cloud frame and
applies `point_could_origin_offset = [-6, 0, 0]`, so the built-in maps start
planning at `(-6, 0, 1.4)`. For the provided scenes that point is free space; for
a raw export it is almost always inside geometry, so A* finds no path and returns
`None`.

Pick `traj_start` and `traj_dest` from free space in your own scene. Open the
`.ply` in any viewer (e.g. `o3d.visualization.draw_geometries([pcd])`) and read
off coordinates well clear of walls, floor, and ceiling.

## 3. Validate before training

```bash
python tools/validate_scene.py --pcd envs/assets/point_cloud/<your_scene>.ply \
  --start <x> <y> <z> --dest <x> <y> <z> --radius 0.3
```

It reports the nearest-obstacle distance for each point and flags any that are
inside occupied space. Fix every `FAIL` before training.

## 4. Set the coordinates

In your `elif self.map_name == '<your_scene>':` branch in `drone_long_traj.py`,
set `traj_start` / `traj_dest` to the validated point-cloud-frame coordinates and
`reward_wp` to the gate positions in training frame (point cloud frame minus
`point_could_origin_offset`). Copy the `examples/cfg/gradnav/drone_long_traj.yaml`
config and set `map_name` to your scene key.
