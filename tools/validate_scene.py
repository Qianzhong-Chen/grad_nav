import sys
import argparse

import numpy as np
import open3d as o3d


def nearest_distance(query_xyz, points):
    diff = points - query_xyz[np.newaxis, :]
    dists = np.linalg.norm(diff, axis=1)
    idx = np.argmin(dists)
    return float(dists[idx]), points[idx]


def check_point(label, xyz, points, radius):
    dist, nearest = nearest_distance(xyz, points)
    ok = dist >= radius
    status = 'OK' if ok else 'FAIL'
    print(f'  [{status}] {label}: xyz=({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})  nearest_obstacle={dist:.3f} m')
    if not ok:
        print(f'         nearest point: ({nearest[0]:.3f}, {nearest[1]:.3f}, {nearest[2]:.3f})')
    return ok


def main():
    parser = argparse.ArgumentParser(description='Check whether start/dest/waypoints are in free space in a .ply point cloud.')
    parser.add_argument('--pcd', required=True, help='Path to the .ply file.')
    parser.add_argument('--start', required=True, nargs=3, type=float, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--dest', required=True, nargs=3, type=float, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--waypoints', nargs='+', type=float, default=None,
                        help='Flat list of floats (multiple of 3).')
    parser.add_argument('--radius', type=float, default=0.3,
                        help='Collision radius in meters (default: 0.3).')

    args = parser.parse_args()

    if args.waypoints is not None and len(args.waypoints) % 3 != 0:
        print('ERROR: --waypoints must contain a multiple of 3 values')
        sys.exit(1)

    print(f'Loading point cloud: {args.pcd}')
    pcd = o3d.io.read_point_cloud(args.pcd)
    points = np.asarray(pcd.points)
    print(f'  Loaded {len(points)} points.')
    print(f'  Bounding box: min={points.min(axis=0).round(3)}  max={points.max(axis=0).round(3)}')

    print(f'\nChecking coordinates (radius={args.radius} m):')
    all_ok = True
    all_ok &= check_point('start', np.array(args.start), points, args.radius)
    all_ok &= check_point('dest', np.array(args.dest), points, args.radius)

    if args.waypoints:
        wp_array = np.array(args.waypoints).reshape(-1, 3)
        for i, wp in enumerate(wp_array):
            all_ok &= check_point(f'waypoint[{i}]', wp, points, args.radius)

    print()
    if all_ok:
        print('All points are in free space. Coordinates look valid.')
    else:
        print('One or more points are inside occupied space -- adjust coordinates before training.')
        print('Note: nerfstudio normalizes the scene so (0,0,0) is near the camera centroid, not a')
        print('physical landmark. The default offset in drone_long_traj.py places start at (-6,0,z)')
        print('in point cloud frame -- verify this is free space for your scene. See CUSTOM_SCENE.md.')
        sys.exit(1)


if __name__ == '__main__':
    main()
