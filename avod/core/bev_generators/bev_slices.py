import numpy as np

from wavedata.tools.core.voxel_grid_2d import VoxelGrid2D

from avod.core.bev_generators import bev_generator


class BevSlices(bev_generator.BevGenerator):

    NORM_VALUES = {
        'lidar': np.log(16),
    }

    def __init__(self, config, kitti_utils):
        """BEV maps created using slices of the point cloud.

        Args:
            config: bev_generator protobuf config
            kitti_utils: KittiUtils object
        """

        # Parse config
        self.height_lo = config.height_lo
        self.height_hi = config.height_hi
        self.num_slices = config.num_slices

        self.kitti_utils = kitti_utils

        # Pre-calculated values
        self.height_per_division = \
            (self.height_hi - self.height_lo) / self.num_slices

    def generate_bev(self,
                     source,
                     point_cloud,
                     ground_plane,
                     area_extents,
                     voxel_size):
        """Generates the BEV maps dictionary. One height map is created for
        each slice of the point cloud. One density map is created for
        the whole point cloud.

        Args:
            source: point cloud source
            point_cloud: point cloud (3, N)
            ground_plane: ground plane coefficients
            area_extents: 3D area extents
                [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
            voxel_size: voxel size in m

        Returns:
            BEV maps dictionary
                height_maps: list of height maps
                density_map: density map
        """

        all_points = np.transpose(point_cloud)

        height_maps = []

        for slice_idx in range(self.num_slices):

            height_lo = self.height_lo + slice_idx * self.height_per_division
            height_hi = height_lo + self.height_per_division

            slice_filter = self.kitti_utils.create_slice_filter(
                point_cloud,
                area_extents,
                ground_plane,
                height_lo,
                height_hi)

            # Apply slice filter
            slice_points = all_points[slice_filter]

            if len(slice_points) > 1:

                # Create Voxel Grid 2D
                voxel_grid_2d = VoxelGrid2D()
                voxel_grid_2d.voxelize_2d(
                    slice_points, voxel_size,
                    extents=area_extents,
                    ground_plane=ground_plane,
                    create_leaf_layout=False)

                # Remove y values (all 0)
                voxel_indices = voxel_grid_2d.voxel_indices[:, [0, 2]]

            # Create empty BEV images
            height_map = np.zeros((voxel_grid_2d.num_divisions[0],
                                   voxel_grid_2d.num_divisions[2]))

            # Only update pixels where voxels have max height values,
            # and normalize by height of slices
            voxel_grid_2d.heights = voxel_grid_2d.heights - height_lo
            height_map[voxel_indices[:, 0], voxel_indices[:, 1]] = \
                np.asarray(voxel_grid_2d.heights) / self.height_per_division

            height_maps.append(height_map)

        # Rotate height maps 90 degrees
        # (transpose and flip) is faster than np.rot90
        height_maps_out = [np.flip(height_maps[map_idx].transpose(), axis=0)
                           for map_idx in range(len(height_maps))]

        density_slice_filter = self.kitti_utils.create_slice_filter(
            point_cloud,
            area_extents,
            ground_plane,
            self.height_lo,
            self.height_hi)

        density_points = all_points[density_slice_filter]

        # Create Voxel Grid 2D
        density_voxel_grid_2d = VoxelGrid2D()
        density_voxel_grid_2d.voxelize_2d(
            density_points,
            voxel_size,
            extents=area_extents,
            ground_plane=ground_plane,
            create_leaf_layout=False)

        # Generate density map
        density_voxel_indices_2d = \
            density_voxel_grid_2d.voxel_indices[:, [0, 2]]

        density_map = self._create_density_map(
            num_divisions=density_voxel_grid_2d.num_divisions,
            voxel_indices_2d=density_voxel_indices_2d,
            num_pts_per_voxel=density_voxel_grid_2d.num_pts_in_voxel,
            norm_value=self.NORM_VALUES[source])

        bev_maps = dict()
        bev_maps['height_maps'] = height_maps_out
        bev_maps['density_map'] = density_map

        return bev_maps
