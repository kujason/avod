import abc

import numpy as np


class BevGenerator:

    @abc.abstractmethod
    def generate_bev(self, **params):
        """Generates BEV maps

        Args:
            **params: additional keyword arguments for
                specific implementations of BevGenerator.

        Returns:
            Dictionary with entries for height maps and one density map
                height_maps: list of height maps
                density_map: density map
        """
        pass

    def _create_density_map(self,
                            num_divisions,
                            voxel_indices_2d,
                            num_pts_per_voxel,
                            norm_value):

        # Create empty density map
        density_map = np.zeros((num_divisions[0],
                                num_divisions[2]))

        # Only update pixels where voxels have num_pts values
        density_map[voxel_indices_2d[:, 0], voxel_indices_2d[:, 1]] = \
            np.minimum(1.0, np.log(num_pts_per_voxel + 1) / norm_value)

        # Density is calculated as min(1.0, log(N+1)/log(x))
        # x=64 for stereo, x=16 for lidar, x=64 for depth
        density_map = np.flip(density_map.transpose(), axis=0)

        return density_map
