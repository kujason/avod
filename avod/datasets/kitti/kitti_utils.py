import os

import numpy as np

from wavedata.tools.core.voxel_grid_2d import VoxelGrid2D
from wavedata.tools.core.voxel_grid import VoxelGrid
from wavedata.tools.obj_detection import obj_utils

from avod.builders import bev_generator_builder
from avod.core.label_cluster_utils import LabelClusterUtils
from avod.core.mini_batch_utils import MiniBatchUtils


class KittiUtils(object):
    # Definition for difficulty levels
    # These values are from Kitti dataset
    # 0 - easy, 1 - medium, 2 - hard
    HEIGHT = (40, 25, 25)
    OCCLUSION = (0, 1, 2)
    TRUNCATION = (0.15, 0.3, 0.5)

    def __init__(self, dataset):

        self.dataset = dataset

        # Label Clusters
        self.label_cluster_utils = LabelClusterUtils(self.dataset)

        self.clusters, self.std_devs = [None, None]

        # BEV source from dataset config
        self.bev_source = self.dataset.bev_source

        # Parse config
        self.config = dataset.config.kitti_utils_config
        self.area_extents = np.reshape(self.config.area_extents, (3, 2))
        self.bev_extents = self.area_extents[[0, 2]]
        self.voxel_size = self.config.voxel_size
        self.anchor_strides = np.reshape(self.config.anchor_strides, (-1, 2))

        self.bev_generator = bev_generator_builder.build(
            self.config.bev_generator, self)

        self._density_threshold = self.config.density_threshold

        # Check that depth maps folder exists
        if self.bev_source == 'depth' and \
                not os.path.exists(self.dataset.depth_dir):
            raise FileNotFoundError(
                'Could not find depth maps, please run '
                'demos/save_lidar_depth_maps.py in wavedata first')

        # Mini Batch Utils
        self.mini_batch_utils = MiniBatchUtils(self.dataset)
        self._mini_batch_dir = self.mini_batch_utils.mini_batch_dir

        # Label Clusters
        self.clusters, self.std_devs = \
            self.label_cluster_utils.get_clusters()

    def class_str_to_index(self, class_str):
        """
        Converts an object class type string into a integer index

        Args:
            class_str: the object type (e.g. 'Car', 'Pedestrian', or 'Cyclist')

        Returns:
            The corresponding integer index for a class type, starting at 1
            (0 is reserved for the background class).
            Returns -1 if we don't care about that class type.
        """
        if class_str in self.dataset.classes:
            return self.dataset.classes.index(class_str) + 1

        raise ValueError('Invalid class string {}, not in {}'.format(
            class_str, self.dataset.classes))

    def create_slice_filter(self, point_cloud, area_extents,
                            ground_plane, ground_offset_dist, offset_dist):
        """ Creates a slice filter to take a slice of the point cloud between
            ground_offset_dist and offset_dist above the ground plane

        Args:
            point_cloud: Point cloud in the shape (3, N)
            area_extents: 3D area extents
            ground_plane: ground plane coefficients
            offset_dist: max distance above the ground
            ground_offset_dist: min distance above the ground plane

        Returns:
            A boolean mask if shape (N,) where
                True indicates the point should be kept
                False indicates the point should be removed
        """

        # Filter points within certain xyz range and offset from ground plane
        offset_filter = obj_utils.get_point_filter(point_cloud, area_extents,
                                                   ground_plane, offset_dist)

        # Filter points within 0.2m of the road plane
        road_filter = obj_utils.get_point_filter(point_cloud, area_extents,
                                                 ground_plane,
                                                 ground_offset_dist)

        slice_filter = np.logical_xor(offset_filter, road_filter)
        return slice_filter

    def create_bev_maps(self, point_cloud, ground_plane):
        """ Calculates bev maps

        Args:
            point_cloud: point cloud
            ground_plane: ground_plane coefficients

        Returns:
            Dictionary with entries for each type of map (e.g. height, density)
        """

        bev_maps = self.bev_generator.generate_bev(self.bev_source,
                                                   point_cloud,
                                                   ground_plane,
                                                   self.area_extents,
                                                   self.voxel_size)

        return bev_maps

    def get_anchors_info(self, classes_name, anchor_strides, sample_name):

        anchors_info = self.mini_batch_utils.get_anchors_info(classes_name,
                                                              anchor_strides,
                                                              sample_name)
        return anchors_info

    def get_point_cloud(self, source, img_idx, image_shape=None):
        """ Gets the points from the point cloud for a particular image,
            keeping only the points within the area extents, and takes a slice
            between self._ground_filter_offset and self._offset_distance above
            the ground plane

        Args:
            source: point cloud source, e.g. 'lidar'
            img_idx: An integer sample image index, e.g. 123 or 500
            image_shape: image dimensions (h, w), only required when
                source is 'lidar' or 'depth'

        Returns:
            The set of points in the shape (N, 3)
        """

        if source == 'lidar':
            # wavedata wants im_size in (w, h) order
            im_size = [image_shape[1], image_shape[0]]

            point_cloud = obj_utils.get_lidar_point_cloud(
                img_idx, self.dataset.calib_dir, self.dataset.velo_dir,
                im_size=im_size)

        else:
            raise ValueError("Invalid source {}".format(source))

        return point_cloud

    def get_ground_plane(self, sample_name):
        """Reads the ground plane for the sample

        Args:
            sample_name: name of the sample, e.g. '000123'

        Returns:
            ground_plane: ground plane coefficients
        """
        ground_plane = obj_utils.get_road_plane(int(sample_name),
                                                self.dataset.planes_dir)
        return ground_plane

    def _apply_offset_filter(
            self,
            point_cloud,
            ground_plane,
            offset_dist=2.0):
        """ Applies an offset filter to the point cloud

        Args:
            point_cloud: A point cloud in the shape (3, N)
            ground_plane: ground plane coefficients,
                if None, will only filter to the area extents
            offset_dist: (optional) height above ground plane for filtering

        Returns:
            Points filtered with an offset filter in the shape (N, 3)
        """
        offset_filter = obj_utils.get_point_filter(
            point_cloud, self.area_extents, ground_plane, offset_dist)

        # Transpose point cloud into N x 3 points
        points = np.asarray(point_cloud).T

        filtered_points = points[offset_filter]

        return filtered_points

    def _apply_slice_filter(self, point_cloud, ground_plane,
                            height_lo=0.2, height_hi=2.0):
        """ Applies a slice filter to the point cloud

        Args:
            point_cloud: A point cloud in the shape (3, N)
            ground_plane: ground plane coefficients
            height_lo: (optional) lower height for slicing
            height_hi: (optional) upper height for slicing

        Returns:
            Points filtered with a slice filter in the shape (N, 3)
        """

        slice_filter = self.create_slice_filter(point_cloud,
                                                self.area_extents,
                                                ground_plane,
                                                height_lo, height_hi)

        # Transpose point cloud into N x 3 points
        points = np.asarray(point_cloud).T

        filtered_points = points[slice_filter]

        return filtered_points

    def create_sliced_voxel_grid_2d(self, sample_name, source,
                                    image_shape=None):
        """Generates a filtered 2D voxel grid from point cloud data

        Args:
            sample_name: image name to generate stereo pointcloud from
            source: point cloud source, e.g. 'lidar'
            image_shape: image dimensions [h, w], only required when
                source is 'lidar' or 'depth'

        Returns:
            voxel_grid_2d: 3d voxel grid from the given image
        """
        img_idx = int(sample_name)
        ground_plane = obj_utils.get_road_plane(img_idx,
                                                self.dataset.planes_dir)

        point_cloud = self.get_point_cloud(source, img_idx,
                                           image_shape=image_shape)
        filtered_points = self._apply_slice_filter(point_cloud, ground_plane)

        # Create Voxel Grid
        voxel_grid_2d = VoxelGrid2D()
        voxel_grid_2d.voxelize_2d(filtered_points, self.voxel_size,
                                  extents=self.area_extents,
                                  ground_plane=ground_plane,
                                  create_leaf_layout=True)

        return voxel_grid_2d

    def create_voxel_grid_3d(self, sample_name, ground_plane,
                             source='lidar',
                             filter_type='slice'):
        """Generates a filtered voxel grid from stereo data

            Args:
                sample_name: image name to generate stereo pointcloud from
                ground_plane: ground plane coefficients
                source: source of the pointcloud to create bev images
                    either "stereo" or "lidar"
                filter_type: type of point filter to use
                    'slice' for slice filtering (offset + ground)
                    'offset' for offset filtering only
                    'area' for area filtering only

           Returns:
               voxel_grid_3d: 3d voxel grid from the given image
        """
        img_idx = int(sample_name)

        points = self.get_point_cloud(source, img_idx)

        if filter_type == 'slice':
            filtered_points = self._apply_slice_filter(points, ground_plane)
        elif filter_type == 'offset':
            filtered_points = self._apply_offset_filter(points, ground_plane)
        elif filter_type == 'area':
            # A None ground plane will filter the points to the area extents
            filtered_points = self._apply_offset_filter(points, None)
        else:
            raise ValueError("Invalid filter_type {}, should be 'slice', "
                             "'offset', or 'area'".format(filter_type))

        # Create Voxel Grid
        voxel_grid_3d = VoxelGrid()
        voxel_grid_3d.voxelize(filtered_points, self.voxel_size,
                               extents=self.area_extents)

        return voxel_grid_3d

    def filter_labels(self, objects,
                      classes=None,
                      difficulty=None,
                      max_occlusion=None):
        """Filters ground truth labels based on class, difficulty, and
        maximum occlusion

        Args:
            objects: A list of ground truth instances of Object Label
            classes: (optional) classes to filter by, if None
                all classes are used
            difficulty: (optional) KITTI difficulty rating as integer
            max_occlusion: (optional) maximum occlusion to filter objects

        Returns:
            filtered object label list
        """
        if classes is None:
            classes = self.dataset.classes

        objects = np.asanyarray(objects)
        filter_mask = np.ones(len(objects), dtype=np.bool)

        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]

            if filter_mask[obj_idx]:
                if not self._check_class(obj, classes):
                    filter_mask[obj_idx] = False
                    continue

            # Filter by difficulty (occlusion, truncation, and height)
            if difficulty is not None and \
                    not self._check_difficulty(obj, difficulty):
                filter_mask[obj_idx] = False
                continue

            if max_occlusion and \
                    obj.occlusion > max_occlusion:
                filter_mask[obj_idx] = False
                continue

        return objects[filter_mask]

    def _check_difficulty(self, obj, difficulty):
        """This filters an object by difficulty.
        Args:
            obj: An instance of ground-truth Object Label
            difficulty: An int defining the KITTI difficulty rate
        Returns: True or False depending on whether the object
            matches the difficulty criteria.
        """

        return ((obj.occlusion <= self.OCCLUSION[difficulty]) and
                (obj.truncation <= self.TRUNCATION[difficulty]) and
                (obj.y2 - obj.y1) >= self.HEIGHT[difficulty])

    def _check_class(self, obj, classes):
        """This filters an object by class.
        Args:
            obj: An instance of ground-truth Object Label
        Returns: True or False depending on whether the object
            matches the desired class.
        """
        return obj.type in classes
