from avod.core.bev_generators import bev_slices


def build(bev_maps_type_config, kitti_utils):

    bev_maps_type = bev_maps_type_config.WhichOneof('bev_maps_type')

    if bev_maps_type == 'slices':
        return bev_slices.BevSlices(
            bev_maps_type_config.slices, kitti_utils)

    raise ValueError('Invalid bev_maps_type', bev_maps_type)
