import numpy as np
import tensorflow as tf


def add_feature_maps_from_dict(end_points, layer_name):
    """ Calls add_feature_maps for a specified layer
        in a dictionary of end points

    Args:
        end_points: dictionary of network end points
        layer_name: dict key of the layer to add
    """
    feature_maps = end_points.get(layer_name)
    add_feature_maps(feature_maps, layer_name)


def add_feature_maps(feature_maps, layer_name):
    """ Adds an image summary showing tiled feature maps

    Args:
        feature_maps: a tensor of feature maps to show, dimensions should be
            (1, ?, ?, ?) (batch_size, height, width, depth)
        layer_name: name of the layer which will show up in tensorboard
    """
    with tf.name_scope(layer_name):
        batch, maps_height, maps_width, num_maps = np.array(
            feature_maps.shape).astype(np.int32)

        # Resize to a visible size
        map_width_out = 300
        ratio = map_width_out / maps_width
        map_height_out = int(maps_height * ratio)
        map_size_out = tf.convert_to_tensor([map_height_out, map_width_out],
                                            tf.int32)

        resized_maps = tf.image.resize_bilinear(feature_maps, map_size_out)

        # Take first image only
        output = tf.slice(resized_maps, (0, 0, 0, 0), (1, -1, -1, -1))
        output = tf.reshape(output, (map_height_out, map_width_out, num_maps))

        # Add padding around each map
        map_width_out += 5
        map_height_out += 5
        output = tf.image.resize_image_with_crop_or_pad(
            output, map_height_out, map_width_out)

        # Find good image size for display
        map_sizes = [1, 32, 64, 128, 256, 512]
        # columns, rows
        image_sizes = [(1, 1), (4, 8), (8, 8), (8, 16), (8, 32), (16, 32)]
        size_idx = map_sizes.index(num_maps)
        desired_image_size = image_sizes[size_idx]
        image_width = desired_image_size[0]
        image_height = desired_image_size[1]

        # Arrange maps into a grid
        output = tf.reshape(output,
                            (map_height_out, map_width_out, image_height,
                             image_width))
        output = tf.transpose(output, (2, 0, 3, 1))
        output = tf.reshape(output, (1, image_height * map_height_out,
                                     image_width * map_width_out, 1))

        layer_name = layer_name.split('/')[-1]
        tf.summary.image(layer_name, output, max_outputs=16)


def add_scalar_summary(summary_name, scalar_value,
                       summary_writer, global_step):
    """ Adds a single scalar summary value to the logs without adding a
        summary node to the graph

    Args:
        summary_name: name of the summary to add
        scalar_value: value of the scalar
        summary_writer: a summary writer object
        global_step: the current global step
    """

    avg_summary = tf.Summary()
    avg_summary.value.add(tag=summary_name,
                          simple_value=scalar_value)

    summary_writer.add_summary(avg_summary, global_step)


def summaries_to_keep(summaries,
                      global_summaries,
                      histograms=True,
                      input_imgs=True,
                      input_bevs=True):

    if histograms and input_imgs and input_bevs:
        # Keep everything
        summaries |= global_summaries

    else:
        for summary in summaries.copy():
            name = summary.name
            if not histograms and name.startswith('histograms'):
                summaries.remove(summary)
            if not input_imgs and name.startswith('img_'):
                summaries.remove(summary)
            if not input_bevs and name.startswith('bev_'):
                summaries.remove(summary)

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    return summary_op
