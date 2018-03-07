from tensorflow.contrib import slim

from avod.core.avod_fc_layers import avod_fc_layer_utils


def build(fc_layers_config,
          input_rois, input_weights,
          num_final_classes, box_rep,
          is_training,
          end_points_collection):
    """Builds fusion layers

    Args:
        fc_layers_config: Fully connected layers config object
        input_rois: List of input roi feature maps
        input_weights: List of weights for each input e.g. [1.0, 1.0]
        num_final_classes: Final number of output classes, including
            'Background'
        box_rep: Box representation (e.g. 'box_3d', 'box_8c', 'box_4c')
        is_training: Whether the network is training or evaluating
        end_points_collection: End points collection to add entries to

    Returns:
        cls_logits: Output classification logits
        offsets: Output offsets
        angle_vectors: Output angle vectors (or None)
        end_points: End points dict
    """

    # Parse configs
    fusion_type = fc_layers_config.fusion_type
    fusion_method = fc_layers_config.fusion_method

    num_layers = fc_layers_config.num_layers
    layer_sizes = fc_layers_config.layer_sizes
    l2_weight_decay = fc_layers_config.l2_weight_decay
    keep_prob = fc_layers_config.keep_prob

    # Validate values
    if not len(input_weights) == len(input_rois):
        raise ValueError('Length of input_weights does not match length of '
                         'input_rois')
    if not len(layer_sizes) == num_layers:
        raise ValueError('Length of layer_sizes does not match num_layers')

    if fusion_type == 'early':
        cls_logits, offsets, angle_vectors = \
            _early_fusion_fc_layers(num_layers=num_layers,
                                    layer_sizes=layer_sizes,
                                    input_rois=input_rois,
                                    input_weights=input_weights,
                                    fusion_method=fusion_method,
                                    l2_weight_decay=l2_weight_decay,
                                    keep_prob=keep_prob,
                                    num_final_classes=num_final_classes,
                                    box_rep=box_rep,
                                    is_training=is_training)

    elif fusion_type == 'late':
        cls_logits, offsets, angle_vectors = \
            _late_fusion_fc_layers(num_layers=num_layers,
                                   layer_sizes=layer_sizes,
                                   input_rois=input_rois,
                                   input_weights=input_weights,
                                   fusion_method=fusion_method,
                                   l2_weight_decay=l2_weight_decay,
                                   keep_prob=keep_prob,
                                   num_final_classes=num_final_classes,
                                   box_rep=box_rep,
                                   is_training=is_training)

    elif fusion_type == 'deep':
        cls_logits, offsets, angle_vectors = \
            _deep_fusion_fc_layers(num_layers=num_layers,
                                   layer_sizes=layer_sizes,
                                   input_rois=input_rois,
                                   input_weights=input_weights,
                                   fusion_method=fusion_method,
                                   l2_weight_decay=l2_weight_decay,
                                   keep_prob=keep_prob,
                                   num_final_classes=num_final_classes,
                                   box_rep=box_rep,
                                   is_training=is_training)
    else:
        raise ValueError('Invalid fusion type {}'.format(fusion_type))

    # Convert to end point dict
    end_points = slim.utils.convert_collection_to_dict(end_points_collection)
    return cls_logits, offsets, angle_vectors, end_points


def build_output_layers(tensor_in,
                        num_final_classes,
                        box_rep):
    """Builds flattened output layers

    Args:
        tensor_in: Input tensor
        num_final_classes: Final number of output classes, including
            'Background'
        box_rep: Box representation (e.g. 'box_3d', 'box_8c', 'box_4c')

    Returns:
        Output layers
    """

    # Classification
    cls_logits = slim.fully_connected(tensor_in,
                                      num_final_classes,
                                      activation_fn=None,
                                      scope='cls_out')

    # Offsets
    off_out_size = avod_fc_layer_utils.OFFSETS_OUTPUT_SIZE[box_rep]
    if off_out_size > 0:
        off_out = slim.fully_connected(tensor_in,
                                       off_out_size,
                                       activation_fn=None,
                                       scope='off_out')
    else:
        off_out = None

    # Angle Unit Vectors
    ang_out_size = avod_fc_layer_utils.ANG_VECS_OUTPUT_SIZE[box_rep]
    if ang_out_size > 0:
        ang_out = slim.fully_connected(tensor_in,
                                       ang_out_size,
                                       activation_fn=None,
                                       scope='ang_out')
    else:
        ang_out = None

    return cls_logits, off_out, ang_out


def _early_fusion_fc_layers(num_layers, layer_sizes,
                            input_rois, input_weights, fusion_method,
                            l2_weight_decay, keep_prob,
                            num_final_classes, box_rep,
                            is_training):

    if not num_layers == len(layer_sizes):
        raise ValueError('num_layers does not match length of layer_sizes')

    if l2_weight_decay > 0:
        weights_regularizer = slim.l2_regularizer(l2_weight_decay)
    else:
        weights_regularizer = None

    # Feature fusion
    fused_features = avod_fc_layer_utils.feature_fusion(fusion_method,
                                                        input_rois,
                                                        input_weights)

    # Flatten
    fc_drop = slim.flatten(fused_features)

    with slim.arg_scope(
            [slim.fully_connected],
            weights_regularizer=weights_regularizer):

        for layer_idx in range(num_layers):
            fc_name_idx = 6 + layer_idx

            # Use conv2d instead of fully_connected layers.
            fc_layer = slim.fully_connected(fc_drop, layer_sizes[layer_idx],
                                            scope='fc{}'.format(fc_name_idx))

            fc_drop = slim.dropout(
                fc_layer,
                keep_prob=keep_prob,
                is_training=is_training,
                scope='fc{}_drop'.format(fc_name_idx))

            fc_name_idx += 1

        output_layers = build_output_layers(fc_drop,
                                            num_final_classes,
                                            box_rep)
    return output_layers


def _late_fusion_fc_layers(num_layers, layer_sizes,
                           input_rois, input_weights, fusion_method,
                           l2_weight_decay, keep_prob,
                           num_final_classes, box_rep,
                           is_training):

    if l2_weight_decay > 0:
        weights_regularizer = slim.l2_regularizer(l2_weight_decay)
    else:
        weights_regularizer = None

    # Build fc layers, one branch per input
    num_branches = len(input_rois)
    branch_outputs = []

    with slim.arg_scope(
            [slim.fully_connected],
            weights_regularizer=weights_regularizer):
        for branch_idx in range(num_branches):

            # Branch feature ROIs
            branch_rois = input_rois[branch_idx]
            fc_drop = slim.flatten(branch_rois,
                                   scope='br{}_flatten'.format(branch_idx))

            for layer_idx in range(num_layers):
                fc_name_idx = 6 + layer_idx

                # Use conv2d instead of fully_connected layers.
                fc_layer = slim.fully_connected(
                    fc_drop, layer_sizes[layer_idx],
                    scope='br{}_fc{}'.format(branch_idx, fc_name_idx))

                fc_drop = slim.dropout(
                    fc_layer,
                    keep_prob=keep_prob,
                    is_training=is_training,
                    scope='br{}_fc{}_drop'.format(branch_idx, fc_name_idx))

            branch_outputs.append(fc_drop)

        # Feature fusion
        fused_features = avod_fc_layer_utils.feature_fusion(fusion_method,
                                                            branch_outputs,
                                                            input_weights)

        # Ouput layers
        output_layers = build_output_layers(fused_features,
                                            num_final_classes,
                                            box_rep)
    return output_layers


def _deep_fusion_fc_layers(num_layers, layer_sizes,
                           input_rois, input_weights, fusion_method,
                           l2_weight_decay, keep_prob,
                           num_final_classes, box_rep,
                           is_training):

    if l2_weight_decay > 0:
        weights_regularizer = slim.l2_regularizer(l2_weight_decay)
    else:
        weights_regularizer = None

    # Apply fusion
    fusion_layer = avod_fc_layer_utils.feature_fusion(fusion_method,
                                                      input_rois,
                                                      input_weights)
    fusion_layer = slim.flatten(fusion_layer, scope='flatten')

    with slim.arg_scope(
            [slim.fully_connected],
            weights_regularizer=weights_regularizer):
        # Build layers
        for layer_idx in range(num_layers):
            fc_name_idx = 6 + layer_idx

            all_branches = []
            for branch_idx in range(len(input_rois)):
                fc_layer = slim.fully_connected(
                    fusion_layer, layer_sizes[layer_idx],
                    scope='br{}_fc{}'.format(branch_idx, fc_name_idx))
                fc_drop = slim.dropout(
                    fc_layer,
                    keep_prob=keep_prob,
                    is_training=is_training,
                    scope='br{}_fc{}_drop'.format(branch_idx, fc_name_idx))

                all_branches.append(fc_drop)

            # Apply fusion
            fusion_layer = avod_fc_layer_utils.feature_fusion(fusion_method,
                                                              all_branches,
                                                              input_weights)

        # Ouput layers
        output_layers = build_output_layers(fusion_layer,
                                            num_final_classes,
                                            box_rep)
    return output_layers
