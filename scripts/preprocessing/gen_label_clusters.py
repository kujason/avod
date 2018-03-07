from avod.builders.dataset_builder import DatasetBuilder


def main(dataset=None):
    if not dataset:
        dataset = DatasetBuilder.build_kitti_dataset(
            DatasetBuilder.KITTI_TRAIN)

    label_cluster_utils = dataset.kitti_utils.label_cluster_utils

    print("Generating clusters in {}/{}".format(
        label_cluster_utils.data_dir, dataset.data_split))
    clusters, std_devs = dataset.get_cluster_info()

    print("Clusters generated")
    print("classes: {}".format(dataset.classes))
    print("num_clusters: {}".format(dataset.num_clusters))
    print("all_clusters:\n {}".format(clusters))
    print("all_std_devs:\n {}".format(std_devs))


if __name__ == '__main__':
    main()
