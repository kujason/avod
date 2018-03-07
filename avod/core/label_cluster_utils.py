import os
import sys

import numpy as np
from sklearn.cluster import KMeans

from wavedata.tools.obj_detection import obj_utils

import avod


class LabelClusterUtils:
    def __init__(self, dataset):

        self._dataset = dataset

        self.cluster_split = dataset.cluster_split

        self.data_dir = avod.root_dir() + "/data/label_clusters"
        self.clusters = []
        self.std_devs = []

    @staticmethod
    def _filter_labels_by_class(obj_labels, classes):
        """
        Splits ground truth labels based on provided classes

        Args:
            obj_labels: ObjectLabel list for an image
            classes: Classes to save

        Returns:
            A list of (l, w, h) of the objects, for each class
        """
        filtered = [[] for _ in range(len(classes))]

        for obj_label in obj_labels:
            if obj_label.type in classes:
                class_idx = classes.index(obj_label.type)

                # l, w, h
                obj_l = obj_label.l
                obj_w = obj_label.w
                obj_h = obj_label.h
                filtered[class_idx].append([obj_l, obj_w, obj_h])

        return filtered

    def _get_cluster_file_path(self, dataset, cls, num_clusters):
        """
        Returns a unique file path for a text file based on
        the dataset name, split, object class, and number of clusters.
        The file path will look like:
            avod/data/<dataset_name>/<data_split>/<class>_<n_clusters>


        Args:
            dataset: Dataset object
            cls: str, Object class
            num_clusters: number of clusters for the class

        Returns: str Unique file path to text file
        """

        file_path = "{}/{}/{}/".format(self.data_dir,
                                       dataset.name,
                                       dataset.cluster_split,
                                       dataset.data_split)
        file_path += '{}_{}.txt'.format(cls, num_clusters)

        return file_path

    def _write_clusters_to_file(self, file_path, clusters, std_devs):
        """
        Writes cluster information to a text file

        Args:
            file_path: path to text file
            clusters: clusters to write
            std_devs: standard deviations to write
        """

        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        new_file = open(file_path, 'w+')

        all_data = np.vstack([clusters,
                              std_devs])

        np.savetxt(file_path, all_data, fmt='%.3f')

        new_file.close()

    def _read_clusters_from_file(self, dataset, cls, num_clusters):
        """
        Reads cluster information from a text file

        Args:
            dataset: Dataset, used to find the saved text file
            cls: class to read cluster information for
            num_clusters: number of clusters
        """

        file_path = self._get_cluster_file_path(dataset, cls, num_clusters)
        if os.path.isfile(file_path):
            cluster_file = open(file_path, 'r')

            data = np.loadtxt(file_path)

            clusters = np.array(data[0:num_clusters])
            std_devs = np.array(data[num_clusters:])

            cluster_file.close()

            return clusters, std_devs

        return None, None

    def _flatten_data(self, data):
        """
        Reshapes a list of cluster data from a list of arrays into an N x 3
        ndarray, for example a list of [2, 1, 2] clusters:
        [
         [[l, w, h], [l, w, h]],
         [[l, w, h]],
         [[l, w, h], [l, w, h]]
        ]
        becomes a 5 x 3 ndarray:
        [[l, w, h], [l, w, h], [l, w, h], [l, w, h], [l, w, h]]

        Args:
            data: a list of clusters separated by class

        Returns:
            The data reshaped into an N x 3 ndarray
        """
        all_data = []
        for class_idx in range(len(data)):
            data_reshaped = np.asarray(data[class_idx]).reshape((-1, 3))
            all_data.extend(data_reshaped)

        return np.asarray(all_data)

    def get_clusters(self):
        """
        Calculates clusters for each class

        Returns:
            all_clusters: list of clusters for each class
            all_std_devs: list of cluster standard deviations for each class
        """

        classes = self._dataset.classes
        num_clusters = self._dataset.num_clusters

        all_clusters = [[] for _ in range(len(classes))]
        all_std_devs = [[] for _ in range(len(classes))]

        classes_not_loaded = []

        # Try to read from file first
        for class_idx in range(len(classes)):
            clusters, std_devs = self._read_clusters_from_file(
                self._dataset, classes[class_idx], num_clusters[class_idx])

            if clusters is not None:
                all_clusters[class_idx].extend(np.asarray(clusters))
                all_std_devs[class_idx].extend(np.asarray(std_devs))
            else:
                classes_not_loaded.append(class_idx)

        # Return the data flattened into N x 3 arrays
        if len(classes_not_loaded) == 0:
            return all_clusters, all_std_devs

        # Calculate the remaining clusters
        # Load labels corresponding to the sample list for clustering
        sample_list = self._dataset.load_sample_names(self.cluster_split)
        all_labels = [[] for _ in range(len(classes))]

        num_samples = len(sample_list)
        for sample_idx in range(num_samples):

            sys.stdout.write("\rClustering labels {} / {}".format(
                sample_idx + 1, num_samples))
            sys.stdout.flush()

            sample_name = sample_list[sample_idx]
            img_idx = int(sample_name)

            obj_labels = obj_utils.read_labels(self._dataset.label_dir,
                                               img_idx)
            filtered_labels = LabelClusterUtils._filter_labels_by_class(
                obj_labels, self._dataset.classes)

            for class_idx in range(len(classes)):
                all_labels[class_idx].extend(filtered_labels[class_idx])

        print("\nFinished reading labels, clustering data...\n")

        # Cluster
        for class_idx in classes_not_loaded:
            labels_for_class = np.array(all_labels[class_idx])

            n_clusters_for_class = num_clusters[class_idx]
            if len(labels_for_class) < n_clusters_for_class:
                raise ValueError(
                    "Number of samples is less than number of clusters "
                    "{} < {}".format(len(labels_for_class),
                                     n_clusters_for_class))

            k_means = KMeans(n_clusters=n_clusters_for_class,
                             random_state=0).fit(labels_for_class)

            clusters_for_class = []
            std_devs_for_class = []

            for cluster_idx in range(len(k_means.cluster_centers_)):
                cluster_centre = k_means.cluster_centers_[cluster_idx]

                labels_in_cluster = labels_for_class[
                    k_means.labels_ == cluster_idx]

                # Calculate std. dev
                std_dev = np.std(labels_in_cluster, axis=0)

                formatted_cluster = [float('%.3f' % value)
                                     for value in cluster_centre]
                formatted_std_dev = [float('%.3f' % value)
                                     for value in std_dev]

                clusters_for_class.append(formatted_cluster)
                std_devs_for_class.append(formatted_std_dev)

            # Write to files
            file_path = self._get_cluster_file_path(self._dataset,
                                                    classes[class_idx],
                                                    num_clusters[class_idx])

            self._write_clusters_to_file(file_path, clusters_for_class,
                                         std_devs_for_class)

            # Add to full list
            all_clusters[class_idx].extend(np.asarray(clusters_for_class))
            all_std_devs[class_idx].extend(np.asarray(std_devs_for_class))

        # Return the data flattened into N x 3 arrays
        return all_clusters, all_std_devs
