# kitti_native_eval

`evaluate_object_3d_offline.cpp`evaluates your KITTI detection locally on your own computer using your validation data selected from KITTI training dataset, with the following metrics:

- Average Precision In 2D Image Frame (AP)
- oriented overlap on image (AOS)
- Average Precision In BEV (AP)
- Average Precision In 3D (AP)

1. Install:
```
sudo apt-get install gnuplot gnuplot5

cd /kitti_native_eval

make
```

2. Copy the results folder into this folder. Each step should contain a 'data' folder.

3. Run the evaluation on all steps in the folder, for example:
```
./all_eval.sh 0.5
```
---
Alternatively, you can run the evaluation using the following command on a single step:
```
./evaluate_object_3d_offline groundtruth_dir result_dir
```

- Place the results folder in data folder and use /kitti_native_eval as results_dir
- Use ~/Kitti/object/training/label_2  as your groundtruth_dir

---

Note that you don't have to detect over all KITTI training data. The evaluator only evaluates samples whose result files exist.

- Results will appear per class in terminal for easy, medium and difficult data.
- Precision-Recall Curves will be generated and saved to 'plot' dir.
