main: evaluate_object_3d_offline.cpp
	g++ -o evaluate_object_3d_offline  evaluate_object_3d_offline.cpp -lboost_system -lboost_filesystem

eval_05_iou: evaluate_object_3d_offline_05_iou.cpp
	g++ -o evaluate_object_3d_offline_05_iou  evaluate_object_3d_offline_05_iou.cpp -lboost_system -lboost_filesystem

.PHONY: clean

clean:
	rm -f evaluate_object_3d_offline
	rm -f evaluate_object_3d_offline_05_iou
