import multiprocessing
import os

import dlib

options = dlib.shape_predictor_training_options()
options.tree_depth = 3
options.nu = 0.3
options.cascade_depth = 30
options.feature_pool_size = 800
options.num_test_splits = 20
options.oversampling_amount = 1
options.oversampling_translation_jitter = 0.4
options.be_verbose = True
options.num_threads = multiprocessing.cpu_count()

train_xml_path = os.path.abspath("./youtube_faces_train.xml")
predictor_dat = "./youtube_faces_68_points.dat"
dlib.train_shape_predictor(train_xml_path, predictor_dat, options)

print("\nTraining MAE: {}".format(
    dlib.test_shape_predictor(train_xml_path, predictor_dat)))

test_xml_path = os.path.abspath("./youtube_faces_test.xml")
print("Testing MAE: {}".format(
    dlib.test_shape_predictor(test_xml_path, predictor_dat)))
