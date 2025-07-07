### How to use the index files for the experiments ?

The index files are named like "session_x.txt", where x indicates the session number. Each "session_t.txt" stores the 25 (5 classes and 5 shots per class) few-shot new class training images.
You may adopt the following steps to perform the experiments.

At session t (t>1), finetune the model trained at the previous session t, only using the images in session_t.txt.

For evaluating the model at session t, first joint all the encountered test sets as a single test set. Then test the current model using all the test images and compute the recognition accuracy. 
