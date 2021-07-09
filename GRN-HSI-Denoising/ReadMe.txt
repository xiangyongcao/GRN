This is the code for the TGRS paper "Deep Spatial-Spectral Global Reasoning Network for Hyperspectral Image Denoising".

Follow these steps to run the code:

step 1: download clean raw datasets and divide them into train, validate, and test. Then, put them into './data/datasets/ raw_clearn_ICVL_train/',
'./data/datasets/ raw_clearn_ICVL_validate/', and './data/datasets/ raw_clearn_ICVL_test/' files separately. 

step 2: "run readfns.m" to read all the mat file name of in the above three files of step 1. 

step 3: "run demo_add_iidgaussian_blind.m" to add i.i.d Gaussian noise to train, validate, and test datasets.

step 4: "run generate_train.m" to generate the ".h5" file of training patch.

step 5: "run generate_validate.m" to generate the ".h5" file of validate patch.

step 6: "run train.py" to train the GRN network

step 7: "run test.py" to test the GRN network in the test datasets.


If you have any question, please contact me (caoxiangyong@mail.xjtu.edu.cn).
