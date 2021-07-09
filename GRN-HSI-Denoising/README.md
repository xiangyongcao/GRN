# GRN-HSI-Denoising

This is the code for the TGRS paper "Deep Spatial-Spectral Global Reasoning Network for Hyperspectral Image Denoising".

If you use this code, pleae cite the following papers in your work.

@article{cao2021deep,\\
	title={Deep Spatial-Spectral Global Reasoning Network for Hyperspectral Image Denoising},\\
	author={Cao, Xiangyong and Fu, Xueyang and Xu, Chen and Meng, Deyu},\\
	journal={IEEE Transactions on Geoscience and Remote Sensing},\\
	year={DOI: 10.1109/TGRS.2020.3016820},\\
	publisher={IEEE}\\
}

## Follow these steps to run the code:

1. Download clean raw datasets and divide them into train, validate, and test. Then, put them into './data/datasets/ raw_clearn_ICVL_train/',
'./data/datasets/ raw_clearn_ICVL_validate/', and './data/datasets/ raw_clearn_ICVL_test/' files separately. 

2. "run demo_add_iidgaussian_blind.m" to add i.i.d Gaussian noise to train, validate, and test datasets.

3. "run demo_add_iidgaussian_blind.m" to add i.i.d Gaussian noise to train, validate, and test datasets.

4. "run generate_train.m" to generate the ".h5" file of training patch.

5. "run generate_validate.m" to generate the ".h5" file of validate patch.

6. "run train.py" to train the GRN network

7. "run test.py" to test the GRN network in the test datasets.

## Contact:
 If you have any question, welcome to contact me (caoxiangyong45@gmail.com  /  caoxiangyong@mail.xjtu.edu.cn).
