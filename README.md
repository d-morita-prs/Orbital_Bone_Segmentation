# Orbital_Bone_Segmentation
This model segments facial bones from CT images into both sides of orbital bone.　The segmentation model was built from convolutional neural network using a 2D U-net architectur and was trained on manual segmentations of 115 cases.

## How to Use
This model assumes the use of a "3D slicer". We will not discuss the how to use 3D slicer.

1. Download the trained model from the URL in the file "weight_115cpu.txt".
2. Create a directory structure as shown in the "directory" file. 
3. "images" folder with the mhd files you want to infer. 
4. Run "Inference.py". 
5. In the "output" folder, a file named "_label.mhd" will be output.

You can check the inference result by loading the original mhd file and the corresponding "_label.mhd" file into 3D slicer.
