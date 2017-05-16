#!/bin/bash

if [ 1 -eq 1 ]; then 
    echo 'runing script which will convert .obj files (polygon meshes) to voxelized model'
    cd meshes2voxels_utils 
    matlab -nodisplay -nosplash -r 'convert_mesh2voxel_folder;exit;' 
    cd ..
fi 
