# AnACor

**AnACor** is an accelerated absorption correction by tomographic reconstruction software written in Python and C. It's currently built at I23 beamline of Diamond Light Souce 

## Installation

This software is successfully installed at I23 beamline of Diamond Light Souce and current open installation process is incomplete. A way to install the nightly version is below:

On Linux, after downloading the software package from https://github.com/yishunlu-222/AnACor_public, the programme can be built with:
	```
	pip install ./AnACor_public/dist/AnACor-1.1-py3-none-any.whl
	```
	

## Algorithm explanation

#### Basic ray tracing method

```
Path length calculation

	Input: S: the scattering vector of incident or diffracted ray
	Ω: the rotating matrix of the goniometer
	n: the positional vector of the crystal voxel to calculate path length by the ray-tracing
	method
	model : 3D segmented model from tomographic reconstruction
	crytal_voxels : crystal voxels in the 3D model
	
	Output: Path_len: incident/diffracted path lengths of different materials
	
	Function RayTracing(S,Ω,n,model):
	
		Ray ← S • Ω • n
		face ← which_face( crytal_voxel, Ray  ) # ray exit face of the 3D model
		Path ← cal_coord_2( crytal_voxel, Ray, model  ) # Store the coordinates along the path 
		Path_len ← cal_num( Path ) # calculate the path lengths of different materials fromt the path coordinates
		
		return Path_len

```
You might find `which_face`  and `cal_num` in the `class RayTracingBasic` in **RayTracing.py**  and `cal_coord_2` in the **Core_accelerated.py**
## Reference

When referencing this code, please cite our related paper:

Y. Li, R. Duman, J. Beilsten-Edmands, G. Winter, M. Basham, G. Evans, H. Kwong, K. Beis, A. Wagner, W. Armour, “Ray-tracing analytical absorption correction for X-ray crystallography based on tomographic reconstructions”.

## License
![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)
