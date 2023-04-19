# AnACor

**AnACor** is an accelerated absorption correction by tomographic reconstruction software written in Python and C. It's currently built at I23 beamline of Diamond Light Souce 

## Installation

This software is successfully installed at I23 beamline of Diamond Light Souce and current open installation process is incomplete. A way to install the nightly version is below:

On Linux, after downloading the software package from https://github.com/yishunlu-222/AnACor_public, the programme can be built with:
	```
	pip install ./AnACor_public/dist/AnACor-1.1-py3-none-any.whl
	```
	

## Software explanation

#### Basics in AnACor
![coordinates](https://github.com/yishunlu-222/AnACor_public/blob/main/img/documentation%20of%20codes-7.png)

- The 3D model in this software is a stack of segmented slices from tomographic reconstruction so it can be seen as a 3D cuboid. The six faces of 3D cuboid are labelled and will be used during the calculation. 


- The coordinates of this software is defined on the right of the image, which obeys the right-hand rule. The X-ray direction is opposite to the X-aixs of this software, while the axis of goniometer is against to the Z-axis. One thing to be careful about is to make sure the coordinates in this software suit the coordinates in the experiements.


- Vectors (both incident and diffracted vector) are resolved into theta(θ) and phi(φ) during the calculation given by 
	- X = r  *  *cos(θ)* *  *cos(φ)*
	- Y = r  *  *cos(θ)* *  *sin(φ)*
	- Z = r  *  *sin(θ)*

#### Basic ray tracing method

```
Path length calculation

	Input: S: the scattering vector of incident or diffracted ray
	Ω: the rotating matrix of the goniometer
	model : 3D segmented model from tomographic reconstruction
	crytal_voxels : crystal voxels in the 3D model
	
	Output: Path_len: incident/diffracted path lengths of different materials
	
	Function RayTracing(S,Ω,n,model):
	
		Ray ← S • Ω 
		face ← which_face( crytal_voxel, Ray  ) # ray exit face of the 3D model
		Path ← cal_coord_2( crytal_voxel, Ray, model  ) # Calculate and store the coordinates along the path 
		Path_len ← cal_num( Path ) # calculate the path lengths of different materials fromt the path coordinates
		
		return Path_len

```
Once it knows which face the Ray (vector) exits or incident, it knows the direction to step over ( `which_face` ). Then, it can store the coordinates where it passes by and it stops until it goes to the edge of the cube (`cal_coord_2` ). Finally, it gets a list of coordinates, which allow it to  compute the path lengths of different materials.


You might find `which_face`  and `cal_num` in the `class RayTracingBasic` in **RayTracing.py**  and `cal_coord_2` in the **Core_accelerated.py**

## Example

The examples are not ready at the moment, but you can find more information in the documentation https://yishunlu-222.github.io/anacor.github.io/

## Reference

When referencing this code, please cite our related paper:

Y. Li, R. Duman, J. Beilsten-Edmands, G. Winter, M. Basham, G. Evans, H. Kwong, K. Beis, A. Wagner, W. Armour, “Ray-tracing analytical absorption correction for X-ray crystallography based on tomographic reconstructions”.

## License
![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)
