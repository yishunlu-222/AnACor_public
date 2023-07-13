# AnACor

**AnACor** is an accelerated absorption correction software for crystallography by written in Python, C and Cuda. It's currently built at I23 beamline of Diamond Light Souce 

## Installation

This software is successfully installed at I23 beamline of Diamond Light Souce and current open installation process is incomplete. A way to install the nightly version is below:

On Linux, after downloading the software package from https://github.com/yishunlu-222/AnACor_public, the programme can be built with:
	```
	pip install ./AnACor_public/dist/AnACor-1.1-py3-none-any.whl
	```
	

## Software explanation

#### Methodology 

In crystallography, scaling needs to be done to make sure the true intensities of the reflections can be obtained. Absorption correction, a part of scaling, is significant when the absorption is high.
The absorption correction method in popular Crystallography Scaling Software, such as DIALS (Beilsten-Edmands  et al., 2020) , is spherical harmonics correction. This method relies on the high multiplicity of the dataset to construct an absorption surface that can estimate the relative absorption of each reflection. However, this can be less effective when the absorption is much higher, or the multiplicity of the dataset is lower. The analytical absorption correction can address this circumstance and improve performance after combining with spherical harmonics correction, as shown in our paper (Lu  et al., 2020). The equations of analytical absorption correction are below:

$$
\begin{align}
A_{hkl} = \frac{1}{V} \int_{V} e^{-\mu (L_1+L_2)} \, dV  \tag{1} \\
A_{hkl}^{(n)} =   \text{exp} \left[ -\sum\limits_{m=1}^{M} \mu_m L_{m}^{(n)} \right]  \tag{2} \\
A_{hkl}= \frac{1}{N}   \sum\limits_{n=1}^{N}A_{hkl}^{(n)}  \tag{3}
\end{align}
$$

where $V$ is the volume of the diffracting crystal, $N$ is the number of crystal voxels, $M$ represents the number of different materials, $L_1$ and $L_2$ are the incident and diffracting X-ray resepectively, and $\mu$ is the absorption coefficient of the materials. They show that an absorption factor for a reflection is the average of the absorption factors of all the crystal volume. For each absorption factor, it is calculated by the Beer-Lambda law in Equation (2).  In this project, the sample is tomographically reconstructed so an absorption factor for a reflection is determined by the average of the absorption factors of crystal voxels shown in Equation (3).

#### Basics in AnACor
![coordinates](https://github.com/yishunlu-222/AnACor_public/blob/main/img/documentation%20of%20codes-7.png)

- The 3D model in this software is a stack of segmented slices from tomographic reconstruction so it can be seen as a 3D cuboid. The six faces of 3D cuboid are labelled and will be used during the calculation. 

- The coordinates of this software is defined on the right of the image, which obeys the right-hand rule. The X-ray direction is opposite to the X-aixs of this software, while the axis of goniometer is against to the Z-axis. One thing to be careful about is to make sure the coordinates in this software suit the coordinates in the experiements.

- Vectors (both incident and diffracted vector) are resolved into theta(θ) and phi(φ) during the calculation given by 
	- X = r  *  *cos(θ)* *  *cos(φ)*
	- Y = r  *  *sin(θ)*
	- Z = r  *  *cos(θ)* *  *sin(φ)*

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
- Once it knows which face the Ray (vector) exits or incident, it knows the direction to step over ( `which_face` ).
- Then, it can step along the direction and store the coordinates where it passes by and it stops until it goes to the edge of the cube (`cal_coord_2` ).  For example, the exiting face is the *Front face* with the crystal voxel of `(z,y,x)`, and the next step will have coordinate of  `(z+dz,y+dz,x+dz)`. For every step, the x will have one increment (`dx=1`), so, y, z will have increments `dy= tan(θ)/cos(φ)`and `dz = tan(φ)` according to the coordinate system in this software. 
- Finally, it gets a list of coordinates, which allow it to  compute the path lengths of different materials.

You might find `which_face`  and `cal_num` in the `class RayTracingBasic` in **RayTracing.py**  and `cal_coord_2` in the **Core_accelerated.py**

### Slice Sampling

### Gridding approximation 

### GPU (Cuda)

## Example

The examples are not ready at the moment, but you can find more information in the documentation https://yishunlu-222.github.io/anacor.github.io/

## Reference

When referencing this code, please cite our related paper:

[1] Y. Lu, R. Duman, J. Beilsten-Edmands, G. Winter, M. Basham, G. Evans, H. Kwong, K. Beis, A. Wagner, W. Armour, “Ray-tracing analytical absorption correction for X-ray crystallography based on tomographic reconstructions”.
[2]Beilsten-Edmands, J., Winter, G., Gildea, R., Parkhurst, J., Waterman, D. & Evans, G. (2020).  
Acta Cryst. D76, 385–399


## License
![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)
