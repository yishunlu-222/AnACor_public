# AnACor

**AnACor** is an accelerated absorption correction software for crystallography by written in Python, C and Cuda. It's currently built at the I23 beamline of Diamond Light Source. 

## Installation

This software is successfully installed at the I23 beamline of Diamond Light Souce, and the current open installation process is incomplete. A way to install the nightly version is below:

On Linux, after downloading the software package from https://github.com/yishunlu-222/AnACor_public, the programme can be built with the following:
	```
	pip install ./AnACor_public/dist/AnACor-1.1-py3-none-any.whl
	```
	

## Software explanation

#### Methodology 

In crystallography, scaling needs to be done to make sure the true intensities of the reflections can be obtained. Absorption correction, a part of scaling, is significant when the absorption is high.
The absorption correction method in popular Crystallography Scaling Software, such as DIALS (Beilsten-Edmands  et al., 2020), is spherical harmonics correction. This method relies on the high multiplicity of the dataset to construct an absorption surface that can estimate the relative absorption of each reflection. However, this can be less effective when the absorption is much higher, or the multiplicity of the dataset is lower.  The analytical absorption correction can address this circumstance and improve performance after combining with spherical harmonics correction, as shown in our paper (Lu  et al., 2023). The equation of analytical absorption correction for a reflection $n$ is below:


$$
\begin{align}
A_{h}^{(n)} = \frac{1}{V} \int_{V} e^{-\mu (L_1+L_2)} \, dV  \tag{1} \\
\end{align}
$$

where $V$ is the volume of the diffracting crystal, $L_1$ and $L_2$ are the incident and diffracting X-ray respectively, and  $\mu$ is the absorption coefficient of the material that the X-ray passes through. However, there are more materials in the real experiment, so for each crystal element $dV$, the absorption factor can be written as Eq. 2, where $L_m$ is the sum of the $L_1$ and $L_2$ for material $m$. To move from the continuous integral in Eq. 1 to a discrete equation that numerically solved on a computer, we treat the volume element $dV$ as a single crystal voxel $\Delta V$ from the tomographic reconstruction of our sample. Hence, the integral can be rewritten discretely as Eq. 3.

$$
\begin{align}
A_{h}^{(v)} =   \text{exp} \left[ -\sum\limits_{m=1}^{M} \mu_m L_{m}^{(v)} \right]  \tag{2} \\
A_{h}=\frac{1}{V}   \sum\limits_{n=1}^{N}A_{h}^{(n)} *\Delta V = \frac{1}{N}   \sum\limits_{n=1}^{N}A_{h}^{(n)}   \tag{3}
\end{align}
$$

where $N$ is the number of the crystal voxels in the 3D model that are bathed in the X-ray beam. When $N$ tends to be infinity and $\Delta V$ is approaching to be $0$, Eq. 3 can recover Eq. 1. Therefore, an absorption factor for reflection $h$ is determined by the average of the $N$ absorption factors, which are calculated from their corresponding crystal voxel $n$ by Beer-Lambda Law in Eq. 2.

#### Conventions in AnACor
![coordinates](https://github.com/yishunlu-222/AnACor_public/blob/main/img/documentation%20of%20codes-7.png)

- The 3D model in this software is a stack of segmented slices from tomographic reconstruction so that it can be seen as a 3D cuboid. The six faces of the 3D cuboid are labelled and will be used during the calculation. 

- The coordinates of this software are defined on the right of the image, which obeys the right-hand rule. The X-ray direction is opposite to the X-axis of this software, while the axis of the goniometer is against the Z-axis. One thing to be careful about is ensuring the coordinates in this software suit the coordinates in the experiments.

- Vectors (both incident and diffracted vector) are resolved into theta(θ) and phi(φ) during the calculation given by 
	- X = r  *  *cos(θ)* *  *cos(φ)*
	- Y = r  *  *sin(θ)*
	- Z = r  *  *cos(θ)* *  *sin(φ)*

#### Algorithm: Basic ray tracing method

```
Path length calculation

	Input: S: the scattering vector of incident or diffracted ray
	Ω: the rotating matrix of the goniometer
	model: 3D segmented model from tomographic reconstruction
	crytal_voxels: crystal voxels in the 3D model
	
	Output: Path_len: incident/diffracted path lengths of different materials
	
	Function RayTracing(S,Ω,n,model):
	
		Ray ← S • Ω 
		face ← cube_face( crytal_voxel, Ray  ) # ray exit face of the 3D model
		Path ← cal_coord_2( crytal_voxel, Ray, model  ) # Calculate and store the coordinates along the path 
		Path_len ← cal_num( Path ) # Calculate the path lengths of different materials from the path coordinates
		
		return Path_len

```
- Once it knows which face the Ray (vector) exits or incident by `cube_face`, it knows the direction to carry on.  https://github.com/yishunlu-222/AnACor_public/blob/0ae41e62e7c1f89412b6bc1a6f6836747ae20114/AnACor/utils/utils_rt.py#L1154
- Then, it can step along the direction and store the coordinates where it passes by, and it stops until it goes to the edge of the cube (`cal_coord` ) or encounters air/vacuum.  For example, the exiting face is the *Front face* with the crystal voxel of `(z,y,x)`, and the next step will have a coordinate of  `(z+dz,y+dz,x+dz)`. For every step, the x will have one increment (`dx=1`), so y, z will have increments `dy= tan(θ)/cos(φ)` and `dz = tan(φ)` according to the coordinate system in this software. https://github.com/yishunlu-222/AnACor_public/blob/3b670a742b23e096703acd23d952a3465cdb2518/AnACor/utils/utils_rt.py#L283
- After this, it gets a list of coordinates, which allows it to  compute the path lengths $L_m$ of different materials.  https://github.com/yishunlu-222/AnACor_public/blob/3b670a742b23e096703acd23d952a3465cdb2518/AnACor/utils/utils_rt.py#L959
- Finally, calculate the absorption factor of this crystal voxel according to Equation (2) (`cal_rate`). https://github.com/yishunlu-222/AnACor_public/blob/3b670a742b23e096703acd23d952a3465cdb2518/AnACor/utils/utils_rt.py#L1095



## Example
### Ompk36 GD at 3.5keV testcase

#### Prerequisite

dials version:  3.15.1

Python numba version: 1.23.2

Python Numpy version:  0.56.2

#### Implementation

Download the necessary dataset from https://drive.google.com/drive/folders/1VBD_5aJXhmAWNi8B3P6mhpv98aJd4dMa?usp=drive_link. Then 

```
python -u ./AnACor/AnACor/main_lite.py  --expt-path "./16010_ompk_10_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt16010_0.json"  --refl-path "./16010_ompk_10_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl16010_0.json"  --liac 0.01208 --crac 0.01053 --loac 0.00931 --buac 0.0322 --model-storepath "./16010_tomobar_cropped_f.npy" --store-dir "./test" --dataset 16010 --sampling-num 5000 --by-c True --num-workers 1

```
If the calculation is finished, then source or module load DIALS environment
Then scaling within DIALS
```
dials.python ./AnACor/AnACor/utils/stacking.py --save-dir ./test --dataset 16010
```

```
dials.python ./AnACor/AnACor/utils/into_flex.py --save-number test --refl-filename ./16010_ompk_10_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.refl   --dataset 16010
```

Perform AAC strategy:
```
dials.scale anacor_test.refl 16010_ompk_10_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt physical.absorption_correction=False physical.analytical_correction=True
```

Perform ACSH strategy:
```
dials.scale anacor_test.refl 16010_ompk_10_3p5keV_AUTOMATIC_DEFAULT_SAD_SWEEP1.expt physical.absorption_level=high physical.analytical_correction=True
```

You can also find more information in the documentation https://yishunlu-222.github.io/anacor.github.io/

## Reference

When referencing this code, please cite our related paper:

[1] Y. Lu, R. Duman, J. Beilsten-Edmands, G. Winter, M. Basham, G. Evans, H. Kwong, K. Beis, A. Wagner, W. Armour, “Ray-tracing analytical absorption correction for X-ray crystallography based on tomographic reconstructions”.

[2] Beilsten-Edmands, J., Winter, G., Gildea, R., Parkhurst, J., Waterman, D. & Evans, G. (2020).  
Acta Cryst. D76, 385–399


## License
![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)
