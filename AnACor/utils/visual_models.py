import vtk
import numpy as np
from vtk.util import numpy_support

filename='/home/yishun/projectcode/anacor_testcase/13295_new/13295_save_data/13295_.npy'
filename='/home/yishun/projectcode/anacor_testcase/16846_test_save_data/test_save_data/test_.npy'
filename='/home/yishun/projectcode/anacor_testcase/13304_test_save_data/13304_save_data/13304_.npy'
# filename = '/home/yishun/projectcode/dials_develop/cld_1704_22_3keV.npy'
data_array=np.load(filename)
image_data = vtk.vtkImageData()
image_data.SetDimensions(data_array.shape)
image_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

# Fill the vtkImageData object with the numpy array data
for z in range(data_array.shape[2]):
    print(f"{z}/{data_array.shape[2]}")
    for y in range(data_array.shape[1]):
        for x in range(data_array.shape[0]):
            image_data.SetScalarComponentFromDouble(x, y, z, 0, data_array[x, y, z])

# Create a lookup table to map scalar values to colors
color_transfer_function = vtk.vtkColorTransferFunction()
color_transfer_function.AddRGBPoint(0, 0.0, 0.0, 0.0)  # Black for 0
color_transfer_function.AddRGBPoint(1, 0.0, 0.0, 1.0)  # Blue for 1
color_transfer_function.AddRGBPoint(2, 0.0, 1.0, 0.0)  # Green for 2
color_transfer_function.AddRGBPoint(3, 1.0, 0.0, 0.0)  # Red for 3

# Opacity transfer function
opacity_transfer_function = vtk.vtkPiecewiseFunction()
opacity_transfer_function.AddPoint(0, 0.0)  # Fully transparent for 0
opacity_transfer_function.AddPoint(1, 0.1)  # 30% opacity for 1
opacity_transfer_function.AddPoint(2, 1.0)  # Fully opaque for 2
opacity_transfer_function.AddPoint(3, 1.0)  # Fully opaque for 3

# The volume property describes how the data will look
volume_property = vtk.vtkVolumeProperty()
volume_property.SetColor(color_transfer_function)
volume_property.SetScalarOpacity(opacity_transfer_function)

# The rest of the code for volume rendering remains the same
volume_mapper = vtk.vtkSmartVolumeMapper()
volume_mapper.SetInputData(image_data)

volume = vtk.vtkVolume()
volume.SetMapper(volume_mapper)
volume.SetProperty(volume_property)

renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
renderer.AddVolume(volume)
renderer.SetBackground(0, 0, 0)

render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

render_window.Render()
render_window_interactor.Start()