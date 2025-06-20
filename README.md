# synthocean

**The repository is currently in progress!**  

Description:

Interp_model2SWOT is an approach designed to interpolate model SSH (Sea Surface Height) data onto the SWOT (Surface Water and Ocean Topography) grid. Given a single daily model data, the script interpolates the SSH data onto a given SWOT swath.

Main Requered Inputs:

- model files containing SSH data.
- Model mask containing the longitude and latitude coordinates.
- SWOT data file containing the target longitude/latitude grid. It is recommended to use the "Expert" version of these files.
- Interpolator: The script offers two interpolation methods. Scipy interpolation method and Pyinterp interpolation method. Only one method can be chosen at a time, depending on user preference.

Other Inputs:

- latitude_var_name: The name of the latitude variable in the model dataset (e.g., "lat" or "latitude").
- longitude_var_name: The name of the longitude variable in the model dataset (e.g., "lon" or "longitude").
- time_name: The name of the time variable in the model dataset (e.g., "time" or "time_counter").
- model_ssh_var: The name of the SSH variable in the model dataset (e.g., "ssh" or "sossheig").

By default, these variables are assumed to be named:

- latitude for latitude_var_name
- longitude for longitude_var_name
- time_counter for time_name
- ssh for model_ssh_var
Users should make sure to provide the correct variable names if their dataset uses different names.

## Usage:

To use the tool, the user can clone the repository. Once the repository is cloned, the user can run the code in the `Interp_model2SWOT.py` file, by executing the command line from a terminal using the following command (after navigating to the directory containing Interp_model2SWOT.py) :

```bash
./Interp_model2SWOT.py -m path_to_your_model_file -k path_to_your_model_mask_file -s path_to_swot_data_file -o path_to_output_file -i interpolator --model-lat-var latitude_var_name --model-lon-var longitude_var_name --model-time-var time_name --model_ssh_var the_model_ssh_variable_name
```
Where:
-m is the path to your model file containing the SSH data.
-k is the path to your model mask file.
-s is the path to the directory containing the SWOT data files.
-o is the path to the output directory where results will be saved.
-i specifies the interpolation method (choose between scipy_interpolation or pyinterp_interpolation).

The user can import the Interp_model2SWOT module into a Python script as follows (refer to the example notebook (In progress...) in the repository for further guidance)

## ⚡ Notes
- If the model mask and data are in the same file, you need to provide the path to this file twice — once for the data and once for the mask argument.
- Input file (model and mask) **must be either a NetCDF or zarr file** (`.nc, .zarr`).
- The output file **must be a NetCDF file** (`.nc`).

The tool is currently tested with **eORCA25** and **eNATL60** data in **NetCDF** and **Zarr** format.

The data needed for testing will soon be provided via an S3 endpoint.
