# synthocean

## 📄 Description:

model2SWOT is an approach designed to interpolate model Sea Surface Height (SSH) data onto Surface Water and Ocean Topography (SWOT) grids. Given a single daily model data, the script interpolates the SSH data onto a given SWOT swath.

### Required Inputs:

- Model netCDF files or Zarr store containing SSH data.
- Model domain / mask netCDF file or Zarr store containing the longitude and latitude coordinates.
- SWOT grid netCDF file containing the target longitude/latitude grid. It is recommended to use the "Expert" version of these files.
- Interpolator: The script offers two interpolation methods. Scipy interpolation method and Pyinterp interpolation method. Only one method can be chosen at a time, depending on user preference.

### Optional Inputs:

- `latitude_var_name`: The name of the latitude variable in the model dataset (e.g., "lat" or "latitude").
- `longitude_var_name`: The name of the longitude variable in the model dataset (e.g., "lon" or "longitude").
- `time_name`: The name of the time variable in the model dataset (e.g., "time" or "time_counter").
- `model_ssh_var`: The name of the SSH variable in the model dataset (e.g., "ssh" or "sossheig").

By default, these variables are assumed to be named:

- latitude for `latitude_var_name`
- longitude for `longitude_var_name`
- time_counter for `time_name`
- ssh for `model_ssh_var`

Users should make sure to provide the correct variable names if their dataset uses different names.

## 🚀 Getting Started:

### Installation:

To install the **synthocean** package, users should first clone this GitHub repository as follows:

```bash
git clone git@github.com:Amine-ouhechou/synthocean.git
```

Once users have successfully cloned the repository, the `model2swot` command line interface (CLI) can be installed using pip (editable install) as follows:

```bash
cd synthocean
pip install -e .
```

### Usage:

To use the tool, users can run the `model2swot` command followed by the required arguments to run the interpolation of model outputs onto a given SWOT swath:

```bash
model2swot -m path_to_your_model_file -k path_to_your_model_mask_file -s path_to_swot_data_file -o path_to_output_file -i interpolator --model_lat_var latitude_var_name --model_lon_var longitude_var_name --model_time_var time_name --model_ssh_var the_model_ssh_variable_name
```

**Note:** When specifying a model SSH file including mutliple time-slices (e.g., daily-mean SSH fields stored in a monthly netCDF file), the nearest time-slice to the average time of the given SWOT swath will be used. The selected model time-slice is reported to users in the log.

#### Required Arguments:

| Flag | Name | Description |
|---|---|---|
| -m | Model SSH file | Path to model file containing the SSH data. |
| -k | Model Mask file | Path to model mask / domain file containing longitude-latitude data. |
| -s | SWOT grid file | Path to SWOT swath file containing longitude-latitude grid to interpolate onto. |
| -o | Output file | Path to the output file where results will be saved as netCDF file. |
| -i | Interpolation | Interpolation method (choose between "scipy_interpolation" or "pyinterp_interpolation"). |

#### Optional Arguments:

| Flag  | Description |
|---|---|
| --model_lat_var | Model latitude variable name |
| --model_lon_var | Model longitude variable name |
| --model_time_var | Model time variable name |
| --model_ssh_var | Model SSH variable name |

## ⚡ Notes
- If the model mask and data are in the same file, users should provide the path to this file twice — once for the data and once for the mask argument. The file will only be read once.
- Model input files **must be either a netCDF files or Zarr stores** (`.nc` or `.zarr`).
- Output file **must be a NetCDF file** (`.nc`).
- SWOT gridded datasets are provided [here](https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/catalog/meomopendap/extract/MEOM/SWOT-geometry/catalog.html)

The tool is currently tested with **eORCA25** and **eNATL60** data in **NetCDF** and **Zarr** format.

## 🐛 Issues and Contributions

If you encounter any issues or would like to contribute, please feel free to [open an issue here](https://github.com/Amine-ouhechou/synthocean/issues).
