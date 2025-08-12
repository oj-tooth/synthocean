"""
model2SWOT.py

Description:
This module defines the CLI for the model2SWOT
interpolation tool.

"""
import os
import sys
import logging
import argparse
import pyinterp
import numpy as np
import xarray as xr
from scipy import interpolate
from inpoly.inpoly2 import inpoly2

# Define permissable interpolators:
# TODO: xesmf inteprolator + saving weights to .nc file.
ALLOWED_INTERPOLATORS = ['scipy_interpolator', 'pyinterp_interpolator']

def initialise_logging():
    """
    Initialise model2SWOT logging configuration.
    """
    logging.basicConfig(
        stream=sys.stdout,
        format=">>>  Model2SWOT  <<<  | %(levelname)10s | %(asctime)s | %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_parser():
    """
    Create argument parser.
    """
    parser = argparse.ArgumentParser(description="A tool to process and interpolate ocean model to SWOT satellite swath",
        formatter_class=argparse.RawTextHelpFormatter # for multiline help messages
                                    )
    parser.add_argument("-m", "--model_file", required=True, help="Path of the input model file (NETCDF or zarr)")
    parser.add_argument("-k", "--mask_file", required=True, help="Path of the input mask file (NETCDF or zarr)")
    parser.add_argument("-s", "--swot_file", required=True, help="Path of the input SWOT file (NETCDF or zarr)")
    parser.add_argument("-o", "--output_file", required=True, help="Path of the output NETCDF file")

    parser.add_argument("-i", "--interpolator", type=validate_interpolator, required=True,
                        help=f"The interpolation method to use. "
                             f"Supported methods are: {', '.join(ALLOWED_INTERPOLATORS)}.\n"
                             f"Choose one from the list based on your needs.")
    
    # Arguments for model's (lat/lon) variable names:
    parser.add_argument("--model_lat_var", default="latitude",
                        help="Name of the latitude variable in the model NetCDF file (default: latitude).")
    parser.add_argument("--model_lon_var", default="longitude",
                        help="Name of the longitude variable in the model NetCDF file (default: longitude).")
    parser.add_argument("--model_time_var", default="time_counter",
                        help="Name of the time variable in the model NetCDF file (default: time_counter).")
    parser.add_argument("--model_ssh_var", default="ssh", 
                        help="Name of the variable in the model file to interpolate (e.g., 'ssh', 'sossheig', ...)")

    return parser

def read_netcdf_files(
    file1:str,
    file2:str,
    file3:str,
    time_name:str="time_counter",
):
    """
    Reads three NetCDF files and returns the datasets.
    
    Parameters
    ----------
    file1 : str
        Path to the first NetCDF file (Model).
    file2 : str
        Path to the second NetCDF file (Mask for model).
    file3 : str
        Path to the third NetCDF file (SWOT data).
    time_name : str, optional
        Name of the time dimension (default is "time_counter").

    Returns
    -------
    tuple
        A tuple containing three xarray datasets.
    """
    
    # File extensions ...
    extension1 = os.path.splitext(file1)[1]
    extension2 = os.path.splitext(file2)[1]

    # Open Datasets depending on format:
    ds1 = xr.open_dataset(file1) if extension1 == ".nc" else xr.open_zarr(file1) # model

    if file1 == file2:
        ds2 = ds1  # save memory if data and mask are in the same file
    else:
        ds2 = xr.open_dataset(file2) if extension2 == ".nc" else xr.open_zarr(file2) #mask model 

    # Open SWOT grid Dataset:
    ds3 = xr.open_dataset(file3)

    # Select first time step if a time dimension exists (SWOT only needs 2D fields):
    if time_name in ds1.dims:
        if ds1[time_name].size == 1:
            ds1 = ds1.squeeze()  # remove time dimension if it has only one value
        elif ds1[time_name].size > 1:
            ds1 = ds1.sel({time_name: ds3.time.mean()}, method="nearest")
            logging.info(f"Selecting nearest time step from model dataset: {ds1[time_name].values}")
            ds1 = ds1.squeeze()

    return ds1, ds2, ds3


def select_area(
    lat:xr.DataArray,
    lon:xr.DataArray,
    var:xr.DataArray,
    latitude_array:xr.DataArray,
    longitude_array:xr.DataArray
):
    """
    Selects the subset of model data (lon, lat, var) located within a swath defined by satellite coordinates.
    
    Parameters
    ----------
    lat : xr.DataArray
        Latitude coordinates of the model grid.
    lon : xr.DataArray
        Longitude coordinates of the model grid.
    var : xr.DataArray
        Variable data to be selected.
    latitude_array : xr.DataArray
        Latitude coordinates of the SWOT swath.
    longitude_array : xr.DataArray
        Longitude coordinates of the SWOT swath.

    Returns
    -------
    tuple
        A tuple containing the selected longitude, latitude, and variable data.
    """
    
    # Remove invalid points (0, 0):
    mask_valid = ~((lon.flatten() == 0) & (lat.flatten() == 0))
    lon_clean = lon.flatten()[mask_valid]
    lat_clean = lat.flatten()[mask_valid]
    var_clean = var.flatten()[mask_valid]
    
    points = np.column_stack((lon_clean, lat_clean))
        
    # Adjust longitudes to [-180, 180]:
    X = xr.where(longitude_array <= 180, longitude_array, longitude_array - 360)
    Y = latitude_array
    
    # Polygon with margin:
    dy = Y.values[-1,0] - Y.values[0,0]
    
    k = 2 if dy > 0 else -2
    k1 = abs(k)
    
    xx = np.concatenate([
        X.isel(num_lines=0).values,
        X.isel(num_pixels=-1).values+k,
        X.isel(num_lines=-1).values[::-1],
        X.isel(num_pixels=0).values[::-1]-k
    ])

    yy = np.concatenate([
        Y.isel(num_lines=0).values-k,
        Y.isel(num_pixels=-1).values-k1,
        Y.isel(num_lines=-1).values[::-1]+k,
        Y.isel(num_pixels=0).values[::-1]+k1
    ])

    polygon = np.column_stack((xx, yy))  
    inside, on_edge = inpoly2(points, polygon)
    mask = inside | on_edge

    lon_in = lon_clean[mask]
    lat_in = lat_clean[mask]
    var_in = var_clean[mask]
     
    return lon_in, lat_in, var_in


def open_model_data(
    ds_var,
    ds_coords,
    interpolator,
    var,
    latitude_array,
    longitude_array,
    lat_name="latitude",
    lon_name="longitude",
):
    """
    Creates an interpolator from a model dataset containing the ssh variable.
    The spatial coordinates (latitude and longitude) are provided as 2D variables in a separate dataset.

    Parameters
    ----------
    ds_var : xarray.Dataset
        Dataset containing the model ssh to interpolate.
    ds_coords : xarray.Dataset
        Dataset containing latitude and longitude as 2D variables.
    var : str
        Name of the variable to interpolate.
    latitude_array : xarray.DataArray or np.array
        Latitude of each satellite pixel (shape = [num_lines, num_pixels])
    longitude_array : xarray.DataArray or np.array
        Longitude of each satellite pixel (shape = [num_lines, num_pixels])
    lat_name : str, optional
        Name of the latitude variable in ds_coords (default: "latitude").
    lon_name : str, optional
        Name of the longitude variable in ds_coords (default: "longitude").

    Returns
    -------
    finterp : LinearNDInterpolator
        Interpolator for irregular 2D (latitude, longitude) grid.
    """
    # Check if the variable exists in ds_var:
    if var not in ds_var:
        raise ValueError(f"Variable '{var}' is not present in the provided dataset.")

    # Extract latitude and longitude from ds_coords (as 2D arrays):
    try:
        lat_values = ds_coords[lat_name].values  # Shape (x, y)
        lon_values = ds_coords[lon_name].values  # Shape: x, y)
    except KeyError:
        raise ValueError(f"Could not find '{lat_name}' or '{lon_name}' in the coordinates dataset.")

    # Extract variable values:
    var_values = ds_var[var].values

    lon_in, lat_in, var_in = select_area(lon=lon_values,
                                         lat=lat_values,
                                         var=var_values,
                                         latitude_array=latitude_array,
                                         longitude_array=longitude_array
                                        )

    # Flatten the 2D grid into 1D arrays:
    lat_flat = lat_in # lat_values.flatten()
    lon_flat = lon_in # lon_values.flatten()
    var_flat = var_in # var_values.flatten()

    if np.size(var_flat)!=0: #Checking is not empty
        # Create a scattered data interpolator (car c'est irregular 2D grids):
        if interpolator == "scipy_interpolator":
            finterp = interpolate.LinearNDInterpolator(
               list(zip(lat_flat, lon_flat)),
                var_flat,
                fill_value=np.nan
            )

        elif interpolator == "pyinterp_interpolator":
            points = np.column_stack((lon_flat, lat_flat))
            finterp = pyinterp.RTree()
            finterp.packing(points, var_flat)
        else:
            raise ValueError(f"Unknown interpolator: {interpolator}")

        return finterp
    else:
        return 0


def interp_satellite(
    latitude_array:xr.DataArray | np.ndarray,
    longitude_array:xr.DataArray | np.ndarray,
    cross_dist:xr.DataArray | np.ndarray,
    quality_flag:xr.DataArray | np.ndarray,
    interpolator:str,
    interp:interpolate.LinearNDInterpolator | pyinterp.RTree,
    var:str,
):
    """
    Interpolates the modeled SSH at satellite observation points (wide swath only).

    Parameters
    ----------
    latitude_array : xarray.DataArray | np.ndarray
        Latitude of each satellite pixel (shape = [num_lines, num_pixels])
    longitude_array : xarray.DataArray | np.ndarray
        Longitude of each satellite pixel (shape = [num_lines, num_pixels])
    cross_dist : xarray.DataArray | np.ndarray
        Cross-track distance of each satellite pixel (shape = [num_lines, num_pixels])
    quality_flag : xarray.DataArray | np.ndarray
        Quality flag for each satellite pixel (shape = [num_lines, num_pixels])
    interpolator : str
        Interpolator type to use (e.g., "scipy_interpolator" or "pyinterp_interpolator")
    interp : interpolate.LinearNDInterpolator | pyinterp.RTree
        Interpolator object created from the model data
    var : str
        Name of the ssh variable (e.g., "ssh_debug")

    Returns
    -------
    ds : xarray.Dataset
        Dataset of interpolated SSH values, structured for wide swath data.
    """

    # Ensure latitude and longitude are NumPy arrays before flattening
    longitude_array = xr.where(longitude_array>180 , longitude_array-360, longitude_array)  # swot longitude conversion from 0/360 to 180/-180

    latitude_array = np.asarray(latitude_array)
    longitude_array = np.asarray(longitude_array)

    # Flatten the satellite lat/lon arrays to feed into the interpolator
    points = np.column_stack((latitude_array.flatten(), longitude_array.flatten()))

    # Apply the interpolator to get SSH values at satellite positions

    logging.info("In Progress: Interpolating model SSH onto SWOT grid...")

    if interpolator == "scipy_interpolator":
        # Flatten the satellite lat/lon arrays to feed into the interpolator
        points = np.column_stack((latitude_array.flatten(), longitude_array.flatten()))       
        ssh_interp = interp(points).reshape(latitude_array.shape)

    elif interpolator == "pyinterp_interpolator":
        points = np.column_stack((longitude_array.flatten(), latitude_array.flatten()))
        ssh_interp = interp.inverse_distance_weighting(
            coordinates=points,                                  
            k=5,    # We are looking for at most ' neighbours
            num_threads=0, # parallel computing                         
            p=2,  #The power to be used by the interpolator inverse_distance_weighting.
            within=True
            )[0].reshape(latitude_array.shape) 

    # Rename variable if needed
    if var != "ssh":
        var = "ssh"

    # Create an xarray dataset for wide swath data
    ds = xr.Dataset({
        var: (["num_lines", "num_pixels"], ssh_interp)
    }, coords={
        "latitude": (["num_lines", "num_pixels"], latitude_array),
        "longitude": (["num_lines", "num_pixels"], longitude_array)
    })
    
    # Removing data from where swot does not have any (inter-swath and periphery areas) and from the continent (eventually) [see quality flag in the documentation]
    # Only values between 10 and 60 km to the nadir are considered as valid data. https://www.aviso.altimetry.fr/fileadmin/documents/data/tools/hdbk_duacs_SWOT_L3.pdf                                                                                               
    mask = xr.where((abs(cross_dist)<=60.0) & (abs(cross_dist)>=10.0) & (quality_flag<101) ,cross_dist,np.nan)                                                                             
    ds["ssh"] = ds["ssh"].where(~np.isnan(mask))
    
    # swot longitude conversion back to 0/360
    ds.coords['longitude']= xr.where(ds.longitude<0 , ds.longitude+360, ds.longitude)  
    return ds


def validate_interpolator(value):
    """
    Validate the interpolator argument.
    """
    if value.lower() not in ALLOWED_INTERPOLATORS:
        raise argparse.ArgumentTypeError(
            f"Invalid interpolator method: '{value}'. "
            f"Allowed methods are: {', '.join(ALLOWED_INTERPOLATORS)}"
        )
    return value.lower() 


def model2swot():
    """
    Main function to perform ocean model to SWOT interpolation.
    """
    # Logging to stdout:
    initialise_logging()
    # Parse CL args:
    parser = create_parser()
    args = parser.parse_args()

    # Pre-check existence of input files:
    for f in [args.model_file, args.mask_file, args.swot_file]:
        if not os.path.exists(f):
            parser.error(f"Error: Input file not found: {f}")
        if not os.path.isfile(f):
            parser.error(f"Error: Path is not a file: {f}")

    # Set interpolator:
    interpolator = args.interpolator
    logging.info(f"In Progress: Processing with {interpolator.split('_')[0]} interpolator...")
    # Read input netCDF files:
    ds_model, ds_mask, ds_swot = read_netcdf_files(args.model_file,
                                                   args.mask_file,
                                                   args.swot_file,
                                                   args.model_time_var,
                                                   )
    # Subset model data using SWOT swath:
    finterp = open_model_data(ds_model,
                              ds_mask,
                              interpolator,
                              args.model_ssh_var,
                              ds_swot.latitude,
                              ds_swot.longitude,
                              args.model_lat_var,
                              args.model_lon_var
                              )

    # Writing outputs to netCDF:
    if finterp !=0: # Checking finterp is not empty
        output_ds = interp_satellite(ds_swot.latitude,
                                     ds_swot.longitude,
                                     ds_swot.cross_track_distance,
                                     ds_swot.quality_flag,
                                     interpolator,
                                     finterp,
                                     var="ssh"
                                     )
        output_ds.to_netcdf(args.output_file)
        logging.info("Completed: Interpolation finished successfully.")
    else:
        logging.info("Error: Model input has no information for the SWOT path.")
    sys.exit(0)
