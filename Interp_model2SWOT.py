#!/usr/bin/env python3
import argparse
import xarray as xr
import numpy as np
import pyinterp
from scipy import interpolate
 
def read_netcdf_files(file1, file2, file3):
    """
    Reads three NetCDF files and returns the datasets.
    
    Parameters:
        file1 (str): Path to the first NetCDF file (Model).
        file2 (str): Path to the second NetCDF file (Mask for model).
        file3 (str): Path to the third NetCDF file (SWOT data).
        
    Returns:
        tuple: A tuple containing three xarray datasets.
    """
    ds1 = xr.open_dataset(file1)# model
    ds2 = xr.open_dataset(file2) #mask model    
    ds3 = xr.open_dataset(file3) #swot
    
    ds1["ssh"] = ds1["sossheig"].isel(time_counter=0)
    del ds1["sossheig"]
    
    ds3["ssh"] = ds3["ssha"]  + ds3["mdt"] 
     
    return ds1, ds2, ds3



def open_model_data(ds_var, ds_coords,interpolator, var, lat_name="latitude", lon_name="longitude"):
    """
    Creates an interpolator from a model dataset containing the ssh variable.
    The spatial coordinates (latitude and longitude) are provided as 2D variables in a separate dataset.

    Parameters:
    - ds_var (xarray.Dataset): Dataset containing the model ssh to interpolate.
    - ds_coords (xarray.Dataset): Dataset containing latitude and longitude as 2D variables.
    - var (str): Name of the variable to interpolate.
    - lat_name (str, optional): Name of the latitude variable in ds_coords (default: "latitude").
    - lon_name (str, optional): Name of the longitude variable in ds_coords (default: "longitude").

    Returns:
    - finterp (LinearNDInterpolator): Interpolator for irregular 2D (latitude, longitude) grid.
    """

    # Check if the variable exists in ds_var
    if var not in ds_var:
        raise ValueError(f"Variable '{var}' is not present in the provided dataset.")

    # Extract latitude and longitude from ds_coords (as 2D arrays)
    try:
        lat_values = ds_coords[lat_name].values  # Shape (x, y)
        lon_values = ds_coords[lon_name].values  # Shape: x, y)
    except KeyError:
        raise ValueError(f"Could not find '{lat_name}' or '{lon_name}' in the coordinates dataset.")

    # Extract variable values
    var_values = ds_var[var].values  

    # Ensure the variable has the correct dimensions (latitude, longitude)
    if var_values.ndim == 3:  # If an extra time dimension exists
        var_values = var_values[0]  # Take only the first time step

    # Flatten the 2D grid into 1D arrays
    lat_flat = lat_values.flatten()
    lon_flat = lon_values.flatten()
    var_flat = var_values.flatten()

    # Create a scattered data interpolator  !!!!!!! car c'est irregular 2D grids)
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



def interp_satellite(latitude_array, longitude_array, ssh_swot,interpolator, interp, var):
    """
    Interpolates the modeled SSH at satellite observation points (wide swath only).

    Parameters:
    - latitude_array (xarray.DataArray or np.array): Latitude of each satellite pixel (shape = [num_lines, num_pixels])
    - longitude_array (xarray.DataArray or np.array): Longitude of each satellite pixel (shape = [num_lines, num_pixels])
    - interp (scipy.interpolate.LinearNDInterpolator): Interpolator from `open_model_data`
    - var (str): Name of the ssh variable (e.g., "ssh_debug")

    Returns:
    - ds (xarray.Dataset): Dataset of interpolated SSH values, structured for wide swath data.
    """

    # Ensure latitude and longitude are NumPy arrays before flattening
    longitude_array = xr.where(longitude_array>180 , longitude_array-360, longitude_array)  # swot longitude conversion from 0/360 to 180/-180
    
    latitude_array = np.asarray(latitude_array)
    longitude_array = np.asarray(longitude_array)

    # Flatten the satellite lat/lon arrays to feed into the interpolator
    points = np.column_stack((latitude_array.flatten(), longitude_array.flatten()))

    # Apply the interpolator to get SSH values at satellite positions
    
    print("Interpolation in progress ...")
    
    if interpolator == "scipy_interpolator":
        # Flatten the satellite lat/lon arrays to feed into the interpolator
        points = np.column_stack((latitude_array.flatten(), longitude_array.flatten()))       
        ssh_interp = interp(points).reshape(latitude_array.shape)
         
    elif interpolator == "pyinterp_interpolator":
        points = np.column_stack((longitude_array.flatten(), latitude_array.flatten()))
        ssh_interp = interp.inverse_distance_weighting(
            coordinates=points,                                  
            k=4,    # We are looking for at most ' neighbours
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

    
    ds["ssh"] = ds["ssh"].where(~np.isnan(ssh_swot)) # removing data from where swot does not have any (inter-swath and periphery areas)
    
    ds.coords['longitude']= xr.where(ds.longitude<0 , ds.longitude+360, ds.longitude)  # swot longitude conversion back to 0/360
    
    return ds


def save_netcdf(result, output_file):
    """
    Save the resulting dataset to a NetCDF file.
    """
    result.to_netcdf(output_file)

def main():
    parser = argparse.ArgumentParser(description="Processing workflow")
    parser.add_argument("file1", help="Path of the model NetCDF file")
    parser.add_argument("file2", help="Path of the mask NetCDF file")
    parser.add_argument("file3", help="Path of the SWOT NetCDF file")
    parser.add_argument("output", help="Path of the output nc file")
    parser.add_argument("interpolator", help="The interpolation method used")   # To be mentioned to the README file

    args = parser.parse_args()

    # read files NetCDF
    interpolator = args.interpolator
    ds_model, ds_mask, ds_swot = read_netcdf_files(args.file1, args.file2, args.file3)
    # Analyse
    finterp = open_model_data(ds_model, ds_mask,interpolator, "ssh","nav_lat","nav_lon")
    output_ds = interp_satellite(ds_swot.latitude, ds_swot.longitude,ds_swot.ssha, interpolator, finterp, var="ssh")
    
    # Sauvegarder le fichier
    save_netcdf(output_ds, args.output)

if __name__ == "__main__":
    main()

 
