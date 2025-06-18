# synthocean

**The repository is currently in progress!**  

The file `Interp_model2SWOT.py` contains the code to execute the command line from your terminal using the following command:

```bash
./Interp_model2SWOT.py -m path_to_your_model_file -k path_to_your_model_mask_file -s path_to_swot_data_file -o path_to_output_file -i interpolator --model-lat-var latitude_var_name --model-lon-var longitude_var_name --model_ssh_var the_model_ssh_variable_name --model-time-var time_name
```

## ⚡ Notes

✅ <strong>Please note</strong>: If you are using a model where the mask and the data are included in the **same file**, you need to provide the link to this file twice — once for the data argument and once for the mask argument.

The output file **must be a NetCDF file** (`.nc`).

Additionally, please make sure to include the following two arguments:

- `--model-lat-var`
- `--model-lon-var`

unless the names of your coordinate variables in the model file are already `latitude` and `longitude`. Otherwise, you should explicitly provide their names.

The tool is currently tested with **eORCA25** and **eNATL60** data in **NetCDF** and **Zarr** format.


The data needed for testing will soon be provided via an S3 endpoint.
