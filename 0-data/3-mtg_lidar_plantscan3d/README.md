# MTGs from LiDAR scans

## From point cloud to MTG

The MTG files in `1-raw_output` were made using the [plantscan3d software](https://plantscan3d.readthedocs.io/) and the LiDAR point clouds of each branch (`0-data/2-lidar_processing/2-grouped_point_cloud/2-branches`). The software was used using default parameter values for the branch skeletization and the node diameters (mean-distance algorithm). The MTGs are the raw results output from plantscan3d.

Then, those MTG files were copied into the `2-manually_corrected` folder, and manually corrected with a maximum working intensity of ~40min for each branch (worker: A. bonnet). The corrections included wrong branching, missed structures, wrong relationship (axis or branch), and diameter smoothing.

## From nodes to segments and axis

The MTGs from the `3-raw_output_segmentized` and `4-corrected_segmentized` folders are the same than the previous one, but the format of the MTG is automatically converted onto the format used on the manual acquisitions, *i.e.* with segments (S) and axis (A) instead of nodes (N).

## Axis identification

The `5-corrected_segmentized_id` folder contains the same MTGs as `4-corrected_segmentized`, but with an added attribute: the id.This id helps to match branch second order axis between the plantscan3d MTGs and the manually measured MTGs. See this [README.md](../1.1-mtg_manual_measurement_corrected_id/README.md) for further details.

## Tests

The `0-tests` folder contains some pre-tests on the effect of the parameter values on the estimation of the node radius. The parameters included points contraction, mean/max radius estimation and filtering).
