# Data files and folders description

## Introduction

The raw data is the data directly from the field. This data was never modified from the moment it was taken. The dataset is composed of LiDAR acquisitions, and manual measurements on branches that were also measured by LiDAR.

### Manual measurements

All obvious mistakes and inconsistencies were modified in a copy of the data in `1.0-mtg_manual_measurement_corrected`. Then the common axis between manual measurements and Plantscan3d reconstruction where identified for comparison (`1.1-mtg_manual_measurement_corrected_id`), adding a new `id` column. Then it was enriched with computed data, and the result is available in `1.2-mtg_manual_measurement_corrected_enriched`. The code for the computation is available in `1-compute_field_mtg_data.jl`.

### LiDAR measurements

The LiDAR data was processed to clean the point cloud, and extract the whole trees or only the branches for each independent scan position (`1-isolations_per_scan`), or all scans positions at once (`2-grouped_point_clouds`). The scan positions were also extracted to compute the distance between each scan and the branches.

The LiDAR point clouds of the branches were then used to compute the 3D topology of each branch using either all scan positions at once, or some. The resulting MTG files are written in `1-raw_output`. These MTGs were then corrected for the biggest mistakes of the algorithm for ~40min each branch (`2-manually_corrected`). Both MTG sets were then re-formatted to match the MTG specification of the ones from the field measurements, *i.e.* with segments and axes instead of nodes. The code for this computation is available in `1-code/3-mtg_plantscan3d_to_segments.jl`.

## Files and folders overview

Here is a tree representing all folders and files in `0-data`, along with a short description of what they are:

```bash
│
├───0-raw # Raw data from field experiment
│   ├───1-lidar # Whole plot LiDAR data from the field + scans positions
│   │       pointcloud-vezenobres.bin
│   │
│   └───2-manual_measurements # Manual measurements on branches (topology + dimensions + density)
│       │
│       ├───1-mtg # Multi-Scale Tree Graph of each branch: topology + dimensions
│       │       tree11h.xlsm # MTG for the tree 11, highest branch
│       │       tree11l.xlsm # MTG for the tree 11, lowest branch
│       │       tree12h.xlsm # ...
│       │       tree12l.xlsm
│       │       tree13h.xlsx
│       │       tree13l.xlsm
│       │       ArchiMacro.xlsm # Template file with a macro for field measurements
│       │
│       ├───Demol_et_al.2022_data.csv # Data from Demol et al. (2022) for comparison (see 1-code/0-Demol_et_al.2022_data.jl for more details)
│       │
│       └───2-wood_density_measurements # Wood density measurements made on branches samples
│               README.md # More detailed readme file
│               sample-data-2-branches-juin.csv # results for tree 11
│               sample-data-4-branches-avril.csv # results for tree 12 and 13
│
├───1.0-mtg_manual_measurement_corrected # Same as 0-raw/2-manual_measurements but corrected for inconsistencies
│       README.md     # More detailed readme file
│       metadata.csv  # metadata about the columns description and units
│       tree12l.xlsm
│       [...]
│
├───1.1-mtg_manual_measurement_corrected_id # Same as 1.0-mtg_manual_measurement_corrected but A2 axis identified with unique ID to match plantscan3d axis
│       metadata.csv  # metadata about the columns description and units
│       tree12l.xlsm
│       [...]
│
├───1.2-mtg_manual_measurement_corrected_enriched # Same as `1-mtg_manual_measurement_corrected` but enriched with computed data
│       tree11h.mtg # Enriched MTG file
│       tree11h.csv # Same as `tree11h.mtg` but in a DataFrame format
│       [...]
│
├───2-lidar_processing # Processed LiDAR data from A. Bonnet (2021)
│   ├───1-isolations_per_scan # point cloud of isolated trees or branches for each scan position
│   │   ├───1-trees # Point clouds of each scan position for all trees
│   │   │       ScanPos001 - SINGLESCANSJust-three - 210218_090717 - Cloud.txt
│   │   │       ScanPos002 - SINGLESCANSjust-three - 210218_091149 - Cloud.txt
│   │   │       [...]
│   │   └───2-branches # Point clouds of each scan position for each tree branch
│   │       ├───tree13l
│   │       │       ScanPos001 - SINGLESCANSJust-A3BL - 210218_090717 - Cloud - Cloud.txt
│   │       │       ScanPos002 - SINGLESCANSjust-A3BL- 210218_091149 - Cloud - Cloud.txt
│   │       │       [...]
│   │       │
│   │       ├───tree11h
│   │       │       ScanPos001 - SINGLESCANSJust-A1BH- 210218_090717 - Cloud - Cloud.txt
│   │       │       ScanPos002 - SINGLESCANSjust-A1BH- 210218_091149 - Cloud - Cloud.txt
│   │       │       [...]
│   │       └─── [...]
│   │
│   ├───2-grouped_point_clouds # Scan positions grouped into a single file
│   │   ├───1-trees # All scans positions into one file for each tree
│   │   │       all_scans_tree_11.txt
│   │   │       all_scans_tree_12.txt
│   │   │       all_scans_tree_13.txt
│   │   └───2-branches # All scans positions into one file for each branch and manually cleaned
│   │           ALLSCANS-tree13l-Cloud.txt
│   │           ALLSCANS-tree11h-Cloud.txt
│   │           [...]
│   └───3-scans_positions # Scan positions data
│           scan_positions.csv
│
├───3-mtg_lidar_plantscan3d # Topology and length reconstruction from point clouds using plantscan3d
│   │   README.md     # More detailed readme file
│   │
│   ├───1-raw_output # MTG output from plantscan3d, without any modification
│   │       tree11h.mtg
│   │       [...]
│   │
│   ├───2-manually_corrected # MTG from plantscan3d, corrected manually for biggest errors (~40 min work each branch)
│   │       tree11h.mtg
│   │       [...]
│   │
│   ├───3-raw_output_segmentized # MTG from `1-raw_output`, modified to match the field format (A + S instead of N)
│   │       tree11h.mtg
│   │       [...]
│   │
│   ├───4-corrected_segmentized # MTG from `2-manually_corrected`, modified to match the field format (A + S instead of N)
│   │       tree11h.mtg
│   │       [...]
│   │
│   ├───5-corrected_segmentized_id # MTG from `4`, with a unique ID for each axis also identified in the mtg from lidar data 
│   │       tree11h.mtg
│   │       [...]
│   │
│   ├───6-corrected_segmentized_id_enriched # MTG from `5`, enriched with computed data
│   │       tree11h.mtg
│   │       [...]
│   │
│   ├───7-tree_scale # MTG at tree scale
│   │       all_scans_tree_11.mtg
│   │       [...]
│   │
│   ├───8-tree_scale_segmentized # MTG at tree scale, segmentized
│   │       all_scans_tree_11.mtg
│   │       [...]
│   │
│   └───9-tree_scale_segmentized_enriched # MTG at tree scale, segmentized and enriched with computed data
│           all_scans_tree_11.mtg
│           [...]
│
└───4-method_visualization
        tree11h_extract.txt # Points for the point cloud extract
        tree11h_extract_pmt.mtg
        tree11h_extract_point_cloud.mtg # MTG resulting from the point cloud extract
```

---

For more details, please read the code in `1-code`.
