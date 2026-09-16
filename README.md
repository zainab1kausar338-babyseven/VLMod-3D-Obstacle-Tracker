
VLMod: Multi-Camera Spatial Obstacle Detection & Vector TrackingA lightweight, automated processing pipeline designed for multi-camera spatial obstacle detection, frame-level vector extraction, and temporal track annotation across distributed video streams.🔍 FeaturesMulti-View Parsing: Extracts timestamped obstacle trajectory files following the Dataset_Camera_Res_StartTS_EndTS_ObstacleID schema.Automated Filtering: Separates zero-padded background/empty detections from active frame entries.Dataset Analytics: Generates per-dataset detection density statistics and counts.CSV Exporting: Consolidates multi-file vector outputs into a single tabular file ready for training pipelines (PyTorch, TensorFlow, OpenCV).📁 Repository StructurePlaintextvlmod/
│
├── result/
│   ├── active/                       # Processed active detection files (non-zero entries)
│   └── active_detections_summary.csv # Consolidated CSV dataset export
│
├── parse_stats.py                    # Dataset statistics & detection count script
├── export_csv.py                     # Aggregation script to convert .txt outputs to CSV
└── README.md                         # Project documentation
⚡ Quick Start1. PrerequisitesEnsure you have Python 3.8+ and PowerShell (for Windows workflow):Bashpip install pandas
2. Parse & Analyze Active DetectionsRun the analysis script to count active frame detections per Dataset ID:Bashpython parse_stats.py
3. Move Active Detection Files (PowerShell)Isolate non-zero obstacle files into an active directory:PowerShellNew-Item -ItemType Directory -Path "D:\vlmod\result\active" -Force
Get-ChildItem -Path "D:\vlmod\result\*.txt" | Where-Object { (Select-String -Path $_.FullName -Pattern "[1-9]") } | Move-Item -Destination "D:\vlmod\result\active\"
📊 Summary Output FormatExported detection records in active_detections_summary.csv include:FieldDescriptionExampledataset_idUnique ID of the dataset capture145044cameraCamera sensor identifierfa2sd4a06W152AIRresolutionInput frame vertical resolution420start_tsTracking epoch start timestamp1626155724end_tsTracking epoch end timestamp1626155908obstacle_idAssigned spatial tracking ID243frame_indexFrame position index inside stream1dataExtracted coordinate / bounding vector1 0 0📜 CitationCode snippet@article{vlmod2026,
  title={VLMod: Multi-Camera Spatial Obstacle Detection and Vector Tracking},
  author={VLMod Development Team},
  year={2026}
}
