# ROI_evaluation_rPPG
## Introduction
This project contains the code for the systematic analysis of the optimal facial ROIs for rPPG algorithms in heart rate (HR) estimation. It is a research project (24ECTS) in BHMT group of D-HEST, ETH Zurich.  
![image](rppg_pipeline.png)
![image](ROI_division.png)
### Summary
This project mainly comprises five parts:
1. Transforming facial video signals into HR values. 28 facial ROIs are defined.
2. Examinining the influence of subject's motion types on ROI performance.
3. Examinining the influence of light intensities on ROI performance.
4. Examinining the impact of fringe and beard occlusion on ROI performance.
### Main Contributions
1. We compared the performance of 28 facial ROIs under different motion types and illumination levels.
2. We examined the impact of fringe and beard occlusion on rPPG signal extraction and HR estimation.
3. We proposed the utilization of glabella as the preferred facial ROI. In contrast to forehead and cheek that are more commonly employed, our findings indicate that glabella demonstrates a performance level at least on par with these regions and a better robustness in complex environments.
## Package Requirements
- dtaidistance==2.3.10   
- matplotlib==3.5.3  
- mediapipe==0.9.0.1  
- numpy==1.20.3  
- open3d==0.17.0  
- opencv_contrib_python==3.4.11.45  
- opencv_python==4.7.0.72  
- opencv_python_headless==4.5.3.56  
- pandas==1.3.5  
- pyVHR==2.0  
- PyYAML==6.0.1  
- scikit_learn==1.0.2  
- scipy==1.7.3  
- seaborn==0.11.1  
- statsmodels==0.12.2  
- tqdm==4.61.1  
- ~atplotlib==3.3.4
## Configuration
The experiments were runned on the author's personal laptop. The configurations are provided as the reference:
- CPU: AMD Ryzen 9 5900HX with Radeon Graphics
- GPU: NVIDIA GeForce RTX 3080 Laptop GPU
- CUDA Version: 11.7
- Operating System: Microsoft Windows 11 (version-10.0.22621)
## Code Structure
```bash
optimal_roi_rppg  
├─ config
│    └─ options.yaml  
├─ data
│    ├─ BUAA-MIHR
│    │    ├─ hr
│    │    └─ rgb
│    ├─ LGI-PPGI
│    │    ├─ hr
│    │    └─ rgb
│    ├─ UBFC-Phys
│    │    ├─ feature
│    │    ├─ hr
│    │    └─ rgb
│    ├─ UBFC-rPPG
│    │    ├─ feature
│    │    ├─ hr
│    │    └─ rgb
│    └─ anatomy.csv
├─ main
│    ├─ main_evaluation.py
│    ├─ main_extract_feature.py
│    ├─ main_extract_rgb.py
│    ├─ main_gen_gtHR.py
│    └─ main_rgb2hr.py
├─ plot
│    ├─ simple
│    │    └─ ...
│    ├─ illumination
│    │    └─ ...
│    ├─ motion
│    │    ├─ ...
│    └─ occlusion
│           ├─ ...
├─ result
│    ├─ BUAA-MIHR
│    │    ├─ evaluation_CHROM.csv
│    │    ├─ evaluation_LGI.csv
│    │    ├─ evaluation_OMIT.csv
│    │    └─ evaluation_POS.csv
│    ├─ LGI-PPGI
│    │    ├─ evaluation_CHROM.csv
│    │    ├─ evaluation_LGI.csv
│    │    ├─ evaluation_OMIT.csv
│    │    └─ evaluation_POS.csv
│    ├─ UBFC-Phys
│    │    ├─ evaluation_CHROM.csv
│    │    ├─ evaluation_LGI.csv
│    │    ├─ evaluation_OMIT.csv
│    │    └─ evaluation_POS.csv
│    └─ UBFC-rPPG
│           ├─ evaluation_CHROM.csv
│           ├─ evaluation_LGI.csv
│           ├─ evaluation_OMIT.csv
│           └─ evaluation_POS.csv
├─ util
│    ├─ util_feature.py
│    └─ util_pre_analysis.py
└─ visualization
       ├─ main_comparison_beard.py
       ├─ main_comparison_fringe.py
       ├─ main_illumination_rank.py
       └─ main_motion_rank.py
       └─ main_simple.py
```
## Datasets
1. UBFC-rPPG: https://sites.google.com/view/ybenezeth/ubfcrppg
2. UBFC-Phys: https://sites.google.com/view/ybenezeth/ubfc-phys and https://ieee-dataport.org/open-access/ubfc-phys-2
3. LGI-PPGI: https://github.com/partofthestars/LGI-PPGI-DB
4. BUAA-MIHR: https://github.com/xilin1991/Large-scale-Multi-illumination-HR-Database
## Usage
First, activate the local environment and then set the folder containing this README file as the current folder.  
For Windows, execute: **python (...).py**  
For Linux, execute: **python3 (...).py**  
1. Transform facial videos into raw RGB traces: **python "./main/main_extract_rgb.py"**
2. Transform raw RGB traces into BVP and HR values: **python "./main/main_rgb2hr.py"**
3. Extract features (ROI size, facial surface orientation) from facial videos: **python **"./main/main_rgb2hr.py"**
4. Generate ground truth heart rate values for UBFC-Phys dataset: **python "./main/main_gen_gtHR.py"**
5. Evaluate the performance of different facial ROIs on selected datasets: **python "./main/main_evaluation.py"**
6. Compare the ROI performance in the simple environment: **python "./visualization/main_simple.py"**
7. Compare the ROI performance under different motion types: **python "./visualization/main_motion_rank.py"**
8. Compare the ROI performance under different illumination levels: **python "./visualization/main_illumination_rank.py"**
9. Examine the influence of beard coverage on the performance of relevant facial ROIs: **python "./visualization/main_comparison_beard.py"**
10. Examine the influence of fringe coverage on the performance of relevant facial ROIs: **python "./visualization/main_comparison_fringe.py"**
### Contact
If you have any questions, please feel free to contact me through email (shuoli199909@outlook.com)!
## Authors and acknowledgment
This research project was supervised by Dr. Mohamed Elgendi and Prof. Dr. Carlo Menon. The code was developed by Shuo Li. Also thank to all of the BMHT members and providers of datasets for the continuous help!
## License - MIT License.
