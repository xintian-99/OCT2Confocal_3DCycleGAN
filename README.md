# OCT2Confocal Dataset

## Overview
The OCT2Confocal dataset uniquely combines in-vivo grayscale Optical Coherence Tomography (OCT) images with ex-vivo colored confocal images from C57BL/6 mice. This dataset includes three sets of retinal images: A2L, A2R, and B3R, representing individual mice and their respective eyes.

![OCT and Confocal Images](images/OCTandconfocal.png)

## Image Acquisition and Processing

### In-vivo OCT Images
- **Acquisition Details**: Captured using a Micron IV fundus camera with a mouse-specific lens. 
- **Resolution and Artifacts**: 1024×512×512 pixels. Artifacts may include motion, speckle noise, multiple scattering, attenuation, or beam-width artifacts.
- **Volume Scans**: Centered around the optic disc, these scans target retinal layers between the Inner Limiting Membrane (ILM) and Inner Plexiform Layer (IPL).
- **2D OCT Projections**: Created by summing in the z-direction for a comprehensive view.

### Ex-vivo Confocal Images
- **Procedure**: Post OCT imaging, the retinas were prepared for confocal imaging using a Leica SP5-AOBS microscope.
- **Staining and Channels**: 
  - **Red (Isolectin IB4)**: Stains endothelial cells in blood vessels.
  - **Green (CD4)**: Highlights CD4+ T cells.
  - **Blue (DAPI)**: Stains cell nuclei.
  - **White (Iba1)**: Stains microglia and macrophages.
- **RGB Images**: Combinations of red, green, and blue channels for model training.

### Test Dataset
- **Acquisition and Purpose**: This dataset contains 22 OCT images acquired in the same manner as the primary dataset, but without corresponding confocal matches. These images are crucial for evaluating model performance and can be further used to advance multimodal image analysis.

## Dataset Applications
The OCT2Confocal dataset is instrumental in enhancing retinal analysis and improving diagnostic accuracy in ophthalmology, with significant potential in medical image processing for multimodal image translation, image registration, and model training.

## Dataset Availability
The full dataset will be released upon the publication of our paper. This release will allow researchers and practitioners full access to the dataset for their studies and applications.

### Registration Form
To request early access, please fill in this [registration form](#). The download link will be shared post submission.

## Citation
If you use the OCT2Confocal dataset in your research, please cite the following paper:
