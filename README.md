# OCT2Confocal Dataset

## Overview
The OCT2Confocal dataset uniquely combines in-vivo grayscale Optical Coherence Tomography (OCT) images with ex-vivo colored confocal images from C57BL/6 mice. This dataset includes three sets of retinal images: A2L, A2R, and B3R, representing individual mice and their respective eyes.

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

## Dataset Application
The dataset aims to enhance retinal analysis and improve diagnostic accuracy in ophthalmology, particularly in autoimmune uveitis studies.

### Test Dataset
Contains 22 OCT images without confocal matches for model evaluation.

## Downloading the Dataset
The database will be available upon the publication of our paper. Access requires registration.

### Registration Form
To request access, please fill in this [registration form](#). The download link will be shared post submission.

For more information, refer to our publication or contact the dataset curator.
