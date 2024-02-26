# OCT2Confocal 3DCycleGAN

## Dataset

### Overview
The OCT2Confocal dataset uniquely combines in-vivo grayscale Optical Coherence Tomography (OCT) images with ex-vivo colored confocal images from C57BL/6 mice. This dataset includes three sets of retinal images: A2L, A2R, and B3R, representing individual mice and their respective eyes.

![OCT and Confocal Images](images/OCTandconfocal.png)

### Image Acquisition and Processing

#### In-vivo OCT Images
- **Acquisition Details**: Captured using a Micron IV fundus camera with a mouse-specific lens. 
- **Resolution and Artifacts**: 1024×512×512 pixels. Artifacts may include motion, speckle noise, multiple scattering, attenuation, or beam-width artifacts.
- **Volume Scans**: Centered around the optic disc, these scans target retinal layers between the Inner Limiting Membrane (ILM) and Inner Plexiform Layer (IPL).
- **2D OCT Projections**: Created by summing in the z-direction for a comprehensive view.

#### Ex-vivo Confocal Images
- **Procedure**: Post OCT imaging, the retinas were prepared for confocal imaging using a Leica SP5-AOBS microscope.
- **Staining and Channels**: 
  - **Red (Isolectin IB4)**: Stains endothelial cells in blood vessels.
  - **Green (CD4)**: Highlights CD4+ T cells.
  - **Blue (DAPI)**: Stains cell nuclei.
  - **White (Iba1)**: Stains microglia and macrophages.
- **RGB Images**: Combinations of red, green, and blue channels for model training.

There are 22 OCT images acquired in the same manner as the primary dataset, but without corresponding confocal matches. These images are for evaluating model performance and can be further used to advance multimodal image analysis.

### Dataset Applications
The OCT2Confocal dataset can be applied to:

- **Multimodal Image Translation**: Facilitating the development of advanced algorithms for translating between different imaging modalities, thereby providing a more comprehensive understanding of retinal conditions.
- **Image Registration**: Aiding in the alignment and analysis of multimodal images, which is crucial for accurate diagnosis and treatment planning.
- **Model Training**: Offering a unique dataset for training machine learning models in tasks related to retinal health, disease diagnosis, and treatment monitoring.

## Dataset Availability
The full dataset will be released upon the publication of our paper. This release will allow researchers and practitioners full access to the dataset for their studies and applications.

## Early Access
To request early access, please email xin.tian@bristol.ac.uk. The download link will be shared post-submission.

<!--### Registration Form
To request early access, please fill in this [registration form](#). The download link will be shared post submission.-->

<!-- ## Citation
If you use the OCT2Confocal dataset in your research, please cite the following paper:

@article{tian2023oct2confocal,
title={OCT2Confocal: 3D CycleGAN based Translation of Retinal OCT Images to Confocal Microscopy},
author={Tian, Xin and Anantrasirichai, Nantheera and Nicholson, Lindsay and Achim, Alin},
journal={arXiv preprint arXiv:2311.10902},
year={2023}
}


For more information, refer to our publication or contact the dataset curator. -->
