README.txt for the OCT2Confocal Dataset

Overview:
The OCT2Confocal dataset, developed by the Autoimmune Inflammation Research (AIR) group at the University of Bristol, combines in-vivo OCT grayscale images with ex-vivo colored confocal images from C57BL/6 mice with induced autoimmune uveitis. It includes three sets of retinal images: A2L, A2R, and B3R.

Dataset Contents:
1. dataset_3cyclegan: 
   - Training datasets for "OCT2Confocal: 3D CycleGAN based Translation of Retinal OCT Images to Confocal Microscopy" (ISBI 2024).
   - Source code for the 3dCycleGAN model at [https://github.com/xintian-99/OCT2Confocal].

2. OCT: 
   - 22 OCT volumes, B-scans are preprocessed for clarity (denoise, enhanced, and registered).
   - Includes 2D OCT Projections: enface images of the entire OCT volume (enface.tif) and at various depths (enfacexxx.tif).

3. OCT-Confocal: 
   - Matched OCT and confocal images.
   - OCT B-scans are preprocessed for clarity (denoise, enhanced, and registered) and 
   - Confocal images in RGB .tif stacks and single-color Red, Green, Blue .tif stacks.
   - Resolutions: A2L (512×512×14), A2R (512×512×11), B3R (512×512×14).

Image Acquisition and Processing Details:
- OCT Images:
  - Captured with a Micron IV fundus camera, OCT scan head, and a mouse objective lens from Phoenix Technologies, California.
  - Resolution: 1024×512×512 pixels.
  
- Confocal Images:
  - Mice euthanized on day 24, retinas extracted and prepared for confocal imaging.
  - Imaging with Leica SP5-AOBS confocal laser scanning microscope and Leica DM I6000 inverted epifluorescence microscope.
  - Retinas stained with antibodies attached to four distinct fluorochrome: 
    - Red (Isolectin IB4) for endothelial cells in blood vessels.
    - Green (CD4) for CD4+ T cells.
    - Blue (DAPI) for cell nuclei.
  - RGB Images: Combination of channels for model training.


Usage and Applications:
Ideal for developing image translation models in medical imaging. Relevant for research in retinal disease diagnosis, autoimmune disorders, and OCT/confocal image processing.

Citation:
Reference as: [OCT2Confocal: 3D CycleGAN based Translation of Retinal OCT Images to Confocal Microscopy, ISBI 2024].

License:
Released under [specify license type]. Details at [specify link or contact info].

Contact:
For more information, contact [Insert Contact Information].

Acknowledgments:
Thanks to the AIR group at the University of Bristol for their work in acquiring and processing the images.

Note: Dataset provided "as is". Authors not responsible for any errors or results from its use.

[End of README.txt]
