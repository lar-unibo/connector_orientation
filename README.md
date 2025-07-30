# Monocular Estimation of Connector Orientation: Combining Deformable Linear Object Priors and Smooth Angle Classification
Accepted at **2024 IEEE International Conference on Advanced Intelligent Mechatronics (AIM)**

## Abstract
In this paper, a novel method for monocular estimation of connector orientation in wire and cable harnesses is introduced. 
The proposed approach combines Deformable Linear Objects (DLOs) priors with a learning-based technique for smooth angle classification, enabling precise prediction of connector orientation using only a single RGB image. 
The integration of DLO perception is crucial in recovering an initial coarse understanding of the connector pose. 
This is accomplished by linearly projecting the identified DLO endpoint using the predicted spline-based DLO representation model. 
To estimate the axial orientation of the connector, the proposed approach incorporates a smooth labeling technique in the angle classification process. 
This ensures effective handling of the circular nature inherent in angular data. Additionally, a self-supervised acquisition and annotation of the dataset samples is employed. 
To assess the effectiveness of the proposed method, we conducted experiments with a collection of real-world connectors sourced from the automotive sector. 
The outcomes underscore the potential applications of the proposed method in tasks related to the robotic manufacturing and assembly of complex deformable linear objects, such as wire harnesses.

### How to Run

This repository contains the official implementation of the paper.

- Download Supplementary Materials
Download the pretrained model checkpoints and test data from the following link: 🔗 [Download from MEGA](https://mega.nz/file/IUU0SLAJ#ydBCGX1tu0426b2I8UnwLhxpTdyUc67AnsONrhnuoWU)

- Unzip the files and place them in the main directory of the codebase.

- Try to run the predict script with the command:
  ```bash
  python predict.py path_to_checkpoint.pth
  ```
