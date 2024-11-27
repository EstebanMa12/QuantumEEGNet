# QuantumEEGNet

# QEEGNet: Quantum Machine Learning for Enhanced Electroencephalography Encoding

**Authors**: Michael Chi-Sheng Chen, Samuel Yen-Chi Chen, Aidan Hung-Wen Tsai, Chun-Shu Wei  
**Affiliations**: Neuro Industry, National Yang Ming Chiao Tung University, Brookhaven National Laboratory  

---

## Overview

**QEEGNet** is a hybrid neural network integrating **quantum computing** and the classical **EEGNet** architecture to enhance the encoding and analysis of EEG signals. By incorporating **variational quantum circuits (VQC)**, QEEGNet captures more intricate patterns within EEG data, offering improved performance and robustness compared to traditional models.

This repository contains the implementation and experimental results for **QEEGNet**, evaluated on the **BCI Competition IV 2a** dataset.

---

## Key Features

- **Hybrid Architecture**: Combines the EEGNet convolutional framework with quantum encoding layers for advanced feature extraction.
- **Quantum Layer Integration**: Leverages the unique properties of quantum mechanics, such as superposition and entanglement, for richer data representation.
- **Improved Robustness**: Demonstrates enhanced accuracy and resilience to noise in EEG signal classification tasks.
- **Generalizability**: Consistently outperforms EEGNet across most subjects in benchmark datasets.

---

## Architecture

QEEGNet consists of:
1. **Classical EEGNet Layers**: Initial convolutional layers process EEG signals to extract temporal and spatial features.
2. **Quantum Encoding Layer**: Encodes classical features into quantum states using a parameterized quantum circuit.
3. **Fully Connected Layers**: Converts quantum outputs into final classifications.

![Architecture Diagram](link-to-architecture-diagram.png)  

---

## Dataset

The **BCI Competition IV 2a** dataset was used for evaluation, featuring EEG signals from motor-imagery tasks.  
- **Subjects**: 9  
- **Classes**: Right hand, left hand, feet, tongue  
- **Preprocessing**: Downsampled to 128 Hz, band-pass filtered (4-38 Hz).  

For more details, refer to the [dataset documentation](https://www.bbci.de/competition/iv/).

---

## Usage
Coming soon!


