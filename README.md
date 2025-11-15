# 🧬 BioImageKit — Biomedical Image Processing Toolkit

A comprehensive collection of **116+ biomedical image processing techniques** implemented in Python, organized into 15 specialized modules for medical imaging research and clinical applications.

## 📋 Overview

This repository provides a complete toolkit for biomedical image analysis using the **ChestMNIST dataset** (64×64 grayscale chest X-rays). Each technique is implemented in self-contained, executable cells with detailed visualizations and explanations.

### 🎯 Key Features

- **116 image processing techniques** across 15 modules
- **Self-contained implementations** - each technique runs independently
- **Professional visualizations** - matplotlib-based comprehensive plots
- **Medical imaging focus** - optimized for biomedical applications
- **ChestMNIST dataset** - 78,468 training images included
- **Production-ready code** - tested and documented

---

## 🗂️ Repository Structure

```
DL/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── SETUP.md                          # Installation & setup guide
├── USAGE.md                          # Detailed usage documentation
│
├── notebooks/
│   ├── 00_Setup_and_Helpers.ipynb           # Setup & utility functions
│   ├── 01_Foundation_Preprocessing.ipynb     # Basic preprocessing (6 techniques)
│   ├── 02_Denoising_Filtering.ipynb         # Noise removal (10 techniques)
│   ├── 03_Segmentation.ipynb                # Segmentation methods (12 techniques)
│   ├── 04_Edge_Detection.ipynb              # Edge & feature detection (11 techniques)
│   ├── 05_Texture_Analysis.ipynb            # Texture features (16 techniques)
│   ├── 06_Shape_Morphology.ipynb            # Morphological operations (15 techniques)
│   ├── 07_Registration_Motion.ipynb         # Image alignment (5 techniques)
│   ├── 08_Advanced_Transforms.ipynb         # Frequency & wavelet (7 techniques)
│   ├── 09_Vessel_Ridge.ipynb                # Vessel enhancement (5 techniques)
│   ├── 10_Quality_Assessment.ipynb          # Image quality metrics (4 techniques)
│   ├── 11_3D_Volume.ipynb                   # 3D processing (2 techniques)
│   ├── 12_Color_Multichannel.ipynb          # Color analysis (3 techniques)
│   ├── 13_Radiomics.ipynb                   # Quantitative features (2 techniques)
│   ├── 14_Advanced_Morphology.ipynb         # Extended morphology (5 techniques)
│   └── 15_Texture_Patterns.ipynb            # Advanced texture (5 techniques)
│
├── output/
│   └── Biomedical_image_processing.ipynb    # Original monolithic notebook
│
└── data/                              # Dataset (auto-downloaded)
```

## 📘 Notebook Descriptions

- `00_Setup_and_Helpers.ipynb`: Environment bootstrap and shared utilities. Installs/validates dependencies, configures the ChestMNIST dataset auto-download, and defines reusable helper functions like `_auto_get_sample_gray`, `_auto_get_sample_rgb`, `_to_hwc_uint8`, and `_ensure_rgb_uint8` used across all modules.

- `01_Foundation_Preprocessing.ipynb`: Core grayscale preprocessing. Covers histogram equalization, CLAHE, gamma correction, normalization, and contrast stretching with side-by-side visual comparisons and parameter sensitivity plots to establish strong baselines.

- `02_Denoising_Filtering.ipynb`: Classical and modern denoisers. Includes Gaussian/Median/Bilateral, Non-Local Means, Anisotropic Diffusion, Total Variation, Wiener, Guided, Morphological filters, and BM3D with PSNR/SSIM metrics and qualitative artifact analysis.

- `03_Segmentation.ipynb`: Thresholding and region-based segmentation. Implements Otsu/Multi-Otsu, adaptive methods, watershed, region growing, active contours (snakes), Chan–Vese, graph-cut, random walker, and superpixel pipelines (Felzenszwalb, SLIC, Quickshift) with evaluation overlays.

- `04_Edge_Detection.ipynb`: Gradient and feature edges. Demonstrates Sobel, Prewitt, LoG/DoG, Canny, Hough-based detection, and corner/keypoint methods (Harris, Shi–Tomasi, FAST, ORB, SIFT) with tuning guidance and robustness notes.

- `05_Texture_Analysis.ipynb`: Statistical and filter-bank texture features. GLCM (Haralick), LBP variants, Gabor/Laws/wavelet features, plus advanced sets like Local Ternary Patterns, fractal dimension, auto-correlation, extended GLCM, and Tamura descriptors.

- `06_Shape_Morphology.ipynb`: Binary morphology and shape analysis. Erosion/dilation/opening/closing, skeletonization, distance transform, convex hull, plus attribute filtering and connected-components summaries for region statistics.

- `07_Registration_Motion.ipynb`: Alignment and motion estimation. Phase correlation, keypoint-based registration, optical flow (Lucas–Kanade), Demons-like approaches, and mutual information—includes before/after alignment metrics and visual checks.

- `08_Advanced_Transforms.ipynb`: Frequency and multi-resolution analysis. FFT diagnostics, multi-level wavelets, Radon/Hough transforms, log-Gabor and steerable filters, and Riesz transform for orientation-selective processing.

- `09_Vessel_Ridge.ipynb`: Tubular and ridge structure enhancement. Frangi, Hessian, Meijering, Sato, and multi-scale vesselness; ideal for vasculature-like patterns with thresholding and post-processing recipes.

- `10_Quality_Assessment.ipynb`: Perceptual and signal-based quality metrics. SSIM, PSNR, blur/sharpness indicators, and utility checks to assess preprocessing impact and detect low-quality inputs.

- `11_3D_Volume.ipynb`: Minimal 3D extensions. Example multi-slice handling, simple volume visualizations, and considerations for adapting 2D pipelines to volumetric data.

- `12_Color_Multichannel.ipynb`: Beyond single-channel images. Color space conversions, per-channel operations, and basic color deconvolution for stain/separation-style analyses.

- `13_Radiomics.ipynb`: Quantitative imaging biomarkers. Efficient extraction of first-order, shape, and texture features with optimization strategies for faster batch processing.

- `14_Advanced_Morphology.ipynb`: Extended morphology set. Rolling ball background subtraction, morphological reconstruction, area opening/closing, attribute filters, and advanced connected-component filtering with distributions.

- `15_Texture_Patterns.ipynb`: Advanced texture and pattern measures. Local Ternary Patterns, fractal dimension (box/differential), spatial auto-correlation, extended GLCM metrics, and Tamura features with interpretation aids.

Each notebook is self-contained, includes clear visualizations, and references `00_Setup_and_Helpers.ipynb` for common utilities. Run the setup notebook first for a smooth experience.

---

---

## 📦 Module Breakdown

### **MODULE 1: Foundation & Preprocessing** (6 techniques)
- Histogram Equalization
- CLAHE (Contrast Limited AHE)
- Adaptive Histogram Equalization
- Gamma Correction
- Image Normalization
- Contrast Stretching

### **MODULE 2: Denoising & Filtering** (10 techniques)
- Gaussian Filtering
- Median Filtering
- Bilateral Filtering
- Non-Local Means
- Anisotropic Diffusion
- Total Variation Denoising
- Wiener Filtering
- Morphological Filtering
- Guided Filtering
- BM3D Denoising

### **MODULE 3: Segmentation & Region Analysis** (12 techniques)
- Otsu Thresholding
- Multi-Otsu
- Adaptive Thresholding
- Watershed Segmentation
- Region Growing
- Active Contours (Snakes)
- Chan-Vese Segmentation
- Graph Cut Segmentation
- Random Walker
- Felzenszwalb Segmentation
- SLIC Superpixels
- Quickshift Superpixels

### **MODULE 4: Edge & Feature Detection** (11 techniques)
- Sobel Edge Detection
- Canny Edge Detection
- Prewitt Filter
- Laplacian of Gaussian (LoG)
- Difference of Gaussians (DoG)
- Corner Detection (Harris)
- Shi-Tomasi Corner Detection
- FAST Keypoint Detection
- ORB Features
- SIFT (via OpenCV)
- Hough Transform

### **MODULE 5: Texture Analysis** (16 techniques)
- Gray Level Co-occurrence Matrix (GLCM)
- Local Binary Patterns (LBP)
- Gabor Filters
- Laws Texture Energy
- Wavelet Texture Features
- Local Ternary Patterns (LTP)
- Fractal Dimension
- Auto-correlation Analysis
- Extended GLCM Features
- Tamura Texture Features
- And more...

### **MODULE 6: Shape & Morphology** (15 techniques)
- Morphological Operations (erosion, dilation, opening, closing)
- Skeletonization
- Distance Transform
- Convex Hull
- Morphological Reconstruction
- Area Opening/Closing
- Attribute Filters
- Connected Component Filtering
- Rolling Ball Background Subtraction
- And more...

### **MODULE 7: Registration & Motion** (5 techniques)
- Phase Correlation Registration
- Feature-based Registration
- Optical Flow (Lucas-Kanade)
- Demons Registration
- Mutual Information Registration

### **MODULE 8: Advanced Transforms** (7 techniques)
- Fourier Transform Analysis
- Wavelet Transform (Multi-level)
- Radon Transform
- Hough Transform
- Log-Gabor Filters
- Steerable Filters
- Riesz Transform

### **MODULE 9: Vessel & Ridge Enhancement** (5 techniques)
- Frangi Vesselness Filter
- Hessian-based Ridge Detection
- Meijering Neurite Filter
- Sato Tubeness Filter
- Multi-scale Vessel Enhancement

### **MODULE 10: Quality & Assessment** (4 techniques)
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)
- Image Quality Metrics
- Blur Detection

### **MODULE 11: 3D & Volume Processing** (2 techniques)
- Volume Rendering Simulation
- Multi-slice Analysis

### **MODULE 12: Color & Multi-channel** (3 techniques)
- Color Space Conversions
- Multi-channel Processing
- Color Deconvolution

### **MODULE 13: Radiomics & Quantification** (2 techniques)
- Radiomics Feature Extraction
- Shape & Intensity Features

### **MODULE 14: Advanced Morphological Operations** (5 techniques)
- Rolling Ball Background Subtraction
- Morphological Reconstruction
- Area Opening/Closing
- Attribute Filters
- Connected Component Filtering

### **MODULE 15: Texture & Pattern Analysis** (5 techniques)
- Local Ternary Patterns (LTP)
- Fractal Dimension Analysis
- Auto-correlation Analysis
- Extended GLCM Features
- Tamura Texture Features

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/jugalmodi0111/BioImageKit.git
cd BioImageKit
```

### 2. Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run Notebooks
```bash
# Start Jupyter
jupyter notebook

# Open notebooks/00_Setup_and_Helpers.ipynb first
# Then explore any module notebook
```

---

## 📚 Dataset

**ChestMNIST** - A subset of the MedMNIST dataset:
- **Training**: 78,468 images
- **Validation**: 11,219 images  
- **Test**: 22,433 images
- **Size**: 64×64 pixels (grayscale)
- **Classes**: 14 thoracic pathologies
- **Auto-downloaded** on first run

---

## 🔧 Requirements

### Core Dependencies
- Python 3.8+
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- OpenCV-Python >= 4.6.0
- scikit-image >= 0.19.0
- scikit-learn >= 1.0.0
- SciPy >= 1.7.0
- PyTorch >= 1.12.0
- torchvision >= 0.13.0
- medmnist >= 2.0.0
- PyWavelets >= 1.3.0
- SimpleITK >= 2.1.0
- radiomics >= 3.0.0

See `requirements.txt` for complete list.

---

## 📖 Documentation

- **[SETUP.md](SETUP.md)** - Detailed installation instructions
- **[USAGE.md](USAGE.md)** - How to use each module
- **Inline Documentation** - Every cell has detailed comments

---

## 🎓 Use Cases

### Research Applications
- Medical image preprocessing pipelines
- Feature extraction for ML/DL models
- Quantitative imaging biomarkers
- Texture analysis studies
- Segmentation algorithm comparison

### Clinical Applications
- X-ray enhancement and analysis
- Pathology detection preprocessing
- Image quality assessment
- Automated measurement tools

### Educational
- Learning image processing concepts
- Understanding biomedical imaging
- Hands-on implementation examples

---

## 🏗️ Architecture

### Self-Contained Cells
Each technique is implemented as a standalone cell with:
1. **Auto-sample generation** - Automatic image loading
2. **Parameter configuration** - Adjustable settings
3. **Processing logic** - Core algorithm
4. **Visualization** - Multi-panel plots
5. **Metrics output** - Quantitative results

### Helper Functions
```python
_auto_get_sample_gray()  # Get grayscale sample with 4-tier fallback
_auto_get_sample_rgb()   # Get RGB sample with 4-tier fallback
_to_hwc_uint8()          # Convert tensor/PIL/array to HWC uint8
_ensure_rgb_uint8()      # Ensure RGB format
```

---

## 📊 Performance

- **Optimized implementations** - Vectorized NumPy operations
- **Memory efficient** - Minimal copies, in-place operations where safe
- **GPU acceleration** - PyTorch-based operations when available
- **Benchmarked** - Radiomics extraction ~3-5x faster after optimization

---

## 🤝 Contributing

Contributions welcome! Areas for contribution:
- Additional techniques
- Performance optimizations
- Bug fixes
- Documentation improvements
- New datasets support

### To Contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewTechnique`)
3. Commit changes (`git commit -m 'Add new technique'`)
4. Push to branch (`git push origin feature/NewTechnique`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **MedMNIST** - Dataset providers
- **scikit-image** - Core image processing library
- **PyRadiomics** - Radiomics feature extraction
- **OpenCV** - Computer vision algorithms
- **PyTorch** - Deep learning framework

---

## 📞 Contact

**Jugal Modi**  
GitHub: [@jugalmodi0111](https://github.com/jugalmodi0111)  
Repository: [BioImageKit](https://github.com/jugalmodi0111/BioImageKit)

---

## 📈 Project Stats

- **116 techniques** implemented
- **15 modules** organized
- **6,700+ lines** of code
- **119 cells** total
- **Professional visualizations** throughout
- **Production-ready** implementations

---

## 🗺️ Roadmap

### Planned Features
- [ ] Deep learning-based techniques (U-Net, ResNet features)
- [ ] 3D volume processing expansion
- [ ] Real-time processing pipeline
- [ ] Web interface for demos
- [ ] Docker containerization
- [ ] Performance benchmarking suite
- [ ] Additional datasets (CT, MRI, Ultrasound)

### Future Modules
- **Module 16**: Deep Learning Methods
- **Module 17**: Advanced Registration
- **Module 18**: Statistical Methods
- **Module 19**: Graph-Based Algorithms
- **Module 20**: Restoration & Reconstruction

---

## ⭐ Star History

If you find this repository useful, please consider giving it a star! ⭐

---

**Built with ❤️ for the biomedical imaging community**
