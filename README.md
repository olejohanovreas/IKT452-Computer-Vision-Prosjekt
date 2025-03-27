## MRI Tumor Classification Project - IKT452 Computer Vision

Welcome to my repository!

### Repository File Structure


- **config/**: Contains the `config.yaml` file, which holds all adjustable parameters for preprocessing, augmentation, feature extraction, and model training.  
- **data/**: Directory for storing MRI images.  
  - `raw/`: Original data.
    - `training/`: Training images.  
    - `testing/`: Testing images.  
- **results/**: Stores output files from the pipeline.  
- **src/**: Python modules that implement the various stages of the pipeline.  
  - `data_preprocessing.py`: Contains preprocessing logic.
  - `data_augmentation.py`: Contains augmentation logic.
  - `feature_extraction.py`: Implements HOG, LBP, Gabor, GLCM, and SIFT feature extraction.
  - `dimensionality_reduction.py`: Handles PCA for reducing feature dimensions.
  - `model_training.py`: Trains and evaluates classifiers (Random Forest, SVM, etc.).
- **pipeline.ipynb**: A Jupyter notebook that orchestrates the entire pipeline using the modules in `src/`. This is the main entrypoint of the codebase.
- **requirements.txt**: Lists Python dependencies for running this project.

---

## Getting Started

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/IKT452-Computer-Vision-Prosjekt.git
   cd IKT452-Computer-Vision-Prosjekt
   ```

2. **Set Up the Environment**  
   - Create and activate a virtual environment (optional but recommended):  
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
   - Install dependencies:  
     ```bash
     pip install -r requirements.txt
     ```

3. **Configure the Pipeline**  
   - Update parameters in `config/config.yaml` (e.g., data paths, augmentation settings, PCA components, classifier options).

4. **Run the Pipeline**  
   - Open `pipeline.ipynb` in Jupyter:  
     ```bash
     jupyter notebook pipeline.ipynb
     ```
   - Follow the notebook cells to execute each stage:
     - Preprocessing (cropping, resizing)
     - Augmentation (optional)
     - Feature Extraction (HOG, LBP, Gabor, etc.)
     - Dimensionality Reduction (PCA)
     - Model Training & Evaluation

5. **View Results**  
   - Check the `results/` folder for trained models (`.joblib` files) and output figures/logs.
