# Practical No. 03: Autoencoder and Variational Autoencoder for Image Reconstruction and Generation

| | |
|---|---|
| **Name** | Andhale Sarthak Adinath |
| **PRN** | 202401110077 |
| **Batch** | AIML-A3 |
| **Department** | CSE-AIML |
| **Notebook** | `PRAC3_SARTHAK_ANDHALE_202401110077.ipynb` |

## Aim

To build an **Autoencoder (AE)** and a **Variational Autoencoder (VAE)** on the MNIST dataset for image reconstruction and generation, and to compare both models on the quality of their reconstructed and generated images.

## Dataset

**MNIST** handwritten digits (loaded directly via `keras.datasets.mnist`):

- 60,000 training images and 10,000 test images
- 28 × 28 grayscale images, 10 classes (digits 0–9)
- Pixel values normalized to the range 0–1, with a channel dimension added (28 × 28 × 1)
- Labels are used only for visualization, not for training

## Notebook Structure

| Part | Description |
|------|-------------|
| Steps 1–3 | Load MNIST, visualize samples, preprocess data |
| **Part A** (Steps 4–11) | Build, train and evaluate the Autoencoder |
| **Part B** (Steps 12–22) | Build, train and evaluate the VAE |
| **Part C** (Steps 23–24) | Generate new digit images from random latent vectors |
| **Part D** (Step 25) | Visualize the 2D VAE latent space |
| **Part E** (Steps 26–30) | Compare AE vs VAE (MSE, visual comparison, summary table) |

## Model Details

### Autoencoder
- **Encoder:** Conv2D(32) → Conv2D(64) → Flatten → Dense(128) → Dense(latent, 128-d)
- **Decoder:** Dense(7×7×64) → Reshape → Conv2DTranspose(64) → Conv2DTranspose(32) → Conv2DTranspose(1, sigmoid)
- **Latent dimension:** 128
- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Training:** 10 epochs, batch size 128 (test set used for validation)

### Variational Autoencoder
- **Encoder:** same convolutional backbone, outputs `z_mean` and `z_log_var`
- **Sampling:** reparameterization trick, `z = μ + exp(0.5 · log σ²) · ε`
- **Decoder:** same architecture as the AE decoder
- **Latent dimension:** 2 (so the latent space can be plotted)
- **Loss:** Reconstruction loss (binary cross-entropy) + KL divergence, implemented in a custom `train_step()`
- **Optimizer:** Adam
- **Training:** 10 epochs, batch size 128

## Requirements

- Python 3.8+
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook, JupyterLab or Google Colab

Install the dependencies with:

```bash
pip install tensorflow numpy matplotlib jupyter
```

## How to Run

1. Clone or download this repository.
2. Open the notebook:
   ```bash
   jupyter notebook PRAC3_SARTHAK_ANDHALE_202401110077.ipynb
   ```
   (or upload it to Google Colab).
3. Run all cells from top to bottom. MNIST is downloaded automatically on the first run.

## Results and Observations

- The **Autoencoder** reconstructs digits closely, since it directly minimizes reconstruction error.
- The **VAE** reconstructs digits slightly smoother or blurrier, because it also optimizes the KL divergence term, but it learns a structured 2D latent space.
- The VAE can **generate new digit images** by sampling random points from the latent space and passing them through the decoder. The Autoencoder cannot do this naturally.
- Reconstruction quality is compared using MSE (on the first 1000 test images) and side-by-side visual comparison.

### Autoencoder vs VAE

| Parameter | Autoencoder | VAE |
|-----------|-------------|-----|
| Main purpose | Reconstruction, feature learning | Reconstruction and generation |
| Encoder output | Fixed latent vector | Mean and log variance |
| Latent space | Deterministic | Probabilistic |
| Loss | Reconstruction loss | Reconstruction + KL divergence |
| Image generation | Limited | Strong |

## Conclusion

The Autoencoder is best suited for **reconstruction, compression and feature extraction**, while the VAE is best suited for **generative modeling and creating new images**, at the cost of slightly lower reconstruction sharpness.
