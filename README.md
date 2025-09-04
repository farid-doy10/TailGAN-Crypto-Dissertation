# TailGAN-Crypto-Dissertation
MSc Dissertation project on generating synthetic cryptocurrency return series using Tail-GAN and Conditional GAN, with a focus on Value-at-Risk (VaR), Expected Shortfall (ES), skewness, and kurtosis alignment

This repository contains the implementation of **Tail-GAN** and **Conditional GAN (C-GAN)** models used in my MSc Dissertation at The University of Sheffield (2025).  

The project focuses on generating **synthetic intraday cryptocurrency return sequences** that preserve **tail risk metrics** such as Value-at-Risk (VaR), Expected Shortfall (ES), skewness, and kurtosis.  
The models are trained on high-frequency data for three assets (BTC, ETH, LTC) and evaluated against real returns to ensure risk alignment.  

---

## 📂 Repository Contents  
- `dissertation.ipynb` — Jupyter Notebook with preprocessing, model training (Tail-GAN / C-GAN), and evaluation (VaR, ES, skewness, kurtosis).  
- `ref_price.pkl` — sample reference price series for quick experiments.  
- `ref_log_return.pkl` — corresponding log-returns used in the notebook.  
- `README.md` — project description and usage instructions.  

> 🔎 Note: Replace the `.pkl` files with your own dataset if you want to test on different assets.  

---

## ⚙️ Requirements  
- Python 3.9+  
- NumPy, Pandas, Matplotlib, SciPy  
- PyTorch  
- Jupyter Lab / Notebook  

You can install dependencies with:  
bash
pip install -r requirements.txt
## 🚀 How to Run
Option A — Local Jupyter
Clone the repository:

bash
Copy code
git clone https://github.com/farid-doy10/TailGAN-Crypto-Dissertation.git
cd TailGAN-Crypto-Dissertation
Launch Jupyter:

bash
Copy code
jupyter lab
Open dissertation.ipynb and run cells in order.
The notebook will automatically load ref_price.pkl and ref_log_return.pkl.

Option B — Google Colab
Open the notebook directly in Colab (adjust username/repo if needed):


## 🧪 Data Notes
Log-returns are used instead of raw prices for stationarity and heavy-tail properties.

You can load the reference files with:

python
Copy code
import pandas as pd
prices = pd.read_pickle("ref_price.pkl")
rets   = pd.read_pickle("ref_log_return.pkl")
Model outputs are 24×3 sequences → 24 hourly returns for 3 assets (BTC, ETH, LTC).

## 🧱 Model Overview
Tail-GAN
Generator: Temporal CNN (TCN) with dilated causal 1D convolutions + residual connections.

Discriminator: CNN with spectral normalization + minibatch standard deviation (diversity check).

Loss: adversarial + tail-aware penalties for VaR (pinball loss) and ES, plus skewness/kurtosis regularization.

C-GAN
Generator & Discriminator: Conditional MLPs, input = [noise + condition vector].

Conditions: asset identity, volatility state.

Loss: standard GAN (BCE) + penalties for VaR/ES (quantile alignment).

## 📊 Results
Tail-GAN reproduced real-world ES and VaR more accurately → risk-aware synthetic returns.

C-GAN captured fat tails under conditional settings but less precise at deep quantiles.

Baseline GAN underestimated tail risk and produced unstable skewness/kurtosis.

Example: On held-out test data, Tail-GAN matched 5% ES within ±0.2% of real values, while baseline GAN underestimated ES by more than 1%.

## 📖 Citation
If you use this code, please cite:

Risk-Aware Synthetic Cryptocurrency Return Generation using Tail-GAN and Conditional GAN (2025), University of Sheffield.

## 🙏 Acknowledgments
Supervisor: Dr. Tahsinur Khan

ICAIF’24 Crypto Market Simulation dataset for source data

Tail-GAN (2022) paper for foundational architecture inspiration
Supervisor: Dr. Tahsinur Khan

ICAIF’24 Crypto Market Simulation dataset for source data

Tail-GAN (2022) paper for foundational architecture inspiration
