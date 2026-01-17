# AI Based Waste Classification with Grad-CAM

🇹🇷 **Türkçe** | 🇬🇧 **English**

---

## 🇬🇧 Overview

This project is an **AI powered waste classification system** that identifies common waste types (**plastic, paper, glass, metal**) from images or a live webcam feed.

Beyond simple classification, the system provides:

* **Model uncertainty estimation** using *Monte Carlo Dropout*
* **Explainable AI (XAI)** via *Grad-CAM visualizations*
* **Actionable environmental guidance** on proper disposal and recycling

The goal is to combine **computer vision**, **uncertainty-aware decision support**, and **environmental impact awareness** into a single, transparent system.

---

## 🇹🇷 Genel Bakış

Bu proje, görseller veya **canlı kamera** üzerinden atık türlerini (**plastik, kağıt, cam, metal**) sınıflandıran **yapay zekâ tabanlı** bir sistemdir.

Sadece tahmin yapmakla kalmaz, aynı zamanda:

* *Monte Carlo Dropout* ile **belirsizlik analizi**
* *Grad-CAM* ile **modelin neden o kararı verdiğini gösteren ısı haritaları**
* Atık türüne göre **nasıl geri dönüştürülmesi gerektiğine dair bilgilendirici metinler**

sunar. Amaç; **şeffaf, güvenilir ve çevresel farkındalık oluşturan** bir yapay zekâ sistemi geliştirmektir.

---

## Model Architecture

* **Backbone:** MobileNetV2
* **Framework:** PyTorch
* **Input Size:** 224 × 224 RGB images
* **Output:** 4 class softmax (glass, metal, paper, plastic)

MobileNetV2 was chosen for its **efficiency and deployability**, making the system suitable for real time and edge based applications.

---

##  Key Features

###  Waste Classification

* Image upload support (JPG / PNG)
* Real time webcam classification

### Uncertainty-Aware Prediction

* Monte Carlo Dropout during inference
* Confidence & uncertainty scores displayed to the user
* Enables safer decision making in ambiguous cases

### Explainable AI with Grad-CAM

* Visual explanation of model focus regions
* Adjustable heatmap intensity
* Improves transparency and trust

### Environmental Guidance

* Disposal & recycling instructions for each waste type
* Dual language (TR / EN) explanations

---

## User Interface

Built with **Streamlit** for rapid prototyping and accessibility:

* Tab based interface (Image / Live Camera)
* Start / Stop camera control
* Interactive sliders for Grad-CAM visualization

---

## How to Run

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## Project Structure

```
ai-waste-classification/
│── app.py              # Streamlit application
│── waste_model.pth     # Trained model weights
│── requirements.txt    # Dependencies
│── README.md           # Project documentation
│── .gitignore          # Ignored files
```

---

## Why This Project Matters

Improper waste disposal is a global environmental challenge. This project demonstrates how **AI can be used responsibly**, not only to classify data but also to:

* Quantify uncertainty
* Explain decisions
* Encourage sustainable behavior

Such systems are critical for deploying AI in **real world, high impact domains**.

---

## Academic & Research Value

This project integrates:

* Computer Vision
* Uncertainty Estimation
* Explainable AI (XAI)
* Human centered AI design

It is suitable for **research showcases, competitions, and university applications**, especially for institutions valuing interdisciplinary AI applications.

---

## License

This project is released for **educational and research purposes**.

## Research Motivation & Design Rationale

Most waste classification systems optimize for accuracy under the assumption that predictions will be blindly trusted.
However, in real world recycling scenarios, misclassification can actively reinforce incorrect disposal behavior.

This project explores a different design question:

Can explainability and uncertainty awareness influence how humans interact with AI predictions in environmentally sensitive decisions?

Instead of treating Grad-CAM as a debugging tool, it is used as a communication interface between the model and the user.
Monte Carlo Dropout is integrated not merely for uncertainty estimation, but to gate system behavior, signaling when human confirmation is required.

The system is intentionally lightweight to reflect real deployment constraints (edge devices, public kiosks), prioritizing interpretability, trust, and interaction over raw benchmark performance.
