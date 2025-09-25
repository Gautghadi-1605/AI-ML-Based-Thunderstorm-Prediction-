# AI-ML-Based-Thunderstorm-Prediction-
# AI-ML-Based Thunderstorm Prediction

An advanced **weather prediction system** designed to enhance aviation safety by delivering **accurate, real-time forecasts of thunderstorms and gale-force winds**. The project combines **machine learning**, **deep neural networks**, and **3D visualization** to provide actionable insights for pilots and aviation personnel.

---

## Project Overview

- **Data Analysis:**  
  Integrated **Numerical Weather Prediction (NWP) models** with **logistic regression** to analyze historical weather data and identify critical storm patterns.  

- **Machine Learning Models:**  
  **Deep Neural Networks (DNNs)** capture complex relationships in large-scale datasets to improve predictive accuracy. Hyperparameter tuning ensures optimal model performance.  

- **3D Visualization:**  
  Implemented **3D radar visualizations** using **Unity**, allowing users to intuitively assess storm severity, movement, and impact in real time.  

- **Purpose:**  
  Multi-faceted AI/ML solution for **aviation safety**, enhancing meteorological forecasting and aiding in real-time decision-making.

---

## Project Structure

C:\thunder
│
├─ Assets/ # Unity project assets for 3D radar visualization
├─ ProjectSettings/ # Unity project settings
├─ archive (8)/ # Large CSV files tracked with Git LFS
│ └─ merged_weather_data.csv
├─ scripts/ # Python scripts
│ ├─ test.py # Logistic Regression + DNN with hyperparameter tuning
│ ├─ bla.py
│ ├─ ra.py
│ ├─ download_era5_demo.py
├─ weather_3Ddd.csv # Dataset CSV
├─ weather_3d.csv # Dataset CSV
├─ weather_3d_map.csv # Dataset CSV
├─ README.md # This file
└─ .gitattributes # Git LFS configuration

## Getting Started

1. **Clone the repository**
```bash
git clone https://github.com/Gautghadi-1605/AI-ML-Based-Thunderstorm-Prediction-.git
Install Python dependencies

bash
Copy code
pip install -r requirements.txt


Open Unity Project

Open the Assets folder in Unity to view and interact with 3D radar visualizations of weather data.

Notes
Large CSV files are tracked with Git LFS. To access them after cloning:

bash
Copy code
git lfs install
git lfs pull
test.py contains the main predictive models (logistic regression and DNN with hyperparameter tuning).

Unity visualizations provide real-time 3D storm assessment, making complex meteorological data intuitive and actionable.


