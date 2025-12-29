# AI Crop Assistant - Brief Report
Date: 2025-11-19 22:58:19.427246

## Summary
- Provides crop disease detection and crop recommendation services

---

# AI Crop Assistant - Detailed Report
Date: 2025-11-19 22:58:19.429245

## Project Overview
This project uses a CNN for disease detection and a RandomForest classifier for crop recommendation.

## Crop Recommendation Model
- Precision per class:
  - apple: 1.000
  - banana: 1.000
  - blackgram: 1.000
  - chickpea: 1.000
  - coconut: 1.000
  - coffee: 1.000
  - cotton: 1.000
  - grapes: 1.000
  - jute: 0.980
  - kidneybeans: 1.000
  - lentil: 0.990
  - maize: 1.000
  - mango: 1.000
  - mothbeans: 1.000
  - mungbean: 1.000
  - muskmelon: 1.000
  - orange: 1.000
  - papaya: 1.000
  - pigeonpeas: 1.000
  - pomegranate: 1.000
  - rice: 1.000
  - watermelon: 1.000
![Crop Precision](reports\precision_crop.png)

## Disease Detection Model
- Precision per class:
  - Healthy: 0.870
  - Powdery: 0.857
  - Rust: 1.000
![Disease Precision](reports\precision_disease.png)

## Recommendations & Next Steps
- Collect more labeled data to improve precision across classes.
- Add monitoring to capture `Last Analysis` and `Registered Users` metrics in the dashboard.