🏠 House Price Prediction in Bangladesh 🇧🇩

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Data](https://img.shields.io/badge/dataset-2010--2022-orange)
![Status](https://img.shields.io/badge/status-active-brightgreen)

This project provides a machine learning solution for predicting house prices in Bangladesh based on various property features. The model is built using a Random Forest Regressor and includes data preprocessing, feature engineering, and model evaluation components.

[Sample](![image](https://github.com/user-attachments/assets/a0369cab-9823-433c-8454-b6ebe45ee0d9)
)
✨ Features

- **🧹 Data Cleaning**: Handles missing values, duplicates, and converts numeric fields
- **⚙️ Feature Engineering**: Extracts area information and identifies commercial properties
- **🤖 Machine Learning Model**: Random Forest Regressor for price prediction
- **🛡️ Error Handling**: Includes input validation with suggestions for similar city/area names
- **📊 Visualization**: Provides insightful plots of price distributions and feature importance
- **💻 Interactive CLI**: Allows users to compare their expected price with model predictions

📂 Dataset

The model uses a dataset (`house_price_bd.csv`) containing property listings with the following features:
- 💰 Price_in_taka
- 🛏️ Bedrooms
- 🚿 Bathrooms
- � Floor_no
- 📏 Floor_area
- 🏙️ City
- 📍 Location
- 🏷️ Title
- 🚪 Occupancy_status

🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/house-price-prediction-bd.git](https://github.com/Souad-Hasan/House-Price-Prediction-in-Bangladesh-)
   cd house-price-prediction-bd
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

 🚀 Usage

Run the script directly:
```bash
python house_price_predictor.py
```

The program will:
1. 📥 Load and preprocess the data
2. 🏋️ Train the machine learning model
3. 📈 Show model performance metrics
4. 📊 Display visualizations
5. 💬 Enter an interactive CLI for price comparisons

 📊 Model Performance

The model provides two key metrics:
- **📉 Mean Absolute Error (MAE)**: Shows the average absolute difference between predicted and actual prices
- **📈 R-squared Score**: Indicates how well the model explains the variance in prices

 💡 Example CLI Interaction

```
🔢 Enter property details to compare your price with the model prediction.

💰 Expected Price (in Taka): 50,00,000
🛏️ Number of Bedrooms: 3
🚿 Number of Bathrooms: 2
🏢 Floor Number: 5
📏 Floor Area (sq ft): 1200
🏙️ City (e.g., dhaka): dhaka
📍 Area (e.g., Dhanmondi): Dhanmondi
🏪 Is Commercial Property? (0 for No, 1 for Yes): 0

📊 Model Predicted Price: ৳4,850,000.00
💰 Your Provided Price: ৳5,000,000.00
⚠️ Overpriced by ৳150,000.00 (3.09%)
```

 📋 Requirements

- 🐍 Python 3.7+
- 🐼 pandas
- 🔢 numpy
- 🤖 scikit-learn
- 📊 matplotlib
- 🌊 seaborn

 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Souad-Hasan/House-Price-Prediction-in-Bangladesh-/blob/725e4ce32eb5a2a9614f7c8c1fdd1932511726b4/LICENSE.txt) file for details.

 🤝 Contribution

Contributions are welcome! 🙌 Please fork the repository and create a pull request with your improvements.
