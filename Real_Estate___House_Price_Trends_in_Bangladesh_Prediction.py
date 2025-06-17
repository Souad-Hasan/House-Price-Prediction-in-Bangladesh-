import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import difflib
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('house_price_bd.csv')
df = df.drop_duplicates()

# Clean numeric fields
df['Price_in_taka'] = df['Price_in_taka'].str.replace('৳', '', regex=False).str.replace(',', '', regex=False).astype(float)
df['Floor_no'] = df['Floor_no'].astype(str).str.replace('th', '', regex=False).str.replace('8th', '8', regex=False)
df['Floor_no'] = pd.to_numeric(df['Floor_no'], errors='coerce')

# Handle missing values
df['Bedrooms'] = df['Bedrooms'].fillna(df['Bedrooms'].median())
df['Bathrooms'] = df['Bathrooms'].fillna(df['Bathrooms'].median())
df['Floor_no'] = df['Floor_no'].fillna(df['Floor_no'].median())

# Feature engineering
df['Area'] = df['Location'].astype(str).str.split(',').str[0].str.strip()
df['Is_Commercial'] = df['Title'].astype(str).str.contains('Commercial|Office|Shop', case=False).astype(int)

# Encode categorical features
label_encoders = {}
categorical_cols = ['City', 'Occupancy_status', 'Area']
for col in categorical_cols:
    df[col] = df[col].astype(str).fillna('Unknown')  # ensure string and no NaN
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature set
features = ['Bedrooms', 'Bathrooms', 'Floor_no', 'Floor_area', 'City', 'Area', 'Is_Commercial']
X = df[features]
y = df['Price_in_taka']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict function with validation
def predict_price(bedrooms, bathrooms, floor_no, floor_area, city, area, is_commercial):
    valid_cities = list(label_encoders['City'].classes_)
    valid_areas = list(label_encoders['Area'].classes_)

    if city not in valid_cities:
        suggestion = difflib.get_close_matches(city, valid_cities, n=1)
        suggestion_text = f" Did you mean: '{suggestion[0]}'?" if suggestion else ""
        raise ValueError(f"City '{city}' not found in training data.{suggestion_text}")

    if area not in valid_areas:
        suggestion = difflib.get_close_matches(area, valid_areas, n=1)
        suggestion_text = f" Did you mean: '{suggestion[0]}'?" if suggestion else ""
        raise ValueError(f"Area '{area}' not found in training data.{suggestion_text}")

    city_encoded = label_encoders['City'].transform([city])[0]
    area_encoded = label_encoders['Area'].transform([area])[0]
    input_data = np.array([[bedrooms, bathrooms, floor_no, floor_area, city_encoded, area_encoded, is_commercial]])
    return model.predict(input_data)[0]
# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📈 Model Performance:")
print(f"Mean Absolute Error: {mae:,.2f}")
print(f"R-squared Score: {r2:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📊 Feature Importance:")
print(feature_importance)

# Visualization
plt.figure(figsize=(15, 10))

# Price distribution
plt.subplot(2, 2, 1)
sns.histplot(df['Price_in_taka'], kde=True, color='skyblue')
plt.title('Price Distribution')
plt.xlabel('Price in Taka')
plt.ylabel('Count')

# Price vs Floor Area
plt.subplot(2, 2, 2)
sns.scatterplot(x='Floor_area', y='Price_in_taka', data=df, alpha=0.6)
plt.title('Price vs Floor Area')
plt.xlabel('Floor Area (sq ft)')
plt.ylabel('Price in Taka')

# Price by Bedrooms
plt.subplot(2, 2, 3)
sns.boxplot(x='Bedrooms', y='Price_in_taka', data=df)
plt.title('Price by Bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel('Price')

# Feature Importance Plot
plt.subplot(2, 2, 4)
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Feature Importance')

plt.tight_layout()
plt.show()

# CLI comparison
def compare_user_price():
    try:
        print("\n🔢 Enter property details to compare your price with the model prediction.\n")
        user_price = float(input("Expected Price (in Taka): ").replace(',', ''))
        bedrooms = int(input("Number of Bedrooms: "))
        bathrooms = int(input("Number of Bathrooms: "))
        floor_no = int(input("Floor Number: "))
        floor_area = float(input("Floor Area (sq ft): "))
        city = input("City (e.g., dhaka): ").strip().lower()
        area = input("Area (e.g., Dhanmondi): ").strip()
        is_commercial = int(input("Is Commercial Property? (0 for No, 1 for Yes): "))

        predicted = predict_price(bedrooms, bathrooms, floor_no, floor_area, city, area, is_commercial)

        diff = user_price - predicted
        percent = (diff / predicted) * 100

        print(f"\n📊 Model Predicted Price: ৳{predicted:,.2f}")
        print(f"💰 Your Provided Price: ৳{user_price:,.2f}")

        if diff > 0:
            print(f"⚠️ Overpriced by ৳{diff:,.2f} ({percent:.2f}%)")
        elif diff < 0:
            print(f"✅ Underpriced by ৳{-diff:,.2f} ({-percent:.2f}%)")
        else:
            print("✔️ Perfectly Priced!")

    except ValueError as ve:
        print(f"\n❌ Error: {ve}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

# CLI loop
if __name__ == "__main__":
    while True:
        compare_user_price()
        cont = input("\nWould you like to check another property? (y/n): ").lower()
        if cont != 'y':
            print("👋 Thank you for using the house price prediction CLI!")
            break
