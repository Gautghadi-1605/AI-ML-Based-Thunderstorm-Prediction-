import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -----------------------
# 1. Synthetic dataset with lat/lon
# -----------------------
np.random.seed(42)
num_samples = 500  # smaller for fast run
feature_cols = ['pressure','humidity','temperature','wind_speed','wind_dir']
weather_classes = ['clear','cloudy','rain','snow']

data = {
    'pressure': np.random.randint(980, 1030, size=num_samples),
    'humidity': np.random.randint(30, 100, size=num_samples),
    'temperature': np.random.randint(-5, 35, size=num_samples),
    'wind_speed': np.random.randint(0, 20, size=num_samples),
    'wind_dir': np.random.randint(0, 360, size=num_samples),
    'weather': np.random.choice(weather_classes, size=num_samples),
    # Assign random lat/lon for visualization (US approx.)
    'latitude': np.random.uniform(24.0, 50.0, size=num_samples),
    'longitude': np.random.uniform(-125.0, -65.0, size=num_samples)
}
df = pd.DataFrame(data)

# -----------------------
# 2. Features & target
# -----------------------
X = df[feature_cols]
y = df['weather']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Impute & scale
X = SimpleImputer(strategy='mean').fit_transform(X)
X = StandardScaler().fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X, y_encoded, df, test_size=0.2, random_state=42
)

# -----------------------
# 3. Logistic Regression
# -----------------------
logreg = LogisticRegression(max_iter=300)
logreg.fit(X_train, y_train)
proba_logreg_orig = logreg.predict_proba(X_test)
y_pred_logreg = logreg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))

# -----------------------
# 4. DNN
# -----------------------
dnn = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(len(le.classes_), activation='softmax')
])
dnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dnn.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
proba_dnn = dnn.predict(X_test, verbose=0)

# -----------------------
# 5. Safe Ensemble (handle class mismatch)
# -----------------------
num_classes = len(le.classes_)
proba_logreg_full = np.zeros((X_test.shape[0], num_classes))
for idx, cls in enumerate(logreg.classes_):
    proba_logreg_full[:, cls] = proba_logreg_orig[:, idx]

# Ensemble probabilities
proba_ensemble = (proba_logreg_full + proba_dnn) / 2
y_pred_ensemble = np.argmax(proba_ensemble, axis=1)

# -----------------------
# 6. Evaluation
# -----------------------
all_labels = np.arange(num_classes)
print("\nEnsemble Accuracy:", accuracy_score(y_test, y_pred_ensemble))
print(classification_report(
    y_test,
    y_pred_ensemble,
    labels=all_labels,
    target_names=le.classes_,
    zero_division=0
))

# -----------------------
# 7. Prepare CSV for 3D Map/Blender
# -----------------------
df_visual = df_test[['latitude','longitude','pressure','temperature','humidity']].copy()

# Add predicted weather & intensity
df_visual['weather_class'] = y_pred_ensemble
# Intensity can be max probability of ensemble
df_visual['intensity'] = np.max(proba_ensemble, axis=1)

# Save CSV
df_visual.to_csv('C:/thunder/weather_3D_map.csv', index=False)
print("CSV saved for 3D visualization! Columns: latitude, longitude, pressure, temperature, humidity, weather_class, intensity")

















