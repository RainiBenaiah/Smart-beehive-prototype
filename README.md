# Smart-beehive-prototype
##  Project Demo Video
Watch the demo here: [Google Drive Video Link](https://drive.google.com/file/d/1fiiZjkyL7J7GnUEnK0Vpof61AzQB9jeN/view?usp=sharing)

##  Front-End Design
View the front-end design on Canva: [Canva Front-End Link](https://www.canva.com/design/DAG1AuAIdBI/HZzYklmMAoe1Nqn1pL1tSA/edit?utm_content=DAG1AuAIdBI&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)


# AsaliAsPossible: Intelligent Beehive Monitoring and Management System

## Project Overview

AsaliAsPossible is a comprehensive machine learning framework for beehive health monitoring, predictive analytics, and reinforcement learning-based colony management. The system integrates multi-modal sensor data, environmental measurements, and acoustic behavioral annotations to provide actionable insights for precision apiculture.

**Key Capabilities:**
- Queen status classification (98.96% accuracy)
- Sensor anomaly detection
- Time-series forecasting for environmental parameters
- Reinforcement learning framework for automated colony management decisions

---

## Table of Contents

1. [Installation](#installation)
2. [Dataset Description](#dataset-description)
3. [Data Pipeline](#data-pipeline)
4. [Machine Learning Models](#machine-learning-models)
5. [Reinforcement Learning Framework](#reinforcement-learning-framework)
6. [Usage Examples](#usage-examples)
7. [Results Summary](#results-summary)
8. [Pretrained Models and Transfer Learning](#pretrained-models-and-transfer-learning-recommendations)
9. [Future Work](#future-work)
10. [Contributing](#contributing)
11. [Contact](#contact)

---

## Installation

### Prerequisites

```bash
Python >= 3.8
CUDA >= 11.0 (optional, for GPU acceleration)
```

### Dependencies

```bash
pip install pandas numpy scikit-learn xgboost
pip install tensorflow keras
pip install statsmodels
pip install matplotlib seaborn plotly
```

### Quick Start

```bash
git clone https://github.com/yourusername/asaliaspossible.git
cd asaliaspossible
pip install -r requirements.txt
```

---

## Dataset Description

### Data Sources

The project integrates seven datasets from public Kaggle repositories spanning 2017-2022:

| Dataset | Records | Features | Temporal Coverage | Source |
|---------|---------|----------|-------------------|--------|
| temperature_2017 | 401,869 | 2 | Jan-Dec 2017 | [Bee Hive Metrics](https://www.kaggle.com/datasets/se18m502/bee-hive-metrics/data) |
| humidity_2017 | 8,737 | 2 | Jan-Dec 2017 | [Bee Hive Metrics](https://www.kaggle.com/datasets/se18m502/bee-hive-metrics/data) |
| weight_2017 | 524,110 | 2 | Jan-Dec 2017 | [Bee Hive Metrics](https://www.kaggle.com/datasets/se18m502/bee-hive-metrics/data) |
| flow_2017 | 1,048,220 | 2 | Jan-Dec 2017 | [Bee Hive Metrics](https://www.kaggle.com/datasets/se18m502/bee-hive-metrics/data) |
| Hive17 | 1,847 | 10 | Aug-Nov 2021 | [Beehives Dataset](https://www.kaggle.com/datasets/vivovinco/beehives?select=Hive17.csv) |
| all_data_updated | 1,275 | 23 | Jun-Jul 2022 | [Beehive Sounds](https://www.kaggle.com/datasets/annajyang/beehive-sounds/data) |
| train | 1,275 | 2 | Jun-Jul 2022 | [Beehive Sounds](https://www.kaggle.com/datasets/annajyang/beehive-sounds/data) |

### Geographical Setting

Primary data collection location: Santa Clara County, California, USA (37.29°N, 121.95°W)

### Data Characteristics

**Sensor Modalities:**
- Temperature (internal hive, ambient)
- Relative humidity (internal, ambient)
- Weight (total hive mass as proxy for honey stores)
- Flow (directional bee traffic measurement)
- Atmospheric pressure
- Wind speed and direction
- Cloud coverage and precipitation

**Behavioral Labels:**
- Queen presence (binary)
- Queen acceptance (3 classes)
- Queen status (4 classes: present and accepted, present and rejected, present or original, not present)
- Audio file associations

**Temporal Granularity:**
- 2017 sensors: Sub-minute to hourly intervals
- 2021 Hive17: Hourly measurements
- 2022 audio annotations: Event-based (irregular intervals)

---

## Data Pipeline

### Stage 1: Data Loading and Profiling

```python
from src.data_processing.load_datasets import load_all_datasets

datasets = load_all_datasets(data_dir='data/raw/')
# Automated profiling: shape, dtypes, missing values, datetime detection
```

**Output:**
- Comprehensive data quality report
- Identified datetime columns with automatic parsing
- Missing value statistics
- Distribution summaries

### Stage 2: Sensor Integration (2017 Data)

```python
from src.data_processing.data_integration import integrate_sensors_2017

integrated_sensors = integrate_sensors_2017(
    temperature_df,
    humidity_df,
    weight_df,
    flow_df,
    resample_freq='1min'
)
```

**Process:**
1. Timestamp alignment across heterogeneous sampling rates
2. Outer join preserving all temporal observations
3. Missing data imputation: forward fill (60min) → backward fill (60min) → linear interpolation (120min)
4. Final output: 524,175 complete records with 4 sensor modalities

### Stage 3: Feature Engineering

#### 2017 Sensor Features (4 → 129 dimensions)

```python
from src.data_processing.feature_engineering import engineer_sensor_features

engineered_sensors = engineer_sensor_features(integrated_sensors)
```

**Feature Categories:**
- **Temporal (13):** Hour, day, month, season, cyclical encodings
- **Rolling Windows (64):** 15min/1hr/6hr/24hr statistics (mean, std, min, max)
- **Derivatives (16):** Rate of change at multiple time scales
- **Lag Features (16):** Historical values at 15min/1hr/6hr/24hr offsets
- **Interactions (11):** Cross-sensor products, velocity measurements
- **Domain-Specific (5):** Temperature deviation from 35°C, humidity comfort zones, daily weight change

#### Audio Dataset Features (23 → 73 dimensions)

```python
from src.data_processing.feature_engineering import engineer_audio_features

engineered_audio = engineer_audio_features(all_data_updated)
```

**Feature Categories:**
- **Temporal (10):** Time-based features with cyclical encoding
- **Temperature (6):** Hive-ambient differentials, deviations from optimal
- **Humidity (4):** Differentials, comfort zone indicators
- **Pressure (2):** Differential and volatility metrics
- **Weather (4):** Wind chill, heat index, ventilation effects
- **Stress Indicators (2):** Environmental stress score, homeostasis quality
- **Interactions (3):** Temperature × humidity, ventilation effects
- **Group Aggregations:** Per-hive statistics and deviation features

### Stage 4: Train-Validation-Test Splitting

#### Time-Series (Temporal Split)

```python
from src.data_processing.train_test_split import temporal_split

train, val, test = temporal_split(
    engineered_sensors,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15
)
```

**Split Details:**
- Train: 366,922 samples (Jan 1 - Sep 13, 2017)
- Validation: 78,626 samples (Sep 13 - Nov 6, 2017)
- Test: 78,627 samples (Nov 6 - Dec 31, 2017)

#### Classification (Stratified Split)

```python
from src.data_processing.train_test_split import stratified_split

X_train, y_train, X_val, y_val, X_test, y_test = stratified_split(
    engineered_audio,
    target_column='queen status',
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42
)
```

**Class Distribution Preservation:**
- Train: 892 samples (70%)
- Validation: 191 samples (15%)
- Test: 192 samples (15%)
- Stratification verified within 0.3% across all classes

---

## Machine Learning Models

### Classification: Queen Status Detection

#### XGBoost (Best Performer)

```python
from src.models.classifiers import train_xgboost

model = train_xgboost(
    X_train, y_train,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8
)
```

**Performance:**
- Test Accuracy: 98.96%
- Test F1-Score: 98.96%
- Training Time: 0.40 seconds
- Overfitting Gap: 1.04% (excellent generalization)

**Feature Importance (Top 5):**
1. day_of_year (18.9%)
2. day_sin (14.3%)
3. day_of_month (11.3%)
4. day_cos (8.8%)
5. week_of_year (7.5%)

#### Model Comparison

| Model | Test Accuracy | Test F1 | Training Time |
|-------|--------------|---------|---------------|
| **XGBoost** | **0.990** | **0.990** | 0.40s |
| SVM | 0.979 | 0.979 | 0.04s |
| Random Forest | 0.974 | 0.974 | 0.27s |
| Gradient Boosting | 0.974 | 0.974 | 4.63s |
| Logistic Regression | 0.958 | 0.959 | 0.08s |
| KNN | 0.901 | 0.895 | 0.00s |

### Time-Series Forecasting: Temperature Prediction

```python
from src.models.time_series import train_lstm_forecast

model = train_lstm_forecast(
    X_train_seq, y_train,
    lookback=24,
    lstm_units=[50, 50],
    dropout=0.2,
    epochs=20,
    batch_size=32
)
```

**Critical Finding:** Persistence baseline outperformed all models

| Model | MAE (°C) | RMSE (°C) |
|-------|----------|-----------|
| **Persistence** | **0.291** | **0.548** |
| ARIMA(5,1,2) | 0.379 | 1.087 |
| LSTM | 1.125 | 1.424 |

**Interpretation:** Extreme autocorrelation due to biological thermoregulation. Bees maintain stable brood temperature (~35°C), making previous observation optimal predictor. Suggests forecasting efforts should focus on variables with higher intrinsic variability (weight, flow).

### Anomaly Detection: Sensor Malfunction

```python
from src.models.anomaly_detection import train_isolation_forest

anomaly_detector = train_isolation_forest(
    X_train,
    n_estimators=100,
    contamination=0.1,
    max_samples='auto'
)

anomaly_scores = anomaly_detector.score_samples(X_test)
anomalies = anomaly_detector.predict(X_test)
```

**Results:**

| Method | Anomaly Rate | Interpretation |
|--------|--------------|----------------|
| **Isolation Forest** | **8.59%** | Plausible (sensor errors + extreme weather) |
| One-Class SVM | 100.00% | Distribution shift (seasonal change) |
| Z-Score | 99.82% | Non-Gaussian distribution |

**Key Insight:** 99.8% anomaly rate in Z-Score method reveals severe distribution shift between training (spring/summer) and test (late autumn/winter) periods. Requires seasonal stratification in training data.

---

## Reinforcement Learning Framework

### Environment Design

**State Space (25 dimensions):**
- Sensor readings (4): temperature, humidity, weight, flow
- Sensor statistics (4): 24hr mean/std for temp and humidity
- Environmental context (5): hour, day, season, weather conditions
- Health indicators (5): queen status prediction, optimal range deviations
- Anomaly scores (3): isolation forest score, binary flags
- Historical context (4): days since inspection/harvest, cumulative production

**Action Space (12 discrete actions):**

| Action | Description | Time | Risk | Prerequisites |
|--------|-------------|------|------|---------------|
| 0 | NO_ACTION | 0h | low | - |
| 1 | INSPECT_HIVE | 0.5h | low | - |
| 2 | ADD_HONEY_SUPER | 1h | low | - |
| 3 | HARVEST_HONEY | 2h | medium | weight > threshold |
| 4 | REQUEEN | 1h | high | queen_absent |
| 5 | FEED_SYRUP | 0.5h | low | weight < threshold |
| 6 | TREAT_VARROA | 1h | medium | - |
| 7 | IMPROVE_VENTILATION | 0.25h | low | temp/humidity high |
| 8 | SPLIT_COLONY | 2h | high | strong colony |
| 9 | RELOCATE_HIVE | 4h | very high | poor conditions |
| 10 | COMBINE_COLONIES | 1.5h | medium | weak colony |
| 11 | EMERGENCY_INTERVENTION | 3h | very high | critical anomalies |

**Reward Function:**

```python
def compute_reward(state, action, next_state):
    reward = 0
    
    # Survival (fundamental)
    reward += 1.0  # per day alive
    if colony_death:
        reward -= 1000.0
    
    # Queen status
    if queen_present:
        reward += 5.0
    if queen_absent and detected_early:
        reward += 10.0
    if successful_requeening:
        reward += 100.0
    
    # Productivity
    reward += 20.0 * honey_harvested_kg
    reward += 2.0 * colony_growth
    
    # Health maintenance
    if temp_in_optimal_range:
        reward += 2.0
    if anomaly_resolved:
        reward += 15.0
    
    # Operational efficiency
    if unnecessary_inspection:
        reward -= 5.0
    reward -= action_cost
    
    # Seasonal bonuses
    if winter_survival:
        reward += 200.0
    
    return reward
```

### Recommended Algorithm: Proximal Policy Optimization (PPO)

```python
from src.models.rl_agent import PPOAgent

agent = PPOAgent(
    state_dim=25,
    action_dim=12,
    hidden_layers=[256, 256],
    learning_rate=3e-4,
    gamma=0.99,
    epsilon_clip=0.2,
    value_coef=0.5,
    entropy_coef=0.01
)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    
    while not done:
        action, log_prob, value = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.store_transition(state, action, reward, log_prob, value)
        state = next_state
        episode_reward += reward
    
    agent.update()
```

**Training Strategy:**

1. **Behavior Cloning (Phase 1):** Pre-train on expert demonstrations or supervised model predictions
2. **Offline RL (Phase 2):** Train on historical sensor data using Conservative Q-Learning
3. **Online Fine-Tuning (Phase 3):** Deploy in real apiaries and collect live feedback

---

## Usage Examples

### Complete Pipeline Example

```python
# 1. Load and integrate data
from src.data_processing import load_all_datasets, integrate_sensors_2017

datasets = load_all_datasets('data/raw/')
integrated = integrate_sensors_2017(
    datasets['temperature_2017'],
    datasets['humidity_2017'],
    datasets['weight_2017'],
    datasets['flow_2017']
)

# 2. Engineer features
from src.data_processing.feature_engineering import engineer_sensor_features

engineered = engineer_sensor_features(integrated)

# 3. Split data
from src.data_processing.train_test_split import temporal_split

train, val, test = temporal_split(engineered, train_ratio=0.7)

# 4. Train model
from src.models.classifiers import train_xgboost

model = train_xgboost(
    train.drop('target', axis=1),
    train['target']
)

# 5. Evaluate
from src.evaluation.metrics import evaluate_classifier

metrics = evaluate_classifier(model, test.drop('target', axis=1), test['target'])
print(f"Test Accuracy: {metrics['accuracy']:.4f}")
print(f"Test F1-Score: {metrics['f1']:.4f}")

# 6. Save model
import joblib
joblib.dump(model, 'models/xgboost_queen_status.pkl')
```

### Real-Time Inference

```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load('models/xgboost_queen_status.pkl')

# New sensor reading
new_data = pd.DataFrame({
    'hive_temp': [34.5],
    'hive_humidity': [45.2],
    'hive_pressure': [1010.3],
    'weather_temp': [22.1],
    # ... all 66 features
})

# Predict queen status
prediction = model.predict(new_data)
probability = model.predict_proba(new_data)

print(f"Predicted Queen Status: {prediction[0]}")
print(f"Confidence: {probability.max():.2%}")
```

### RL Agent Deployment

```python
from src.models.rl_agent import load_trained_agent

agent = load_trained_agent('models/ppo_beehive_agent.pth')

# Current hive state
current_state = extract_state_features(sensor_readings)

# Get recommended action
action = agent.select_action(current_state, deterministic=True)

print(f"Recommended Action: {action_names[action]}")
print(f"Expected Reward: {agent.get_value(current_state):.2f}")
```

---

## Results Summary

### Classification Performance

**Best Model:** XGBoost
- Test Accuracy: **98.96%**
- Test F1-Score: **98.96%**
- Training Time: 0.40 seconds
- Generalization Gap: 1.04%

**Key Insights:**
- Temporal features dominate importance (day_of_year: 18.9%)
- Near-perfect classification enables reliable RL state representation
- Minimal overfitting across all tree-based models

### Time-Series Forecasting

**Best Model:** Persistence (Naive Baseline)
- MAE: **0.291°C**
- RMSE: **0.548°C**

**Key Insights:**
- Biological thermoregulation creates extreme autocorrelation
- LSTM and ARIMA failed due to distribution shift (seasonal change)
- Temperature forecasting provides minimal value
- Recommendation: Focus on weight/flow prediction for actionable insights

### Anomaly Detection

**Best Model:** Isolation Forest
- Anomaly Rate: **8.59%**
- Training Time: 0.45 seconds

**Key Insights:**
- Z-Score 99.8% anomaly rate reveals severe seasonal distribution shift
- Requires training data spanning full annual cycle
- Detected anomalies align with sensor malfunction periods and extreme weather

### Data Engineering Outcomes

**2017 Sensor Integration:**
- Initial missingness: 44.6% → Final: 0%
- Integration: 4 heterogeneous time-series → 524,175 unified records
- Feature expansion: 4 → 129 dimensions

**Audio Dataset Enhancement:**
- Original: 23 features → Engineered: 73 dimensions
- Missing data: 3.4% → 0%
- Stratified splits maintain class balance within 0.3%

---

## Pretrained Models and Transfer Learning Recommendations Deadline October 19

### For Audio Analysis (Future Work)

**VGGish (Google)**
- Pretrained on AudioSet (2M YouTube videos)
- 128-dimensional embeddings ideal for beehive acoustics
- Usage: Extract audio features from WAV files for classification
- Repository: https://github.com/tensorflow/models/tree/master/research/audioset/vggish

**YAMNet (Yet Another Mobile Network)**
- Lightweight audio event detection
- 521 AudioSet classes including environmental sounds
- TensorFlow Hub integration for easy deployment
- Hub: https://tfhub.dev/google/yamnet/1

**OpenL3 (Open Look, Listen, and Learn)**
- Self-supervised audio-visual embeddings
- No labels required for fine-tuning
- Excellent for domain adaptation to beehive sounds
- Repository: https://github.com/marl/openl3

### For Time-Series (Current Task Enhancement)

**N-BEATS (Neural Basis Expansion Analysis)**
- SOTA univariate forecasting
- Interpretable architecture with trend/seasonality decomposition
- Outperforms statistical methods on M4 competition
- Paper: https://arxiv.org/abs/1905.10437

**Temporal Fusion Transformer (TFT)**
- Handles multi-horizon forecasting with covariates
- Attention mechanisms for interpretability
- Ideal for integrating weather + sensor data
- Paper: https://arxiv.org/abs/1912.09363

**TimeGPT (Nixtla)**
- Foundation model pretrained on 100B time-series data points
- Zero-shot forecasting capabilities
- API: https://docs.nixtla.io/

### For Reinforcement Learning

**Pretrained Vision Encoders (if using camera data):**
- ResNet-50 (ImageNet pretrained)
- EfficientNet-B0 (lightweight, high accuracy)
- CLIP (OpenAI) for zero-shot visual understanding

**RL Algorithm Implementations:**
- Stable Baselines3 (PPO, SAC, TD3 implementations)
  - Repository: https://github.com/DLR-RM/stable-baselines3
  - Includes pretrained locomotion agents for transfer learning
  
- RLlib (Ray) for distributed RL training
  - Documentation: https://docs.ray.io/en/latest/rllib/index.html
  - Scalable PPO implementation

**Transfer Learning Strategy:**

1. **Audio Domain:** Fine-tune VGGish on beehive sounds for queen detection
2. **Time-Series Domain:** Use N-BEATS for weight forecasting (honey production)
3. **RL Domain:** Initialize PPO policy with behavior cloning from XGBoost predictions

---

## Future Work Next Step

### Immediate Next Steps

1. **Fix Hive17 Parsing:** Re-load with `sep=';'` delimiter for proper column separation
2. **Seasonal Anomaly Detection:** Retrain models with winter data included
3. **Multi-Variate Forecasting:** Implement weight and flow prediction models
4. **Audio Feature Extraction:** Integrate VGGish embeddings for acoustic analysis

### Medium-Term Enhancements

1. **Multi-Modal Fusion:** Combine sensor, audio, and visual (camera) data streams
2. **Causal Inference:** Identify causal relationships between interventions and outcomes
3. **Explainable AI:** SHAP values for model interpretability and beekeeper trust
4. **Edge Deployment:** TensorFlow Lite conversion for on-device inference

### Long-Term Vision

1. **Federated Learning:** Train models across multiple apiaries while preserving privacy
2. **Digital Twin:** Real-time simulation environment for policy testing
3. **Multi-Agent RL:** Coordinate management across multiple hives
4. **Climate Adaptation:** Model colony responses to climate change scenarios

---

## Contributing

Contributions welcome! Please follow these guidelines:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

**Code Style:** PEP 8 for Python, docstrings required for all functions

**Testing:** All new models must include unit tests with >80% coverage

---

## Contact

**Project Maintainer:** Raini B Raini

**Email:** b.raini@alustudent.com

**Issues:** https://github.com/yourusername/asaliaspossible/issues

**Discussions:** https://github.com/yourusername/asaliaspossible/discussions

---

## Project Status

**Current Phase:** Offline RL Development

**Last Updated:** October 2024

**Version:** 1.0.0

---

## Acknowledgments

**Data Sources:**
- Bee Hive Metrics: se18m502 (Kaggle)
- Beehives Dataset: vivovinco (Kaggle)
- Beehive Sounds: Anna Yang (Kaggle)

**Inspiration:**
- Precision agriculture research community
- Open-source machine learning ecosystem
- Global beekeeping community fighting colony collapse disorder
