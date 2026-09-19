# cargo-delivery-ml-insights
Comparative evaluation of 5 Machine Learning models on a shipping dataset, featuring diagnostic analysis of predictive limitations and high-variance data constraints

### Precision & Recall Trade-off
| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Decision Tree | 0.78 | 0.65 | 0.71 |
| Neural Network | 0.72 | 0.61 | 0.66 |
| Logistic Regression | 0.68 | 0.58 | 0.63 |
| Naive Bayes | 0.65 | 0.55 | 0.60 |
| KNN | 0.62 | 0.52 | 0.57 |

### ROC-AUC Scores
- **Decision Tree:** 0.75
- **Neural Network:** 0.72
- **Logistic Regression:** 0.70
- **Naive Bayes:** 0.68
- **KNN:** 0.65

---

## 🔍 Key Insights

### Feature Importance (Decision Tree Analysis)
1. **Weight of Package** - Primary predictor
2. **Discount Offered** - Strong secondary indicator
3. **Warehouse Block** - Moderate impact
4. **Mode of Shipment** - Categorical influence
5. **Customer Care Calls** - Engagement metric

### Why Decision Tree Performed Best
✅ Captures **non-linear relationships** between features  
✅ Handles **mixed data types** (numerical + categorical)  
✅ Provides **interpretable decision rules**  
✅ No scaling required (robust to feature magnitude)  
✅ Effective for **moderate-sized datasets** like ours

### Dataset Characteristics
- **Total Samples:** 10,999
- **Features:** 11 (7 numerical, 4 categorical)
- **Class Distribution:** 70% On-time, 30% Delayed (imbalanced)
- **Train-Test Split:** 80-20 (stratified to maintain class distribution)

### Challenges & Limitations
⚠️ **Modest accuracy (~68%)** indicates:
- Shipping outcomes influenced by **unmeasured external factors** (weather, customs delays, traffic)
- **Weak feature correlation** with target variable
- **High variance** in real-world logistics data
- Need for **feature engineering** and **domain expertise**

---

## 🛠️ Technology Stack

| Tool | Purpose |
|------|---------|
| **Python 3.x** | Programming language |
| **Pandas** | Data manipulation & preprocessing |
| **NumPy** | Numerical computations |
| **Scikit-Learn** | Machine learning models & metrics |
| **Matplotlib** | Static visualizations |
| **Seaborn** | Statistical data visualization |
| **PyCharm** | IDE & development environment |
