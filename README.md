
# **Machine Learning Binary Classification for Offer Prioritization**

Schneider Electric manages complex solutions across markets such as Data Centers, Buildings, Industry, and Infrastructure. Historically, offer prioritization has been a manual process requiring engineers and architects. This project aims to develop a machine learning-based tool to intelligently predict the outcomes of offers (win or lose) and rank them by priority, enabling data-driven decision-making.

---

## **Objective**
The primary goal is to build and evaluate machine learning models to:
1. Predict the probability of an offer being **won** (label = 1) or **lost** (label = 0).
2. Rank and prioritize offers based on their win probability.
3. Compare multiple machine learning algorithms to determine the best-performing model.

---

## **Dataset Description**
- **Source**: Historical offer data spanning two years.
- **Type**: Tabular dataset.
- **Features**: Quantitative and qualitative attributes, including client information, offer details, amounts, and material types.
- **Target Variable**: Binary column indicating offer status (0 = Lost, 1 = Won).
- **Unique Identifier**: Offer ID.

---

## **Machine Learning Models**
The following models were implemented and compared:
1. **K-Nearest Neighbors (KNN)**
2. **Naive Bayes (NB)**
3. **Logistic Regression**
4. **Decision Tree**
5. **Random Forest**
6. **Support Vector Machine (SVM)**

---

## **Evaluation Metrics**
The models were assessed using the following metrics:
1. **F1-Score**: Balance between precision and recall.
2. **Recall**: Ability to identify true positive outcomes.
3. **Precision**: Accuracy of positive predictions.

---

## **Results**
### **Performance Summary**

| **Model**             | **F1-Score** | **Precision** | **Recall** |
|------------------------|--------------|---------------|------------|
| K-Nearest Neighbors    | 0.78         | 0.80          | 0.75       |
| Naive Bayes            | 0.76         | 0.77          | 0.74       |
| Logistic Regression    | 0.82         | 0.85          | 0.80       |
| Decision Tree          | 0.84         | 0.83          | 0.85       |
| Random Forest          | **0.88**     | **0.90**      | 0.86       |
| Support Vector Machine | 0.85         | 0.87          | **0.88**   |

### **Key Insights**
- **Random Forest** emerged as the best model, achieving the highest F1-score and precision.
- **Support Vector Machine** had the highest recall, making it suitable for minimizing false negatives.
- Simpler models like **Logistic Regression** and **Decision Tree** also demonstrated competitive performance.

---

## **Technical Workflow**
1. **Data Preprocessing**:
   - Handled missing values and outliers.
   - Encoded categorical variables.
   - Scaled numerical features for SVM and KNN models.

2. **Feature Selection**:
   - Identified the most important features using correlation analysis and feature importance techniques.

3. **Model Training and Evaluation**:
   - Split the data into training and testing sets (80:20 ratio).
   - Trained each model using cross-validation.
   - Evaluated the models on the test set.

4. **Offer Prioritization**:
   - Calculated win probabilities for each offer.
   - Sorted offers in descending order of probability.

---

## **Future Enhancements**
1. **Hyperparameter Optimization**:
   - Apply Grid Search or Random Search for fine-tuning model parameters.
2. **Integration with Business Tools**:
   - Develop an API to integrate the predictive tool into Schneider Electric's CRM systems.
3. **Explainability**:
   - Use tools like SHAP or LIME for model interpretability to explain predictions to stakeholders.

---

## **How to Run**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/machine-learning-binary-classification.git
   cd machine-learning-binary-classification
   ```

2. **Set Up the Environment**:
   - Create a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Execute the Workflow**:
   - **Data Preprocessing**: Run `01_Data_Preprocessing.ipynb`.
   - **Model Training**: Run `02_Model_Training.ipynb`.
   - **Evaluation**: Run `03_Model_Evaluation.ipynb`.

---

