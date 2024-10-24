import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(page_title="Binary Classifier Metric Elicitation Demo", layout="wide")

@st.cache(allow_output_mutation=True)
def load_data():
    """Load the Breast Cancer Wisconsin dataset."""
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    return X, y, feature_names

@st.cache(allow_output_mutation=True)
def train_model(X_train, y_train):
    """Train a Logistic Regression model."""
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    return model

def get_threshold_classifiers(y_probs, thresholds):
    """Generate classifiers by applying different thresholds."""
    classifiers = {}
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        classifiers[threshold] = y_pred
    return classifiers

def compare_classifiers(y_true, classifiers, thresholds, metric='accuracy'):
    """Perform pairwise comparisons based on a specified metric."""
    comparisons = {}
    for i in range(len(thresholds)):
        for j in range(i + 1, len(thresholds)):
            thresh_i = thresholds[i]
            thresh_j = thresholds[j]
            if metric == 'accuracy':
                acc_i = np.mean(classifiers[thresh_i] == y_true)
                acc_j = np.mean(classifiers[thresh_j] == y_true)
                if acc_i > acc_j:
                    comparisons[(thresh_i, thresh_j)] = thresh_i
                else:
                    comparisons[(thresh_i, thresh_j)] = thresh_j
            elif metric == 'precision':
                tp_i = np.sum((classifiers[thresh_i] == 1) & (y_true == 1))
                fp_i = np.sum((classifiers[thresh_i] == 1) & (y_true == 0))
                tp_j = np.sum((classifiers[thresh_j] == 1) & (y_true == 1))
                fp_j = np.sum((classifiers[thresh_j] == 1) & (y_true == 0))
                precision_i = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else 0
                precision_j = tp_j / (tp_j + fp_j) if (tp_j + fp_j) > 0 else 0
                if precision_i > precision_j:
                    comparisons[(thresh_i, thresh_j)] = thresh_i
                else:
                    comparisons[(thresh_i, thresh_j)] = thresh_j
            elif metric == 'recall':
                tp_i = np.sum((classifiers[thresh_i] == 1) & (y_true == 1))
                fn_i = np.sum((classifiers[thresh_i] == 0) & (y_true == 1))
                tp_j = np.sum((classifiers[thresh_j] == 1) & (y_true == 1))
                fn_j = np.sum((classifiers[thresh_j] == 0) & (y_true == 1))
                recall_i = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0
                recall_j = tp_j / (tp_j + fn_j) if (tp_j + fn_j) > 0 else 0
                if recall_i > recall_j:
                    comparisons[(thresh_i, thresh_j)] = thresh_i
                else:
                    comparisons[(thresh_i, thresh_j)] = thresh_j
            # Additional metrics can be added here
    return comparisons

def elicit_optimal_threshold(comparisons):
    """Determine the optimal threshold based on pairwise comparisons."""
    preference_counts = {}
    for (thresh_i, thresh_j), preferred in comparisons.items():
        preference_counts[preferred] = preference_counts.get(preferred, 0) + 1
    # The threshold with the highest preference count is the optimal one
    optimal_threshold = max(preference_counts, key=preference_counts.get)
    return optimal_threshold, preference_counts

def plot_roc_curve(y_true, y_probs):
    """Plot the ROC Curve."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
    ax.plot([0,1], [0,1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    return fig

def main():
    st.title("🔍 Binary Classifier Metric Elicitation Demo")
    st.write("""
    This demo illustrates how to **elicit an optimal decision threshold** for a binary classifier using **pairwise comparisons**. 
    You can choose between **automated comparisons** based on a predefined metric or perform **user-driven comparisons** to determine the optimal threshold.
    """)

    # Sidebar for user options
    st.sidebar.header("Configuration")
    comparison_type = st.sidebar.selectbox("Comparison Type", options=["Automated", "User-Driven"], index=0)
    selected_metric = "accuracy"
    if comparison_type == "Automated":
        selected_metric = st.sidebar.selectbox("Select Metric for Automated Comparisons", options=["accuracy", "precision", "recall"], index=0)

    # Load and split the data
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)
    y_probs = model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

    # Generate threshold classifiers
    thresholds = np.linspace(0.1, 0.9, 9)  # Thresholds from 0.1 to 0.9
    classifiers = get_threshold_classifiers(y_probs, thresholds)

    st.header("📊 Dataset Overview")
    st.write("Using the **Breast Cancer Wisconsin** dataset for binary classification.")
    st.write(f"**Number of samples**: {X.shape[0]}")
    st.write(f"**Number of features**: {X.shape[1]}")

    st.header("🧠 Classifier Performance at Different Thresholds")
    performance_data = []
    for thresh in thresholds:
        y_pred = classifiers[thresh]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        performance_data.append({
            'Threshold': thresh,
            'Accuracy': round(accuracy, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1-Score': round(f1, 4)
        })
    performance_df = pd.DataFrame(performance_data)
    performance_df = performance_df.set_index('Threshold')
    st.dataframe(performance_df)

    st.header("🔄 Pairwise Comparisons")
    if comparison_type == "Automated":
        st.subheader("Automated Pairwise Comparisons")
        st.write(f"Comparisons based on **{selected_metric.capitalize()}** metric.")
        comparisons = compare_classifiers(y_test, classifiers, thresholds, metric=selected_metric)
    else:
        st.subheader("User-Driven Pairwise Comparisons")
        st.write("Compare classifiers pairwise and select your preferred threshold for each pair.")
        comparisons = {}
        for i in range(len(thresholds)):
            for j in range(i + 1, len(thresholds)):
                thresh_i = thresholds[i]
                thresh_j = thresholds[j]
                st.markdown(f"**Compare Threshold {thresh_i:.2f} vs Threshold {thresh_j:.2f}**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Threshold {thresh_i:.2f} Performance**")
                    y_pred_i = classifiers[thresh_i]
                    tn_i, fp_i, fn_i, tp_i = confusion_matrix(y_test, y_pred_i).ravel()
                    acc_i = (tp_i + tn_i) / (tp_i + tn_i + fp_i + fn_i)
                    prec_i = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else 0
                    rec_i = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0
                    st.write(f"**Accuracy**: {acc_i:.4f}")
                    st.write(f"**Precision**: {prec_i:.4f}")
                    st.write(f"**Recall**: {rec_i:.4f}")
                with col2:
                    st.write(f"**Threshold {thresh_j:.2f} Performance**")
                    y_pred_j = classifiers[thresh_j]
                    tn_j, fp_j, fn_j, tp_j = confusion_matrix(y_test, y_pred_j).ravel()
                    acc_j = (tp_j + tn_j) / (tp_j + tn_j + fp_j + fn_j)
                    prec_j = tp_j / (tp_j + fp_j) if (tp_j + fp_j) > 0 else 0
                    rec_j = tp_j / (tp_j + fn_j) if (tp_j + fn_j) > 0 else 0
                    st.write(f"**Accuracy**: {acc_j:.4f}")
                    st.write(f"**Precision**: {prec_j:.4f}")
                    st.write(f"**Recall**: {rec_j:.4f}")
                preference = st.radio(
                    f"**Which threshold do you prefer?**",
                    options=[f"Threshold {thresh_i:.2f}", f"Threshold {thresh_j:.2f}"],
                    key=f"radio_{thresh_i}_{thresh_j}"
                )
                if preference == f"Threshold {thresh_i:.2f}":
                    comparisons[(thresh_i, thresh_j)] = thresh_i
                else:
                    comparisons[(thresh_i, thresh_j)] = thresh_j
                st.markdown("---")

    st.header("📈 Preference Counts")
    optimal_threshold, preference_counts = elicit_optimal_threshold(comparisons)
    preference_df = pd.DataFrame(list(preference_counts.items()), columns=['Threshold', 'Preference Count'])
    preference_df = preference_df.sort_values(by='Preference Count', ascending=False)
    st.bar_chart(preference_df.set_index('Threshold'))

    st.header("🏆 Optimal Threshold")
    st.write(f"The **optimal threshold** based on pairwise comparisons is **{optimal_threshold:.2f}** with a preference count of **{preference_counts[optimal_threshold]}**.")

    st.header("📉 ROC Curve")
    fig = plot_roc_curve(y_test, y_probs)
    st.pyplot(fig)

    st.header("🔄 Compare Automated vs. User-Driven")
    if comparison_type == "Automated":
        st.write("You have selected **Automated** comparisons based on a specific metric.")
    else:
        st.write("You have performed **User-Driven** pairwise comparisons to determine the optimal threshold.")
    st.write(f"**Optimal Threshold**: {optimal_threshold:.2f}")

    st.header("💡 Conclusion")
    st.write("""
    This demo showcases how pairwise comparisons can be utilized to elicit an optimal decision threshold for a binary classifier. 
    Whether through automated metrics or user-driven preferences, understanding and selecting the right threshold is crucial for model performance and alignment with specific objectives.
    """)

if __name__ == "__main__":
    main()
