import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression

# Set page title and layout
st.set_page_config(page_title="LDA & Logistic Regression", layout="centered")

st.title("Logistic Regression with LDA (Wine Dataset)")

# Load dataset with caching
@st.cache_data
def load_data():
    dataset = pd.read_csv('Wine.csv')
    return dataset

try:
    dataset = load_data()

    st.write("### Dataset Preview:")
    st.dataframe(dataset.head())

    # Features and target variable extraction
    X = dataset.iloc[:, 0:13].values
    y = dataset.iloc[:, 13].values

    # Splitting dataset into Training and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Applying Linear Discriminant Analysis (LDA)
    lda = LDA(n_components=2)
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)

    # Training Logistic Regression classifier
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)

    # Plotting Decision Boundary for Test set
    st.write("### Decision Boundary (Test Set):")
    fig, ax = plt.subplots(figsize=(8, 6))

    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
        np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
    )

    ax.contourf(
        X1, X2, 
        classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        alpha=0.75, 
        cmap=ListedColormap(('red', 'green', 'blue'))
    )

    ax.set_xlim(X1.min(), X1.max())
    ax.set_ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):
        ax.scatter(
            X_set[y_set == j, 0], X_set[y_set == j, 1],
            color=ListedColormap(('red', 'green', 'blue'))(i), label=f"Class {j}"
        )

    ax.set_title('Logistic Regression (Test set)')
    ax.set_xlabel('LD1')
    ax.set_ylabel('LD2')
    ax.legend()

    # Display plot in Streamlit
    st.pyplot(fig)

except FileNotFoundError:
    st.error("Error: Could not find 'Wine.csv'. Please make sure the dataset file is uploaded to the same directory.")
