import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score

# --- Page Config ---
st.set_page_config(page_title="Auto-ML Pipeline Pro", layout="wide")

st.markdown("""
<style>
/* --- Global Background --- */
.main {
    background-color: #f8fafc;
}

/* --- Title Styling --- */
h1 {
    color: #1f2937;
    font-weight: 700;
}

/* --- Tabs Styling --- */
.stTabs [data-baseweb="tab-list"] {
    gap: 20px;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    background-color: #e5e7eb;
    border-radius: 10px 10px 0 0;
    padding: 10px 18px;
    font-weight: 600;
    color: #374151;
    transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: #d1d5db;
}

.stTabs [aria-selected="true"] {
    background-color: #4e79a7;
    color: white;
}

/* --- Cards (Containers) --- */
.block-container {
    padding-top: 2rem;
}

section[data-testid="stVerticalBlock"] > div {
    background: white;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
    margin-bottom: 15px;
}

/* --- Buttons --- */
.stButton > button {
    background-color: #4e79a7;
    color: white;
    border-radius: 8px;
    height: 42px;
    font-weight: 600;
    border: none;
}

.stButton > button:hover {
    background-color: #3b5f87;
}

/* --- Inputs --- */
.stSelectbox, .stMultiSelect, .stTextInput, .stNumberInput {
    border-radius: 8px;
}

/* --- Sidebar --- */
.css-1d391kg {
    background-color: #f1f5f9;
}

/* --- Metrics --- */
[data-testid="metric-container"] {
    background: #ffffff;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* --- Smooth UI --- */
* {
    transition: all 0.15s ease-in-out;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 Advanced ML Pipeline Dashboard")

# --- Session State Initialization ---
if 'data' not in st.session_state:
    st.session_state.data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None

# --- Sidebar: Problem Setup ---
st.sidebar.header("1. Problem Configuration")
problem_type = st.sidebar.selectbox("Select Problem Type", ["Classification", "Regression"])

# --- Main Tabs ---
tabs = st.tabs([
    "📂 Data Input", "📊 EDA", "🧹 Cleaning & Engineering", 
    "🎯 Feature Selection", "🤖 Model Selection & Training", "⚙️ Hyperparameter Tuning"
])

# --- Tab 1: Data Input & PCA ---
with tabs[0]:
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file, encoding='latin-1')
        st.session_state.data = df
        st.write("### Data Preview", df.head())
        
        target_col = st.selectbox("Select Target Feature", df.columns)
        feature_cols = st.multiselect("Select Features for Analysis", [c for c in df.columns if c != target_col], default=[c for c in df.columns if c != target_col])
        
        if len(feature_cols) >= 2:
            st.write("### Data Shape Visualization (PCA)")
            # Simple numeric encoding for PCA
            temp_df = df[feature_cols].copy()
            for col in temp_df.select_dtypes(include=['object']).columns:
                temp_df[col] = LabelEncoder().fit_transform(temp_df[col].astype(str))
            
            scaled_data = StandardScaler().fit_transform(temp_df.dropna())
            pca = PCA(n_components=3)
            components = pca.fit_transform(scaled_data)
            
            dim = st.radio("PCA Dimensions", ["2D", "3D"], horizontal=True)
            if dim == "2D":
                fig = px.scatter(x=components[:,0], y=components[:,1], color=df.loc[temp_df.dropna().index, target_col],
                                 labels={'x': 'PC1', 'y': 'PC2'}, title="PCA 2D Projection")
            else:
                fig = px.scatter_3d(x=components[:,0], y=components[:,1], z=components[:,2], 
                                    color=df.loc[temp_df.dropna().index, target_col],
                                    labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'}, title="PCA 3D Projection")
            st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: EDA ---
with tabs[1]:
    if st.session_state.data is not None:
        st.write("### Statistical Summary")
        st.write(st.session_state.data.describe())
        
        st.write("### Correlation Matrix")
        corr = st.session_state.data.select_dtypes(include=[np.number]).corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr)
        
        col1, col2 = st.columns(2)
        with col1:
            dist_col = st.selectbox("Distribution View", st.session_state.data.columns)
            fig_dist = px.histogram(st.session_state.data, x=dist_col, color=target_col if target_col else None)
            st.plotly_chart(fig_dist)
    else:
        st.warning("Please upload data first.")

# --- Tab 3: Data Engineering & Cleaning ---
with tabs[2]:
    if st.session_state.data is not None:
        df_clean = st.session_state.data.copy()
        
        st.subheader("Imputation")
        strat = st.selectbox("Imputation Strategy", ["mean", "median", "most_frequent"])
        if st.button("Apply Imputation"):
            imputer = SimpleImputer(strategy=strat)
            num_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[num_cols] = imputer.fit_transform(df_clean[num_cols])
            st.success("Missing values handled!")

        st.subheader("Outlier Detection")
        outlier_method = st.selectbox("Method", ["IQR", "Isolation Forest", "DBSCAN", "OPTICS"])
        
        outliers = np.zeros(len(df_clean), dtype=bool)
        numeric_df = df_clean.select_dtypes(include=[np.number])
        
        if outlier_method == "IQR":
            Q1 = numeric_df.quantile(0.25)
            Q3 = numeric_df.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
        elif outlier_method == "Isolation Forest":
            clf = IsolationForest(contamination=0.05, random_state=42)
            outliers = clf.fit_predict(numeric_df) == -1
        
        st.write(f"Detected {sum(outliers)} outliers using {outlier_method}")
        
        if sum(outliers) > 0:
            if st.button("Remove Selected Outliers"):
                st.session_state.cleaned_data = df_clean[~outliers]
                st.success("Outliers removed from dataset!")
        else:
            st.session_state.cleaned_data = df_clean

# --- Tab 4: Feature Selection ---
with tabs[3]:
    if st.session_state.cleaned_data is not None:
        df_fs = st.session_state.cleaned_data.copy()
        # Separate features and target
        X = df_fs.drop(columns=[target_col])
        y = df_fs[target_col]
        
        # Encoding categorical for math
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        # Ensure target is discrete for classification
        if problem_type == "Classification":
            y = y.astype(int)
            
            # If still object (edge case)
            if y.dtype == 'object':
                y = LabelEncoder().fit_transform(y.astype(str))

        method = st.multiselect("Selection Methods", ["Variance Threshold", "Correlation", "Information Gain"])
        
        selected_feats = list(X.columns)
        
        if "Variance Threshold" in method:
            vt = VarianceThreshold(threshold=0.1)
            vt.fit(X)
            selected_feats = [f for f, s in zip(selected_feats, vt.get_support()) if s]
            
        st.write("Selected Features:", selected_feats)
        st.session_state.final_features = selected_feats
        st.session_state.X = X[selected_feats]
        st.session_state.y = y

# --- Tab 5: Training & Metrics ---
with tabs[4]:
    if 'X' in st.session_state:
        st.subheader("Data Split")
        test_size = st.slider("Test Size (%)", 10, 50, 20)
        X_train, X_test, y_train, y_test = train_test_split(st.session_state.X, st.session_state.y, test_size=test_size/100, random_state=42)
        
        st.subheader("Model Selection")
        if problem_type == "Classification":
            model_name = st.selectbox("Model", ["Random Forest", "SVM", "Logistic Regression", "K-Means"])
        else:
            model_name = st.selectbox("Model", ["Random Forest", "SVR", "Linear Regression"])

        k_val = st.number_input("K-Fold Cross Validation (K)", min_value=2, max_value=10, value=5)
        
        if st.button("Train Model"):
            if model_name == "Random Forest":
                model = RandomForestClassifier() if problem_type == "Classification" else RandomForestRegressor()
            elif model_name == "SVM" or model_name == "SVR":
                kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
                model = SVC(kernel=kernel) if problem_type == "Classification" else SVR(kernel=kernel)
            elif model_name == "Linear Regression" or model_name == "Logistic Regression":
                model = LogisticRegression() if problem_type == "Classification" else LinearRegression()
            
            # Train
            model.fit(X_train, y_train)
            cv_scores = cross_val_score(model, X_train, y_train, cv=k_val)
            
            # Metrics
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            st.metric("Train Accuracy/R2", f"{train_score:.4f}")
            st.metric("Test Accuracy/R2", f"{test_score:.4f}")
            st.write(f"CV Scores (K={k_val}):", cv_scores.mean())
            
            if train_score > test_score + 0.15:
                st.error("Warning: Potential Overfitting detected!")
            elif train_score < 0.5:
                st.warning("Warning: Potential Underfitting detected!")
            else:
                st.success("Model fit seems stable.")
            
            st.session_state.current_model = model

# --- Tab 6: Hyperparameter Tuning ---
with tabs[5]:
    if 'current_model' in st.session_state:
        st.subheader("Grid Search Optimization")
        if isinstance(st.session_state.current_model, (RandomForestClassifier, RandomForestRegressor)):
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20]
            }
            search = GridSearchCV(st.session_state.current_model, param_grid, cv=3)
            if st.button("Run Hyperparameter Tuning"):
                with st.spinner("Tuning..."):
                    search.fit(st.session_state.X, st.session_state.y)
                    st.write("Best Params:", search.best_params_)
                    st.write("Best Score:", search.best_score_)
                    
                    # Performance Effect visualization
                    results_df = pd.DataFrame(search.cv_results_)
                    fig_tune = px.line(results_df, y='mean_test_score', title="Tuning Effect on Performance")
                    st.plotly_chart(fig_tune)
    else:
        st.info("Please train a model in the previous tab first.")