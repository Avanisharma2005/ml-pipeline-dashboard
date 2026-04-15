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
from sklearn.metrics import silhouette_score
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
    "🎯 Feature Selection", "🤖 Model Selection & Training", 
    "⚙️ Hyperparameter Tuning", "🧪 Model Testing"
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

        # --- Separate features and target ---
        X = df_fs.drop(columns=[target_col])
        y = df_fs[target_col]

        # --- Encode categorical features ---
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        # --- Encode target properly ---
        if problem_type == "Classification":
            if y.dtype == 'object':
                y = LabelEncoder().fit_transform(y.astype(str))

        # --- Feature Selection Methods ---
        method = st.multiselect(
            "Selection Methods",
            ["Variance Threshold", "Correlation", "Information Gain"]
        )

        selected_features = list(X.columns)

        # -------------------------------
        # 1. Variance Threshold
        # -------------------------------
        if "Variance Threshold" in method:
            threshold = st.slider("Variance Threshold", 0.0, 1.0, 0.1)
            vt = VarianceThreshold(threshold=threshold)
            vt.fit(X)

            selected_features = [
                f for f, s in zip(X.columns, vt.get_support()) if s
            ]

            X = X[selected_features]

        # -------------------------------
        # 2. Correlation Filtering
        # -------------------------------
        if "Correlation" in method:
            corr_threshold = st.slider("Correlation Threshold", 0.7, 0.99, 0.9)

            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            to_drop = [
                column for column in upper.columns if any(upper[column] > corr_threshold)
            ]

            X = X.drop(columns=to_drop)
            selected_features = list(X.columns)

            st.write("Dropped due to high correlation:", to_drop)

        # -------------------------------
        # 3. Information Gain (Mutual Info)
        # -------------------------------
        if "Information Gain" in method:
            k = st.slider("Top K Features (Information Gain)", 1, len(X.columns), min(5, len(X.columns)))

            if problem_type == "Classification":
                scores = mutual_info_classif(X, y)
            else:
                scores = mutual_info_regression(X, y)

            mi_df = pd.DataFrame({
                "Feature": X.columns,
                "Score": scores
            }).sort_values(by="Score", ascending=False)

            top_features = mi_df.head(k)["Feature"].tolist()

            X = X[top_features]
            selected_features = top_features

            st.write("Top Features based on Information Gain:")
            st.dataframe(mi_df)

        # -------------------------------
        # FINAL OUTPUT
        # -------------------------------
        st.write("### Final Selected Features:", selected_features)

        # Save to session
        st.session_state.final_features = selected_features
        st.session_state.X = X
        st.session_state.y = y
# --- Tab 5: Training & Metrics ---


# --- Tab 5: Training & Metrics ---
with tabs[4]:
    if 'X' in st.session_state:
        st.subheader("Data Split")

        test_size = st.slider("Test Size (%)", 10, 50, 20)
        X_train, X_test, y_train, y_test = train_test_split(
            st.session_state.X, st.session_state.y,
            test_size=test_size/100, random_state=42
        )

        st.subheader("Model Selection")

        if problem_type == "Classification":
            model_name = st.selectbox("Model", ["Random Forest", "SVM", "Logistic Regression", "K-Means"])
        else:
            model_name = st.selectbox("Model", ["Random Forest", "SVR", "Linear Regression"])

        k_val = st.number_input("K-Fold Cross Validation (K)", min_value=2, max_value=10, value=5)

        if st.button("Train Model"):

            # -------------------------
            # K-MEANS (SPECIAL CASE)
            # -------------------------
            if model_name == "K-Means":
                n_clusters = st.slider("Number of Clusters (K)", 2, 10, 3)

                model = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = model.fit_predict(st.session_state.X)

                # Evaluate clustering
                sil_score = silhouette_score(st.session_state.X, clusters)

                st.metric("Silhouette Score", f"{sil_score:.4f}")

                # Visualization (2D PCA)
                pca = PCA(n_components=2)
                reduced = pca.fit_transform(st.session_state.X)

                fig = px.scatter(
                    x=reduced[:, 0],
                    y=reduced[:, 1],
                    color=clusters.astype(str),
                    title="K-Means Clustering (PCA Projection)"
                )

                st.plotly_chart(fig, use_container_width=True)

                st.info("K-Means is unsupervised → no accuracy or R².")

                st.session_state.current_model = model

            # -------------------------
            # SUPERVISED MODELS
            # -------------------------
            else:
                if model_name == "Random Forest":
                    model = RandomForestClassifier() if problem_type == "Classification" else RandomForestRegressor()

                elif model_name == "SVM":
                    kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
                    model = SVC(kernel=kernel) if problem_type == "Classification" else SVR(kernel=kernel)

                elif model_name == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)

                elif model_name == "Linear Regression":
                    model = LinearRegression()

                # Train model
                model.fit(X_train, y_train)

                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=k_val)

                # Predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                # Metrics
                if problem_type == "Classification":
                    train_score = accuracy_score(y_train, y_pred_train)
                    test_score = accuracy_score(y_test, y_pred_test)
                    f1 = f1_score(y_test, y_pred_test, average='weighted')

                    st.metric("Train Accuracy", f"{train_score:.4f}")
                    st.metric("Test Accuracy", f"{test_score:.4f}")
                    st.metric("F1 Score", f"{f1:.4f}")

                else:
                    train_score = r2_score(y_train, y_pred_train)
                    test_score = r2_score(y_test, y_pred_test)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

                    st.metric("Train R²", f"{train_score:.4f}")
                    st.metric("Test R²", f"{test_score:.4f}")
                    st.metric("RMSE", f"{rmse:.4f}")

                st.write(f"Cross Validation Score (K={k_val}):", cv_scores.mean())

                # Overfitting check
                if train_score > test_score + 0.15:
                    st.error("⚠️ Potential Overfitting detected!")
                elif train_score < 0.5:
                    st.warning("⚠️ Potential Underfitting detected!")
                else:
                    st.success("✅ Model fit looks good.")

                st.session_state.current_model = model
# --- Tab 6: Hyperparameter Tuning ---
# --- Tab 6: Hyperparameter Tuning ---
from sklearn.metrics import make_scorer, accuracy_score, r2_score

with tabs[5]:
    if 'current_model' in st.session_state:

        st.subheader("Hyperparameter Tuning (Grid Search)")

        # Use already prepared data
        X = st.session_state.X
        y = st.session_state.y

        # Train-test split (avoid leakage)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = st.session_state.current_model

        # -------------------------
        # DEFINE PARAM GRIDS
        # -------------------------
        param_grid = None

        if isinstance(model, RandomForestClassifier):
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }

        elif isinstance(model, RandomForestRegressor):
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }

        elif isinstance(model, SVC):
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }

        elif isinstance(model, SVR):
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'epsilon': [0.1, 0.2]
            }

        elif isinstance(model, LogisticRegression):
            param_grid = {
                'C': [0.1, 1, 10],
                'solver': ['lbfgs', 'liblinear']
            }

        elif isinstance(model, LinearRegression):
            st.info("Linear Regression has no major hyperparameters to tune.")
            param_grid = None

        elif isinstance(model, KMeans):
            st.info("K-Means tuning based on number of clusters.")
            
            k_range = st.slider("Select range of K", 2, 10, (2, 6))
            scores = []

            for k in range(k_range[0], k_range[1] + 1):
                km = KMeans(n_clusters=k, random_state=42)
                labels = km.fit_predict(X)
                score = silhouette_score(X, labels)
                scores.append(score)

            # Plot elbow-like curve
            fig = px.line(
                x=list(range(k_range[0], k_range[1] + 1)),
                y=scores,
                markers=True,
                title="K vs Silhouette Score"
            )
            st.plotly_chart(fig)

            best_k = list(range(k_range[0], k_range[1] + 1))[np.argmax(scores)]
            st.success(f"Best K (clusters): {best_k}")

            st.stop()

        # -------------------------
        # RUN GRID SEARCH
        # -------------------------
        if param_grid is not None:

            # Select scoring metric
            if problem_type == "Classification":
                scoring = 'accuracy'
            else:
                scoring = 'r2'

            if st.button("Run Hyperparameter Tuning"):

                with st.spinner("Running Grid Search..."):

                    grid = GridSearchCV(
                        estimator=model,
                        param_grid=param_grid,
                        cv=3,
                        scoring=scoring,
                        n_jobs=-1
                    )

                    grid.fit(X_train, y_train)

                    best_model = grid.best_estimator_

                    # Evaluate best model
                    y_pred = best_model.predict(X_test)

                    if problem_type == "Classification":
                        score = accuracy_score(y_test, y_pred)
                        st.metric("Best Test Accuracy", f"{score:.4f}")
                    else:
                        score = r2_score(y_test, y_pred)
                        st.metric("Best Test R²", f"{score:.4f}")

                    st.write("### Best Parameters")
                    st.json(grid.best_params_)

                    st.write("### Best CV Score")
                    st.write(grid.best_score_)

                    # Visualization
                    results_df = pd.DataFrame(grid.cv_results_)
                    fig = px.line(
                        results_df,
                        y="mean_test_score",
                        title="Hyperparameter Tuning Performance"
                    )
                    st.plotly_chart(fig)

                    # Save tuned model
                    st.session_state.current_model = best_model

    else:
        st.info("Please train a model first in the previous tab.")
# --- Tab 7: Model Testing / Prediction ---
with tabs[6]:
    st.subheader("Test Your Model on New Data")

    if 'current_model' not in st.session_state or 'X' not in st.session_state:
        st.warning("Please train a model first.")
    else:
        model = st.session_state.current_model
        feature_names = st.session_state.X.columns.tolist()

        st.write("### Enter Feature Values")

        user_input = {}

        # Create input fields dynamically
        for feature in feature_names:
            user_input[feature] = st.number_input(f"{feature}", value=0.0)

        input_df = pd.DataFrame([user_input])

        st.write("### Input Data")
        st.dataframe(input_df)

        if st.button("Predict"):

            try:
                prediction = model.predict(input_df)[0]

                # --------------------------
                # Handle Output
                # --------------------------
                if prediction in [0, 1]:
                    if prediction == 1:
                        st.success("✅ Prediction: Genuine Banknote")
                    else:
                        st.error("❌ Prediction: Fake Banknote")
                else:
                    st.info(f"Prediction Output: {prediction}")

            except Exception as e:
                st.error(f"Error during prediction: {e}")