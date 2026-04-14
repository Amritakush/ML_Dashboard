import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ===== PAGE CONFIG =====
st.set_page_config(page_title="ML Pipeline Dashboard", layout="wide")

# ===== SIDEBAR =====
st.sidebar.title("📊 ML Pipeline")
page = st.sidebar.radio("Go to", [
    "1️⃣ Input Data",
    "2️⃣ EDA",
    "3️⃣ Cleaning",
    "4️⃣ Model Pipeline",
    "5️⃣ Metrics",
    "6️⃣ Prediction"   # ✅ ADDED
])

st.title("🎓 Student Performance ML Pipeline Dashboard")

# =========================
# STEP 1: INPUT DATA
# =========================
if page == "1️⃣ Input Data":
    st.header("🟢 Step 1: Input Data")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.success("Dataset Loaded!")
        st.dataframe(df.head())

        st.session_state["data"] = df

# =========================
# STEP 2: EDA
# =========================
elif page == "2️⃣ EDA":
    st.header("🟢 Step 2: Exploratory Data Analysis")

    if "data" in st.session_state:
        df = st.session_state["data"]

        st.write(df.describe())

        col = st.selectbox("Select Column", df.columns)
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig)

    else:
        st.warning("Upload dataset first")

# =========================
# STEP 3: CLEANING
# =========================
elif page == "3️⃣ Cleaning":
    st.header("🟢 Step 3: Data Engineering & Cleaning")

    if "data" in st.session_state:
        df = st.session_state["data"]

        method = st.selectbox("Handle Missing Values", ["None", "Mean", "Median"])

        if method != "None":
            for col in df.select_dtypes(include=np.number).columns:
                if method == "Mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif method == "Median":
                    df[col].fillna(df[col].median(), inplace=True)

        if st.button("Detect Outliers"):
            iso = IsolationForest(contamination=0.1)
            preds = iso.fit_predict(df.select_dtypes(include=np.number).fillna(0))

            st.write("Outliers Found:", sum(preds == -1))

        st.session_state["data"] = df
        st.success("Cleaning Done")

    else:
        st.warning("Upload dataset first")

# =========================
# STEP 4–8: MODEL PIPELINE
# =========================
elif page == "4️⃣ Model Pipeline":
    st.header("🟢 Step 4–8: Feature Selection → Split → Model → KFold")

    if "data" in st.session_state:
        df = st.session_state["data"]

        target = st.selectbox("Select Target", df.columns)
        features = st.multiselect("Select Features", df.columns.drop(target))

        if features:
            X = df[features]
            y = df[target]

            # SAVE for prediction
            st.session_state["features"] = features
            st.session_state["target"] = target

            if st.checkbox("Apply Variance Threshold"):
                selector = VarianceThreshold()
                X = selector.fit_transform(X)
                st.success("Feature Selection Applied")

            test_size = st.slider("Test Size", 0.1, 0.4, 0.2)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size
            )

            model_name = st.selectbox("Choose Model", [
                "Linear Regression",
                "SVR",
                "Random Forest"
            ])

            if model_name == "Linear Regression":
                model = LinearRegression()
            elif model_name == "SVR":
                model = SVR()
            else:
                model = RandomForestRegressor()

            k = st.slider("K-Fold Value", 2, 10, 5)

            if st.button("Train Model"):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                st.session_state["metrics"] = (mse, r2)
                st.session_state["model"] = model   # ✅ SAVE MODEL

                kf = KFold(n_splits=k)
                scores = cross_val_score(model, X, y, cv=kf)

                st.session_state["kfold"] = scores

                st.success("Model Trained!")

    else:
        st.warning("Upload dataset first")

# =========================
# STEP 9: METRICS
# =========================
elif page == "5️⃣ Metrics":
    st.header("🟢 Step 9: Performance Metrics")

    if "metrics" in st.session_state:
        mse, r2 = st.session_state["metrics"]

        col1, col2 = st.columns(2)
        col1.metric("MSE", f"{mse:.2f}")
        col2.metric("R² Score", f"{r2:.2f}")

        if "kfold" in st.session_state:
            scores = st.session_state["kfold"]
            st.write("K-Fold Scores:", scores)
            st.write("Average Score:", scores.mean())

    else:
        st.warning("Train model first")

# =========================
# STEP 10: PREDICTION (NEW)
# =========================
elif page == "6️⃣ Prediction":
    st.header("🟣 Step 10: Make Prediction")

    if "model" in st.session_state and "features" in st.session_state:
        model = st.session_state["model"]
        features = st.session_state["features"]

        input_data = []

        st.write("Enter values:")

        for feature in features:
            val = st.number_input(f"{feature}", value=0.0)
            input_data.append(val)

        if st.button("Predict"):
            prediction = model.predict([input_data])
            st.success(f"Predicted Value: {prediction[0]:.2f}")

    else:
        st.warning("Train model first")

# ===== DARK STYLE =====
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}
</style>
""", unsafe_allow_html=True)



# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px

# from sklearn.model_selection import train_test_split, KFold, cross_val_score
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.ensemble import RandomForestRegressor, IsolationForest
# from sklearn.svm import SVR
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

# # ===== PAGE CONFIG =====
# st.set_page_config(page_title="ML Pipeline Dashboard", layout="wide")

# # ===== SIDEBAR =====
# st.sidebar.title("📊 ML Pipeline")
# page = st.sidebar.radio("Go to", [
#     "1️⃣ Input Data",
#     "2️⃣ EDA",
#     "3️⃣ Cleaning",
#     "4️⃣ Model Pipeline",
#     "5️⃣ Metrics"
# ])

# st.title("🎓 Student Performance ML Pipeline Dashboard")

# # =========================
# # STEP 1: INPUT DATA
# # =========================
# if page == "1️⃣ Input Data":
#     st.header("🟢 Step 1: Input Data")

#     file = st.file_uploader("Upload CSV", type=["csv"])

#     if file:
#         df = pd.read_csv(file)
#         st.success("Dataset Loaded!")
#         st.dataframe(df.head())

#         st.session_state["data"] = df

# # =========================
# # STEP 2: EDA
# # =========================
# elif page == "2️⃣ EDA":
#     st.header("🟢 Step 2: Exploratory Data Analysis")

#     if "data" in st.session_state:
#         df = st.session_state["data"]

#         st.write(df.describe())

#         col = st.selectbox("Select Column", df.columns)
#         fig = px.histogram(df, x=col)
#         st.plotly_chart(fig)

#     else:
#         st.warning("Upload dataset first")

# # =========================
# # STEP 3: CLEANING
# # =========================
# elif page == "3️⃣ Cleaning":
#     st.header("🟢 Step 3: Data Engineering & Cleaning")

#     if "data" in st.session_state:
#         df = st.session_state["data"]

#         method = st.selectbox("Handle Missing Values", ["None", "Mean", "Median"])

#         if method != "None":
#             for col in df.select_dtypes(include=np.number).columns:
#                 if method == "Mean":
#                     df[col].fillna(df[col].mean(), inplace=True)
#                 elif method == "Median":
#                     df[col].fillna(df[col].median(), inplace=True)

#         # Outlier detection
#         if st.button("Detect Outliers"):
#             iso = IsolationForest(contamination=0.1)
#             preds = iso.fit_predict(df.select_dtypes(include=np.number).fillna(0))

#             st.write("Outliers Found:", sum(preds == -1))

#         st.session_state["data"] = df
#         st.success("Cleaning Done")

#     else:
#         st.warning("Upload dataset first")

# # =========================
# # STEP 4–8: MODEL PIPELINE
# # =========================
# elif page == "4️⃣ Model Pipeline":
#     st.header("🟢 Step 4–8: Feature Selection → Split → Model → KFold")

#     if "data" in st.session_state:
#         df = st.session_state["data"]

#         target = st.selectbox("Select Target", df.columns)
#         features = st.multiselect("Select Features", df.columns.drop(target))

#         if features:
#             X = df[features]
#             y = df[target]

#             # STEP 4: Feature Selection
#             if st.checkbox("Apply Variance Threshold"):
#                 selector = VarianceThreshold()
#                 X = selector.fit_transform(X)
#                 st.success("Feature Selection Applied")

#             # STEP 5: Data Split
#             test_size = st.slider("Test Size", 0.1, 0.4, 0.2)

#             X_train, X_test, y_train, y_test = train_test_split(
#                 X, y, test_size=test_size
#             )

#             # STEP 6: Model Selection
#             model_name = st.selectbox("Choose Model", [
#                 "Linear Regression",
#                 "SVR",
#                 "Random Forest"
#             ])

#             if model_name == "Linear Regression":
#                 model = LinearRegression()
#             elif model_name == "SVR":
#                 model = SVR()
#             else:
#                 model = RandomForestRegressor()

#             # STEP 7 & 8: Training + KFold
#             k = st.slider("K-Fold Value", 2, 10, 5)

#             if st.button("Train Model"):
#                 model.fit(X_train, y_train)
#                 y_pred = model.predict(X_test)

#                 mse = mean_squared_error(y_test, y_pred)
#                 r2 = r2_score(y_test, y_pred)

#                 # Save results
#                 st.session_state["metrics"] = (mse, r2)

#                 # KFold
#                 kf = KFold(n_splits=k)
#                 scores = cross_val_score(model, X, y, cv=kf)

#                 st.session_state["kfold"] = scores

#                 st.success("Model Trained!")

#     else:
#         st.warning("Upload dataset first")

# # =========================
# # STEP 9: METRICS
# # =========================
# elif page == "5️⃣ Metrics":
#     st.header("🟢 Step 9: Performance Metrics")

#     if "metrics" in st.session_state:
#         mse, r2 = st.session_state["metrics"]

#         col1, col2 = st.columns(2)
#         col1.metric("MSE", f"{mse:.2f}")
#         col2.metric("R² Score", f"{r2:.2f}")

#         # KFold results
#         if "kfold" in st.session_state:
#             scores = st.session_state["kfold"]
#             st.write("K-Fold Scores:", scores)
#             st.write("Average Score:", scores.mean())

#     else:
#         st.warning("Train model first")

# # ===== DARK STYLE =====
# st.markdown("""
# <style>
# .stApp {
#     background-color: #0e1117;
#     color: white;
# }
# </style>
# """, unsafe_allow_html=True)


# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

# # ===== PAGE CONFIG =====
# st.set_page_config(page_title="ML Dashboard", layout="wide")

# # ===== SIDEBAR =====
# st.sidebar.title("📊 Navigation")
# page = st.sidebar.radio("Go to", [
#     "Upload Data",
#     "EDA",
#     "Cleaning",
#     "Model Training",
#     "Metrics"
# ])

# st.title("🎓 Student Performance ML Dashboard")

# # =========================
# # PAGE 1: Upload Data
# # =========================
# if page == "Upload Data":
#     st.header("📂 Upload Dataset")

#     file = st.file_uploader("Upload CSV", type=["csv"])

#     if file:
#         df = pd.read_csv(file)
#         st.success("Dataset Loaded Successfully!")
#         st.dataframe(df.head())

#         st.session_state["data"] = df

# # =========================
# # PAGE 2: EDA
# # =========================
# elif page == "EDA":
#     st.header("📊 Exploratory Data Analysis")

#     if "data" in st.session_state:
#         df = st.session_state["data"]

#         st.write(df.describe())

#         col = st.selectbox("Select Column", df.columns)
#         fig = px.histogram(df, x=col)
#         st.plotly_chart(fig)

#     else:
#         st.warning("Upload dataset first")

# # =========================
# # PAGE 3: Cleaning
# # =========================
# elif page == "Cleaning":
#     st.header("🧹 Data Cleaning")

#     if "data" in st.session_state:
#         df = st.session_state["data"]

#         method = st.selectbox("Fill Missing Values", ["None", "Mean", "Median"])

#         if method != "None":
#             for col in df.select_dtypes(include=np.number).columns:
#                 if method == "Mean":
#                     df[col].fillna(df[col].mean(), inplace=True)
#                 elif method == "Median":
#                     df[col].fillna(df[col].median(), inplace=True)

#         st.success("Cleaning Applied")

#     else:
#         st.warning("Upload dataset first")

# # =========================
# # PAGE 4: Model Training
# # =========================
# elif page == "Model Training":
#     st.header("🤖 Model Training")

#     if "data" in st.session_state:
#         df = st.session_state["data"]

#         target = st.selectbox("Select Target", df.columns)
#         features = st.multiselect("Select Features", df.columns.drop(target))

#         if features:
#             X = df[features]
#             y = df[target]

#             model_name = st.selectbox("Choose Model", [
#                 "Linear Regression",
#                 "SVR",
#                 "Random Forest"
#             ])

#             if model_name == "Linear Regression":
#                 model = LinearRegression()
#             elif model_name == "SVR":
#                 model = SVR()
#             else:
#                 model = RandomForestRegressor()

#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#             if st.button("Train Model"):
#                 model.fit(X_train, y_train)
#                 y_pred = model.predict(X_test)

#                 mse = mean_squared_error(y_test, y_pred)
#                 r2 = r2_score(y_test, y_pred)

#                 st.session_state["metrics"] = (mse, r2)

#                 st.success("Model Trained Successfully!")

#     else:
#         st.warning("Upload dataset first")

# # =========================
# # PAGE 5: Metrics
# # =========================
# elif page == "Metrics":
#     st.header("📈 Model Performance")

#     if "metrics" in st.session_state:
#         mse, r2 = st.session_state["metrics"]

#         col1, col2 = st.columns(2)

#         col1.metric("MSE", f"{mse:.2f}")
#         col2.metric("R² Score", f"{r2:.2f}")

#     else:
#         st.warning("Train model first")

# # ===== OPTIONAL DARK STYLE =====
# st.markdown("""
# <style>
# .stApp {
#     background-color: #0e1117;
#     color: white;
# }
# </style>
# """, unsafe_allow_html=True)






# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px

# from sklearn.model_selection import train_test_split, KFold, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.ensemble import RandomForestRegressor, IsolationForest
# from sklearn.svm import SVR
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

# st.set_page_config(layout="wide")
# st.title("🎓 Student Performance ML Pipeline Dashboard")

# # ================= STEP 1 =================
# st.header("🟢 Step 1: Upload Dataset")

# file = st.file_uploader("Upload your CSV file", type=["csv"])

# if file:
#     df = pd.read_csv(file)

#     st.write("Preview of Data", df.head())
#     st.write("Shape of Dataset:", df.shape)

#     # ================= STEP 2 =================
#     st.header("🟢 Step 2: Select Target & Features")

#     target = st.selectbox("Select Target Column (Marks)", df.columns)
#     features = st.multiselect("Select Feature Columns", df.columns.drop(target))

#     if features:
#         X = df[features]
#         y = df[target]

#         # ================= STEP 3 =================
#         # st.header("🟢 Step 3: PCA Visualization")

#         # pca = PCA(n_components=2)
#         # X_pca = pca.fit_transform(X.select_dtypes(include=np.number).fillna(0))

#         # fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1],
#         #                  title="PCA Visualization of Features")
#         # st.plotly_chart(fig)

#         # ================= STEP 4 =================
#         st.header("🟢 Step 4: Exploratory Data Analysis (EDA)")

#         st.write(df.describe())

#         fig1 = px.histogram(df, x=target, title="Target Distribution")
#         st.plotly_chart(fig1)

#         fig2 = px.box(df, y=target, title="Outlier Detection")
#         st.plotly_chart(fig2)

#         # ================= STEP 5 =================
#         st.header("🟢 Step 5: Data Cleaning")

#         method = st.selectbox("Handle Missing Values",
#                               ["None", "Mean", "Median"])

#         if method != "None":
#             for col in X.columns:
#                 if method == "Mean":
#                     X[col].fillna(X[col].mean(), inplace=True)
#                 elif method == "Median":
#                     X[col].fillna(X[col].median(), inplace=True)

#         # Outlier Detection
#         if st.button("Detect Outliers"):
#             iso = IsolationForest(contamination=0.1)
#             preds = iso.fit_predict(X.select_dtypes(include=np.number).fillna(0))

#             st.write("Outliers Found:", sum(preds == -1))

#             if st.checkbox("Remove Outliers"):
#                 X = X[preds == 1]
#                 y = y[preds == 1]
#                 st.success("Outliers Removed")

#         # ================= STEP 6 =================
#         st.header("🟢 Step 6: Train-Test Split")

#         test_size = st.slider("Test Size", 0.1, 0.4, 0.2)

#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=test_size)

#         # ================= STEP 7 =================
#         st.header("🟢 Step 7: Model Selection")

#         model_name = st.selectbox(
#             "Choose Model",
#             ["Linear Regression", "SVR", "Random Forest"]
#         )

#         if model_name == "Linear Regression":
#             model = LinearRegression()
#         elif model_name == "SVR":
#             model = SVR()
#         else:
#             model = RandomForestRegressor()

#         # ================= STEP 8 =================
#         st.header("🟢 Step 8: Training + K-Fold Validation")

#         k = st.slider("Select K value", 2, 10, 5)

#         if st.button("Train Model"):
#             model.fit(X_train, y_train)

#             y_pred = model.predict(X_test)

#             st.subheader("📊 Performance Metrics")

#             mse = mean_squared_error(y_test, y_pred)
#             r2 = r2_score(y_test, y_pred)

#             st.write("MSE:", mse)
#             st.write("R² Score:", r2)

#             # Overfitting check
#             train_score = model.score(X_train, y_train)
#             test_score = model.score(X_test, y_test)

#             st.write("Train Score:", train_score)
#             st.write("Test Score:", test_score)

#             if train_score > test_score:
#                 st.warning("⚠ Model may be overfitting")

#             # K-Fold
#             kf = KFold(n_splits=k)
#             scores = cross_val_score(model, X, y, cv=kf)

#             st.write("K-Fold Scores:", scores)
#             st.write("Average Score:", scores.mean())

#         # ================= STEP 9 =================
#         st.header("🟢 Step 9: Hyperparameter Tuning")

#         if st.checkbox("Enable Tuning"):
#             n_estimators = st.slider("n_estimators", 10, 200, 100)

#             tuned_model = RandomForestRegressor(n_estimators=n_estimators)
#             tuned_model.fit(X_train, y_train)

#             y_pred = tuned_model.predict(X_test)

#             st.success("Tuned Model Results")

#             st.write("New MSE:", mean_squared_error(y_test, y_pred))
#             st.write("New R²:", r2_score(y_test, y_pred))
