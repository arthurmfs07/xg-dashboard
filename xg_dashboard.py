import streamlit as st
from streamlit_plotly_events import plotly_events
import pandas as pd
import plotly.express as px
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import plotly.graph_objects as go
import numpy as np
#from transformers import PreprocessingTransformer, FeatureEngineeringTransformer #, Word2VecEmbedder,
import joblib
from sklearn.metrics import  accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from imblearn.metrics import geometric_mean_score
from sklearn import set_config
from streamlit.components.v1 import html as st_html
from sklearn.ensemble import StackingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
#import gensim
from huggingface_hub import hf_hub_download
import os
st.set_page_config(layout="wide")

# -------------- DATA & MODEL PLACEHOLDERS --------------

# Available models
available_models = [
    "ADASYN", "BSMOTE", "RENN", "ROS",
    "RUS", "SMOTE", "SMOTEENN", "SMOTETomek"
]

# Hugging Face repo
HF_REPO = "arthurmfs07/xg-dashboard-artifacts"

# Sidebar model selector
selected_model_name = st.sidebar.selectbox("Select Model", available_models)

# Helper: Load file from local cache or Hugging Face
def load_artifact(filename):
    local_path = os.path.join("artifacts", filename)
    if not os.path.exists(local_path):
        os.makedirs("artifacts", exist_ok=True)
        st.info(f"Downloading {filename} from Hugging Face Hub...")
        local_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
            repo_type="dataset",
            local_dir="artifacts"
        )
    return joblib.load(local_path)

# Load preprocessor and model
preprocessor = load_artifact(f"preprocessor_{selected_model_name}.joblib")
model = load_artifact(f"stacked_clf_{selected_model_name}.joblib")

X_csv_path = hf_hub_download(
    repo_id="arthurmfs07/xg-dashboard-artifacts",
    filename="X_full_original.csv",
    repo_type="dataset"
)

y_csv_path = hf_hub_download(
    repo_id="arthurmfs07/xg-dashboard-artifacts",
    filename="y_full_original.csv",
    repo_type="dataset"
)

# Load full data
X_full = pd.read_csv(X_csv_path)
y_full = pd.read_csv(y_csv_path).squeeze()

# Reset index to ensure alignment (safety)
X_full = X_full.reset_index(drop=True)
y_full = y_full.reset_index(drop=True)

# Predict xG
X_full_transformed = preprocessor.transform(X_full)
xG_preds = model.predict_proba(X_full_transformed)[:, 1]

# Combine for dashboard use
X_full["xG"] = xG_preds
X_full["y_true"] = y_full


# -------------- STREAMLIT APP --------------

st.title("⚽ Expected Goals (xG) Dashboard")


st.info("""
**What is Expected Goals (xG)?**

Expected Goals is a metric that helps us understand **how likely a shot is to result in a goal** based on its characteristics.

Unlike simply counting goals, xG gives deeper insights into **shot quality** and **chance creation**, helping analysts, coaches, and fans evaluate performance more objectively - it even allows experts to build player profiles

In this dashboard, we use **machine learning** to estimate xG for every shot based on thousands of past examples — turning raw match data into **actionable insights**.

> 💡 Think of xG as answering the question:  
> “*Given where and how this shot was taken, how often would it be expected to score?*”
""")

# -------------------------------------


# -------------------------------------
# 🔧 Preprocessing
# -------------------------------------

# --- Your feature lists ---
embedding_cols = ['player', 'team']
one_hot = ['shot_technique', 'shot_type']
numerical = ['goal_distance', 'shot_angle', 'shot_zone_area']
binary = ['shot_deflected', 'shot_first_time', 'shot_open_goal']
body_part = ['shot_body_part']
locations = ['location_x', 'location_y']
timestamp = ['timestamp']
engineered = [ "interaction_dist_angle",'effective_angle',"adjusted_shot_power","shot_cone_area","x_y_ratio","distance_y_product",
              "quick_first_time","open_goal_adjusted_angle","location_r","location_theta","rel_x","rel_y","abs_distance_to_goal_center_y",
              "distance_squared","inverse_distance","time_seconds","time_minutes","time_remaining_half"]
binary += ['match_pressure', 'time_half']

class DummyFootballEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, dim_player=4, dim_team=3):
        self.dim_player = dim_player
        self.dim_team = dim_team

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        n = len(X)
        player_embs = np.zeros((n, self.dim_player))
        team_embs = np.zeros((n, self.dim_team))
        return np.hstack([player_embs, team_embs])


def football_feature_engineering_func(X):
    X = X.copy()
    if not np.issubdtype(X["timestamp"].dtype, np.timedelta64):
        X["timestamp"] = pd.to_timedelta(X["timestamp"])

    X["interaction_dist_angle"] = X["goal_distance"] * X["shot_angle"]
    X['effective_angle'] = np.cos(X["shot_angle"]) / X['goal_distance']
    # ... rest unchanged
    return X

def build_football_preprocessor(player2vec, team2vec):
    feature_engineering = FunctionTransformer(football_feature_engineering_func)

    col_transformer = ColumnTransformer([
        ("bodypart", OneHotEncoder(), body_part),
        ("one_hot", OneHotEncoder(handle_unknown="ignore"), one_hot),
        ("locations", "passthrough", locations),
        ("numerical", "passthrough", numerical),
        ("engineered", "passthrough", engineered),
        ("binary", "passthrough", binary),
        ("w2vec", DummyFootballEmbedder(dim_player=4, dim_team=3), embedding_cols),
    ])

    preprocessor = Pipeline([
        ('feature_engineering', feature_engineering),
        ('column_transform', col_transformer),
    ])

    return preprocessor


# --- Streamlit UI ---

# For demonstration, let's use a dummy classifier
st.markdown("### 🔧🧠 Let's Build a Pipeline Together")

st.markdown("""
### Understanding the Preprocessing Step

Before we can make accurate predictions about the likelihood of a goal (expected goals or xG), we need to **prepare and structure the data properly** — this is called **preprocessing** and where we start our **Pipeline** [ℹ️].

In the context of football, our data contains rich information about each shot: who took it, from where, with what technique, and under what circumstances. But this raw data isn't immediately usable by machine learning models. So we perform several transformations:

- **Embedding Players and Teams**: We use Word2Vec to turn player and team identities into meaningful numeric representations (embeddings), so the model can understand patterns based on individual tendencies or team strategies.
- **Categorical Encoding**: We one-hot encode features like the shot type or the body part used, so the model treats each category independently.
- **Numerical and Spatial Features**: Coordinates like shot location, distance to goal, and angle are passed as-is or used to compute more informative variables.
- **Feature Engineering**: It's all about giving the right context! We derive additional variables (e.g., effective shooting angle, pressure situation, time remaining) to give the model a deeper reasoning behind each shot.
- **Binary Flags**: Simple True/False features like whether the shot was deflected or a first-time effort are also included.

Together, these transformations help the model **"understand"** the context of each shot in a format it can learn from — turning *match events* into *meaningful machine learning features*.

""")

# Tooltip-style explanation of "Pipeline"
with st.expander("[ℹ️] What is a Machine Learning Pipeline?"):
    st.markdown("""
    A **Pipeline** in machine learning is a way to streamline and organize the full modeling process — from data preparation to model training and prediction — a sequence of steps.

    It might include:
    - **Preprocessing** 
    - **Resampling Methods**
    - **Feeding the data into a classifier** to predict expected goals (xG)

    Pipelines ensure **reproducibility**, **modularity**, **no data leak**, and **clean integration** between transformations and models.
    """)



st.markdown("""
### 🤖 Modelling - The Machines are Learning!

In Machine Learning Problems, it's important to correctly tune our models. Let's call it the *art of hypertuning*

""")

st.success("""
           
**1) We first split data into two datasets**

*Why do we split the data at all?*

To make sure our model learns patterns that **generalize to new data**, we split the dataset into:
- **Training set**: used to teach the model.
- **Test set**: held back until the end, used to evaluate how well the model performs on **unseen shots**.


*What’s special about our split?*

We group the data by **player** using a strategy called `GroupShuffleSplit`. This means:
- All shots from a given player appear **only in the training set or only in the test set**, never both.
- This avoids something called **data leakage**, where the model could “cheat” by seeing similar shots from the same player during training.

> Think of it like this: If Lionel Messi is in the training data, he’s completely excluded from the test set — so we’re truly testing the model on new players.

---

**2) What about model tuning and validation?**

Within the training set, we use `Group K-Fold Cross Validation` (5 folds):
- It splits training data into 5 folds, again by player.
- On each fold, the model is trained on 4 folds and validated on 1 — with **no player appearing in both**.

This strategy helps us fine-tune the model while ensuring it's not overfitting to specific players.

---

**3) Fitting the Model**                     

It all comes to minimizing this fancy function (the pot of gold) but don't worry about it
           
$$
\\hat{\\beta} = \\arg\\min_{\\beta} \\frac{1}{n} \\sum_{i=1}^{n} \\mathcal{L}(y_i, \\hat{y}(x_i, \\beta))
$$
           
*What kind of models are we using?*

We don’t rely on a single algorithm — instead, we use a technique called **model stacking**:
- Think of it like assembling a **team of diverse experts**, where each brings a different perspective.
- Some are good at **capturing linear relationships** (like Logistic Regression), others at **handling complex patterns** (like XGBoost or Balanced Random Forests).

Each model makes a prediction, and then a **meta-model** (another Logistic Regression) learns how to best combine these opinions.

*But wait — goals are rare, right?*

Exactly! In our data, most shots are **not goals**, so the model might learn to always predict "no goal" and still be mostly right. That’s not helpful.

To fix this, we use **Resampling Techniques**:
- We **balance the training data** by keeping all the goal events and randomly selecting an equal number of non-goal events. This is called **Random Undersampling**. This forces the model to **pay attention to goals**, rather than just playing it safe.
- Alternatively, we offer the option to use **Random Oversampling**: instead of removing data, we **duplicate goal events** so they’re better represented in training.
- ⚙️ It's up to the user to choose which strategy to apply depending on the task and data size, you can select your preferred Resampling Technique in the toggle bar on your left.

---
           
**4) How does this all come together?**

1. 🧼 We preprocess the data (with embeddings, encodings, engineered features).
2. ⚖️ We balance the data to avoid bias against goals.
3. 🧠 We train **multiple models** in parallel and stack their predictions.
4. 📊 We evaluate using **grouped cross-validation**, where players never leak across folds.
5. ✅ A robust and fair model that learns from context, handles rare events like goals, and combines the strengths of multiple algorithms!

---
**5) What is the output of the Model?**

Machine learning models for classification (like ours) are trained to **distinguish between two outcomes**:  
- **Class 0**: No goal  
- **Class 1**: Goal  

By default, many models can directly predict the class (`0` or `1`). But we usually want more than just a hard decision — we want to know **how confident** the model is.

That’s where `predicting` comes in:

- Instead of giving just a class, the model returns **a probability** belonging to the interval (0,1).
- For each shot, this score represents the **estimated chance that it results in a goal** — this is exactly our **expected goals (xG)** value!

For example:
```python
model.predict(X)        ➜ [0, 0, 1, ...]              # Class predictions  
model.predict_proba(X)  ➜ [[0.95, 0.05], [0.70, 0.30], [0.20, 0.80], ...]
                              ↑     ↑
                     P(no goal)     P(goal)

We extract the second value from each pair — `P(goal)` — to get the final xG.
                                 
""")


#player2vec = gensim.models.Word2Vec(vector_size=4, min_count=1)  # dummy
#team2vec = gensim.models.Word2Vec(vector_size=3, min_count=1)    # dummy

player2vec = None  # Not needed anymore
team2vec = None

# Build preprocessing pipeline
preprocessor_dummy = build_football_preprocessor(player2vec, team2vec)

# Combine preprocessor and model into one pipeline
full_pipeline = Pipeline([
    ('preprocessor', preprocessor_dummy),
    ('classifier', model),
])

with st.expander("📜 Check our Full Pipeline Diagram with a Toy-Example"):
    set_config(display='diagram')
    try:
        pipeline_html = full_pipeline._repr_html_()
        st_html(pipeline_html, height=500, scrolling=True)
    except Exception as e:
        st.warning("Could not render full pipeline diagram. Falling back to text.")
        st.text(str(full_pipeline))


# -------------------------------------
# INTRODUCING THE MODELS FOR USERS
# -------------------------------------

st.markdown("""### 🚀 Deploy - You are ready to Deploy your first ML Model!""")

st.markdown(f"### You are using the Model with: `{selected_model_name} Sampling Strategy`")

# Short explanations for each sampling method
if selected_model_name == "ADASYN":
    st.info("**ADASYN** (Adaptive Synthetic Sampling) creates synthetic samples for minority class examples that are harder to learn — focusing on regions near the decision boundary.")
elif selected_model_name == "BSMOTE":
    st.info("**Borderline-SMOTE** generates synthetic samples only for minority class instances that are near the class boundary, improving decision regions.")
elif selected_model_name == "RENN":
    st.info("**RENN** (Repeated Edited Nearest Neighbours) removes noisy majority samples that are misclassified by their nearest neighbors — a cleaning strategy.")
elif selected_model_name == "ROS":
    st.info("**Random Oversampling** duplicates existing minority class examples to balance the dataset without losing any information.")
elif selected_model_name == "RUS":
    st.info("**Random Undersampling** reduces the majority class by randomly removing examples, simplifying the problem but potentially discarding useful data.")
elif selected_model_name == "SMOTE":
    st.info("**SMOTE** (Synthetic Minority Oversampling Technique) creates synthetic examples of the minority class by interpolating between existing ones.")
elif selected_model_name == "SMOTEENN":
    st.info("**SMOTEENN** combines SMOTE with Edited Nearest Neighbors to both generate new minority samples and clean noisy majority ones.")
elif selected_model_name == "SMOTETomek":
    st.info("**SMOTETomek** merges SMOTE with Tomek links, which removes borderline examples to make the class boundary cleaner and less ambiguous.")
else:
    st.warning("Unknown sampling strategy selected.")

# -------------------------------------

# -------------------------------------
# 📊 Class Balance Plot Section
# -------------------------------------

st.markdown("### 📊 Class Distribution")
st.write("Check how Classes Distribution changes!")

def plot_class_count(y_before, y_after):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['Before: Class 0', 'Before: Class 1'], np.bincount(y_before), color='steelblue', alpha=0.7)
    ax.bar(['After: Class 0', 'After: Class 1'], np.bincount(y_after), color='darkorange', alpha=0.7)
    ax.set_title(f"Class Distribution Before and After {selected_model_name}")
    ax.set_ylabel("Sample Count")
    st.pyplot(fig)

with st.expander(f"📊 Check how Class Distribution changes Before/After {selected_model_name} Technique"):
    try:
        y_before_path = hf_hub_download(
            repo_id="arthurmfs07/xg-dashboard-artifacts",
            filename=f"y_before_{selected_model_name}.csv",
            repo_type="dataset"
        )
        y_after_path = hf_hub_download(
            repo_id="arthurmfs07/xg-dashboard-artifacts",
            filename=f"y_after_{selected_model_name}.csv",
            repo_type="dataset"
        )
        y_before = pd.read_csv(y_before_path).squeeze()
        y_after = pd.read_csv(y_after_path).squeeze()
        plot_class_count(y_before, y_after)
    except FileNotFoundError:
        st.warning(f"Class distribution data not found for model: {selected_model_name}")

# -------------------------------------


# -------------------------------------
# 📊 Evaluate Metrics
# -------------------------------------

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Classification report as dict
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).T.round(4)

    # Basic metrics
    metrics = {
        "Geometric Mean": round(geometric_mean_score(y_test, y_pred), 4),
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4)
    }

    cm = confusion_matrix(y_test, y_pred)

    return metrics, cm, report_df

st.markdown("### 📈 Model Evaluation - Let's choose a Model")

with st.expander(f"📋 Check Evaluation Metrics for {selected_model_name}"):
    try:
        y_true_path = hf_hub_download(
            repo_id="arthurmfs07/xg-dashboard-artifacts",
            filename="y_full_original.csv",
            repo_type="dataset"
        )
        X_clean_path = hf_hub_download(
            repo_id="arthurmfs07/xg-dashboard-artifacts",
            filename="X_full_original.csv",
            repo_type="dataset"
        )

        y_true = pd.read_csv(y_true_path).squeeze()
        X_clean = pd.read_csv(X_clean_path)
        X_transformed = preprocessor.transform(X_clean)

        # Evaluate
        metrics, cm, report_df = evaluate_model(model, X_transformed, y_true)

        st.markdown("""
        When building predictive models, we often face a core dilemma: how to make the model generalize well to new, unseen data.

        This is where the **Bias-Variance Tradeoff** comes in:

        - **Bias** refers to errors due to overly simplistic assumptions in the model. A model with high bias pays too little attention to the training data — it underfits and misses important patterns.
        - **Variance** refers to errors due to excessive sensitivity to small fluctuations in the training set. A high-variance model memorizes the data too closely — it overfits and fails to generalize.

        The **ideal model** finds a balance between the two: it captures the true signal in the data (low bias) while being robust to noise (low variance).

        Choosing the right **model complexity** and applying **resampling, regularization, or ensembling** strategies (like we do with stacking) help us manage this tradeoff.

        Below is a visual that summarizes this balance.
        """)

        st.image(
            r"pics\bias_variance.png",
            caption="Bias-Variance Tradeoff: Understanding model complexity and generalization.",
            use_container_width=False
        )

        st.markdown("""Once the model is trained, we need to **evaluate its performance** on unseen data. Each section below helps us understand different aspects of how well the model is doing:""")

        st.subheader("Overall Metrics")

        st.markdown("""
        - **Geometric Mean**: Useful for imbalanced datasets — it balances performance on both the positive and negative classes.
        - **Accuracy**: The proportion of all predictions the model got right.
        - **AUC (Area Under the ROC Curve)**: Measures how well the model ranks positive vs. negative examples regardless of threshold — the higher, the better.
        """)

        st.write(pd.DataFrame(metrics, index=["Value"]).T)

        fpr, tpr, thresholds = roc_curve(y_true, model.predict_proba(X_transformed)[:, 1])
        # Plot ROC curve
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.4f}")
        ax.plot([0, 1], [0, 1], 'k--', label="Random Guessing")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")

        st.subheader("ROC Curve")

        st.markdown("""
        - This plot shows how the model balances **True Positive Rate (Sensitivity)** against **False Positive Rate** at different classification thresholds.
        - A curve closer to the top-left indicates better performance.
        - The dashed diagonal line represents **random guessing** — so we want our curve well above it.
        """)

        st.pyplot(fig)

        st.subheader("Classification Report")

        st.markdown("""
        - **Precision**: Out of all predicted goals, how many were actually goals?
        - **Recall**: Out of all actual goals, how many did the model catch?
        - **F1-score**: Harmonic mean of precision and recall — balances the two.
        """)

        report_df_no_accuracy = report_df.drop(index=["accuracy"], errors="ignore")
        st.dataframe(report_df_no_accuracy.style.format("{:.4f}").background_gradient(cmap='Blues', axis=1))

        st.subheader("Confusion Matrix")

        st.markdown("""
        - **True Positives (bottom-right)**: Correctly predicted goals.
        - **False Positives (top-right)**: Incorrectly predicted goals.
        - **False Negatives (bottom-left)**: Missed actual goals.
        - **True Negatives (top-left)**: Correctly predicted non-goals.
        """)

        cm_df = pd.DataFrame(cm, columns=["Pred: 0", "Pred: 1"], index=["True: 0", "True: 1"])
        st.dataframe(cm_df.style.background_gradient(cmap='Oranges'))

    except FileNotFoundError:
        st.warning("Evaluation data files not found. Make sure the dataset files are uploaded in the Hugging Face repo.")

    st.markdown("""
### Choosing the Best Model: Focus on Class 1 (Goals)

Our ultimate goal is to build a model that **accurately predicts whether a shot will result in a goal** — in other words, we care most about how the model performs on **class 1**.

Because **goals are rare events** (the positive class is highly imbalanced), we don't want a model that just predicts "no goal" and gets away with a high accuracy. Instead, we focus on metrics that tell us how well the model identifies *actual goals*:

#### Recommended Metrics for Class 1:
- → High recall = fewer missed goals.
- → High F1-Score = if we want both reliability and robustness in goal predictions. 
- → AUC (ROC Curve) = Useful if you plan to use the output probabilities downstream (e.g., for expected goals modeling).

#### How to Choose a Model:
- If your priority is **capturing all potential goals**, go for the model with the **highest recall on class 1**.
- If you want a **balanced model** that still makes confident predictions, use the **F1-score for class 1** as the guiding metric.
- If you're building **ranking systems or visualizations** (e.g., shot quality maps), favor models with the **highest AUC**.

> **In summary:** You should choose your final model based on how well it identifies *real goals* — not just overall accuracy — and match that to your use-case: whether it's precision analysis, ranking chances, or tactical coaching tools.
""")



# -------------------------------------


# Sidebar filters
player_list = sorted(X_full["player"].unique())
team_list = sorted(X_full["team"].unique())

player_selected = st.sidebar.selectbox("Select Player", ["All"] + player_list)
team_selected = st.sidebar.selectbox("Select Team", ["All"] + team_list)

# Apply filters
filtered_data = X_full.copy()

if player_selected != "All":
    filtered_data = filtered_data[filtered_data["player"] == player_selected]

if team_selected != "All":
    filtered_data = filtered_data[filtered_data["team"] == team_selected]

# -------------------------------------



# -------------------------------------
# 📊 DOWNLOADERS
# -------------------------------------

st.info("""### Are you done choosing your model? 
Great, now you can download data summaries based on each player or team according to the model of your preference! Just click the buttons.
""")

def create_player_summary(df):
    # Ensure these columns exist in df: player, team, xG, y_true
    player_summary = (
        df.groupby(["player", "team"])
        .agg(
            total_shots=("xG", "count"),
            total_goals=("y_true", "sum"),
            total_xG=("xG", "sum"),
        )
        .reset_index()
    )

    player_summary["avg_xG_per_shot"] = player_summary["total_xG"] / player_summary["total_shots"]
    player_summary["conversion_rate"] = player_summary["total_goals"] / player_summary["total_shots"]
    player_summary["xG_efficiency"] = player_summary["total_goals"] / player_summary["total_xG"]
    player_summary["overperformance"] = player_summary["total_goals"] - player_summary["total_xG"]
    
    player_summary["big_chances"] = (
        df[df["xG"] > 0.3].groupby("player")["xG"]
        .count()
        .reindex(player_summary["player"])
        .fillna(0)
        .astype(int)
        .values
    )
    player_summary["model_name"] = selected_model_name
    return player_summary


def create_team_summary(df):
    # Ensure these columns exist in df: team, xG, y_true
    team_summary = (
        df.groupby("team")
        .agg(
            total_shots=("xG", "count"),
            total_goals=("y_true", "sum"),
            total_xG=("xG", "sum"),
        )
        .reset_index()
    )

    team_summary["avg_xG_per_shot"] = team_summary["total_xG"] / team_summary["total_shots"]
    team_summary["conversion_rate"] = team_summary["total_goals"] / team_summary["total_shots"]
    team_summary["xG_efficiency"] = team_summary["total_goals"] / team_summary["total_xG"]
    team_summary["overperformance"] = team_summary["total_goals"] - team_summary["total_xG"]
    
    team_summary["big_chances"] = (
        df[df["xG"] > 0.3].groupby("team")["xG"]
        .count()
        .reindex(team_summary["team"])
        .fillna(0)
        .astype(int)
        .values
    )
    team_summary["model_name"] = selected_model_name
    return team_summary

st.markdown("### 📥 Download Summaries")

if st.button("Prepare and Download Player Summary CSV"):
    player_summary = create_player_summary(X_full)
    csv_player = player_summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Click here to download Player xG Summary",
        data=csv_player,
        file_name=f"player_xg_summary_{selected_model_name}.csv",
        mime="text/csv"
    )

if st.button("Prepare and Download Team Summary CSV"):
    team_summary = create_team_summary(X_full)
    csv_team = team_summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Click here to download Team xG Summary",
        data=csv_team,
        file_name=f"team_xg_summary_{selected_model_name}.csv",
        mime="text/csv"
    )
# -------------------------------------

st.info(f"### Check your stats! The player selected is {player_selected}")

# -------------- SUMMARY METRICS --------------

total_shots = len(filtered_data)
total_goals = filtered_data["y_true"].sum()
total_xg = filtered_data["xG"].sum()
conversion = total_goals / total_shots if total_shots > 0 else 0

st.metric("Shots", total_shots)
st.metric("Goals", int(total_goals))
st.metric("Expected Goals (xG)", round(total_xg, 2)) # how many goals the player/team should have scored given the chances
st.metric("Conversion Rate", f"{conversion:.2%}")



# -------------- ADDITIONAL METRICS TABLE --------------
st.subheader("📊 Advanced Shot Metrics")

if total_shots > 0:
    avg_xg_per_shot = total_xg / total_shots
    overperformance = total_goals - total_xg
    big_chances = (filtered_data["xG"] > 0.3).sum()
    conversion_rate = conversion,
    xg_efficiency = total_goals/ total_xg

    metrics_df = pd.DataFrame({
        "Metric": [
            "Average xG per Shot",
            "Overperformance/ Underperformance (Goals - xG)",
            "Number of Big Chances (xG > 0.3)",
            "xG efficiency"
        ],
        "Value": [
            round(avg_xg_per_shot, 3),
            round(overperformance, 2),
            big_chances,
            xg_efficiency
        ]
    })

    st.table(metrics_df)
else:
    st.write("No shots available for selected filters.")



st.info("""### Quick Insights
- **xG**: How many goals the player/team should have scored given the chances?         
- **If Total Goals > xG**, the player or team is finishing better than expected (overperforming their chance quality).
- **If Total Goals < xG**, they are finishing worse than expected (underperforming).
- **Perfect xG efficiency** should be close to 1
            """)


# -------------- SHOT MAP WITH PLOTLY --------------

# Pitch dimensions (StatsBomb standard: 120 x 80)
pitch_length = 120
pitch_width = 80

# Create figure
fig = go.Figure()

# Add pitch layout with lines using shapes
pitch_shapes = [
    # Outer boundaries
    dict(type="rect", x0=0, y0=0, x1=pitch_length, y1=pitch_width,
         line=dict(color="white", width=2), fillcolor="green", layer="below"),
    # Halfway line
    dict(type="line", x0=pitch_length/2, y0=0, x1=pitch_length/2, y1=pitch_width,
         line=dict(color="white", width=2)),
    # Center circle
    dict(type="circle", xref="x", yref="y", x0=pitch_length/2-10, y0=pitch_width/2-10,
         x1=pitch_length/2+10, y1=pitch_width/2+10,
         line=dict(color="white", width=2)),
    # Penalty areas (left)
    dict(type="rect", x0=0, y0=(pitch_width/2)-22, x1=18, y1=(pitch_width/2)+22,
         line=dict(color="white", width=2)),
    dict(type="rect", x0=0, y0=(pitch_width/2)-10, x1=6, y1=(pitch_width/2)+10,
         line=dict(color="white", width=2)),
    # Penalty areas (right)
    dict(type="rect", x0=pitch_length-18, y0=(pitch_width/2)-22, x1=pitch_length, y1=(pitch_width/2)+22,
         line=dict(color="white", width=2)),
    dict(type="rect", x0=pitch_length-6, y0=(pitch_width/2)-10, x1=pitch_length, y1=(pitch_width/2)+10,
         line=dict(color="white", width=2)),
    # Goal areas (left)
    dict(type="rect", x0=0, y0=(pitch_width/2)-7.32/2, x1=2.44, y1=(pitch_width/2)+7.32/2,
         line=dict(color="white", width=2)),
    # Goal areas (right)
    dict(type="rect", x0=pitch_length-2.44, y0=(pitch_width/2)-7.32/2, x1=pitch_length, y1=(pitch_width/2)+7.32/2,
         line=dict(color="white", width=2)),
    # Penalty spots (circles with small radius)
    dict(type="circle", xref="x", yref="y", x0=12-0.3, y0=pitch_width/2-0.3,
         x1=12+0.3, y1=pitch_width/2+0.3,
         line=dict(color="white", width=2), fillcolor="white"),
    dict(type="circle", xref="x", yref="y", x0=pitch_length-12-0.3, y0=pitch_width/2-0.3,
         x1=pitch_length-12+0.3, y1=pitch_width/2+0.3,
         line=dict(color="white", width=2), fillcolor="white"),
]

fig.update_layout(
    shapes=pitch_shapes,
    plot_bgcolor="green",
    xaxis=dict(
        range=[-5, pitch_length + 5],  # give some padding outside
        showgrid=False,
        zeroline=False,
        visible=False,
        scaleanchor="y",
        scaleratio=1
    ),
    yaxis=dict(
        range=[-5, pitch_width + 5],  # padding around the sides
        showgrid=False,
        zeroline=False,
        visible=False
    ),
    margin=dict(l=30, r=30, t=60, b=30),  # more space around the plot
    height=700,  # taller plot
    width=2000,  # wide layout
    title=dict(text="Shot Map on Interactive Pitch (xG colored & sized by probability)", x=0.5, xanchor='center'),
)


# Add shots as scatter points


# Split data
goals = filtered_data[filtered_data["y_true"] == 1]
misses = filtered_data[filtered_data["y_true"] == 0]

# 1. Non-goal shots (circles, xG-colored)
fig.add_trace(go.Scatter(
    x=misses['location_x'],
    y=misses['location_y'],
    mode='markers',
    marker=dict(
        size=misses['xG'] * 40 + 5,
        color=misses['xG'],
        colorscale='Viridis',
        cmin=0,
        cmax=filtered_data['xG'].max(),
        colorbar=dict(title="xG", x=1.02, y=0.5, len=0.75),
        line=dict(width=1, color='black'),
        sizemode='area',
        opacity=0.8,
        symbol='circle'
    ),
    text=misses.apply(lambda row: 
                      f"Player: {row['player']}<br>Team: {row['team']}<br>xG: {row['xG']:.3f}<br>Goal: 0", axis=1),
    hoverinfo='text',
    name="Missed Shots"
))

# 2. Goal shots (stars, red)
fig.add_trace(go.Scatter(
    x=goals['location_x'],
    y=goals['location_y'],
    mode='markers',
    marker=dict(
        size=goals['xG'] * 40 + 5,
        color='red',
        line=dict(width=1, color='black'),
        symbol='star',
        sizemode='area',
        opacity=0.9
    ),
    text=goals.apply(lambda row: 
                     f"Player: {row['player']}<br>Team: {row['team']}<br>xG: {row['xG']:.3f}<br>Goal: 1", axis=1),
    hoverinfo='text',
    name="Goals"
))

# Display in Streamlit
st.plotly_chart(fig)



# -------------- BOXPLOTS --------------
st.markdown("### 📦 xG Distribution by Feature")
st.write("Can we assess how each feature influences the predicted goal probability throughout the Model?")

feature_display_map = {
    'shot_technique': "Shot Technique",
    'shot_type': "Shot Type",
    'shot_body_part': "Shot Body Part",
    'goal_distance': "Goal Distance (m)",
    'shot_angle': "Shot Angle (degrees)",
    'shot_zone_area': "Shot Zone Area (m²)"
}

# Create the full list of feature keys
all_features = list(feature_display_map.keys())

# Streamlit dropdown with display names
selected_feature = st.selectbox(
    "Select a feature for boxplot comparison",
    options=all_features,
    format_func=lambda x: feature_display_map[x]
)

# Copy data for safe plotting
df_plot = X_full.copy()

# Bin numerical features if selected
numerical_features = ['goal_distance', 'shot_angle', 'shot_zone_area']
if selected_feature in numerical_features:
    df_plot[f"{selected_feature}_binned"] = pd.qcut(df_plot[selected_feature], q=4, duplicates='drop')
    x_col = f"{selected_feature}_binned"
else:
    x_col = selected_feature

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df_plot, x=x_col, y="xG", ax=ax)
ax.set_title(f"xG Distribution by {feature_display_map[selected_feature]}", fontsize=14)
ax.set_xlabel(feature_display_map[selected_feature])
ax.set_ylabel("Predicted xG")
plt.xticks(rotation=30)
st.pyplot(fig)



# streamlit run "c:/Users/arthu/OneDrive/Área de Trabalho/statsbomb_arthur/xg_dashboard.py"
#cd "C:/Users/arthu/OneDrive/Área de Trabalho/xg-dashboard"
