import streamlit as st
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error as MSE, r2_score

MAIN_DIR = Path(__file__).resolve().parent 
PIPELINE_PATH = MAIN_DIR / 'prediction_pipeline.pkl'

#придется здесь тоже класс обозначать :(
class CombinedPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, missing_cols, medians, impute_columns):
        self.missing_cols = missing_cols
        self.medians = medians
        self.impute_columns = impute_columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if 'name' in X.columns:
            X['name'] = X['name'].str.split().str[0]
    
        if 'torque' in X.columns:
            conv_ = np.where(X['torque'].str.contains('nm', case=False, na=False), 9.80665, 1)
        
        parse_cols = [c for c in self.missing_cols if c != 'seats']
        for c in parse_cols:
            if c != 'torque':
                X[c] = X[c].str.split().str[0]
            else:
                X[c] = X[c].str.extract(r'([0-9]*\.?[0-9]+)', expand=False)
        
        for col in self.missing_cols:
            X[col] = pd.to_numeric(X[c], errors='coerce')
        
        if 'torque' in X.columns:
            X['torque'] /= conv_
        
        for c in self.impute_columns:
            if c in X.columns:
                X[c] = X[c].fillna(self.medians[c])

        for c in ['engine', 'seats']:
            if c in X.columns:
                X[c] = X[c].astype(int, errors='ignore')
        
        return X

@st.cache_resource
def load_pipeline():
    return joblib.load(PIPELINE_PATH)

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def display_model_details(pipeline):
    model = pipeline.named_steps['model']
    scaler = pipeline.named_steps['scaler']
    ohe_cols = pipeline.named_steps['col_transform'].named_transformers_['ohe'].get_feature_names_out()
    feature_names = list(ohe_cols) + pipeline.named_steps['col_transform'].named_transformers_['passthrough'].feature_names_in_
    
    st.sidebar.title("Model Details")
    
    st.sidebar.subheader("Model Weights (Coefficients)")
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    })
    st.sidebar.dataframe(coef_df)
    
    st.sidebar.subheader("Feature Importance (Abs Coefficient)")
    importance_df = coef_df.copy()
    importance_df['Abs Coefficient'] = np.abs(importance_df['Coefficient'])
    importance_df = importance_df.sort_values(by='Abs Coefficient', ascending=False)
    st.sidebar.dataframe(importance_df)
    
    st.sidebar.subheader("Intercept")
    st.sidebar.write(model.intercept_)

def prepare_viz_data(df):
    df_viz = df.copy()
    if 'year' in df_viz.columns:
        df_viz['age'] = df_viz['year'].max() - df_viz['year'] + 1
        df_viz['km_per_year'] = df_viz['km_driven'] / df_viz['age']
    if 'selling_price' in df_viz.columns:
        df_viz = df_viz[(df_viz['selling_price'] < 5e6)]
    if 'km_per_year' in df_viz.columns:
        df_viz['usage_intensity'] = pd.cut(
            df_viz['km_per_year'], 
            bins=[0, 8000, 16000, 25000, np.inf], 
            labels=['<8k/yr', '8-16k/yr', '16-25k/yr', '>25k/yr']
        )
    return df_viz

def display_data_overview(df):
    st.subheader("Data Sample")
    st.write(df.sample(30))
    
    st.subheader("Data Head")
    st.write(df.head(5))
    
    st.subheader("Data Tail")
    st.write(df.tail(5))
    
    st.subheader("Missing Value Columns")
    st.write(df.columns[df.isnull().any()])
    
    st.subheader("Numeric Description")
    st.write(df.describe())
    
    st.subheader("Object Description")
    st.write(df.describe(include='object'))

def display_visualizations(df, df_viz):
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    st.subheader("Pairplot (Numeric Columns)")
    fig_pair = sns.pairplot(df[num_cols])
    st.pyplot(fig_pair)
    
    st.subheader("Correlation Heatmap")
    with st.container():
        fig_heat, ax_heat = plt.subplots()
        sns.heatmap(df[num_cols].corr(), ax=ax_heat)
        st.pyplot(fig_heat)
    
    st.subheader("Violin Plot: Selling Price by Usage Intensity and Fuel")
    if all(col in df_viz.columns for col in ['usage_intensity', 'selling_price', 'fuel']):
        with st.container():
            fig_violin, ax_violin = plt.subplots(figsize=(16, 8))
            sns.violinplot(
                data=df_viz,
                x='usage_intensity',
                y='selling_price',
                hue='fuel',
                split=True,
                palette={'Diesel': '#1f77b4', 'Petrol': '#ff7f0e', 'CNG': 'red', 'LPG': 'green'},
                inner='quartile',
                ax=ax_violin
            )
            ax_violin.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_violin)
    
    st.subheader("Scatter Plot: Year vs Price colored by km_driven (log)")
    if all(col in df.columns for col in ['year', 'selling_price', 'km_driven']):
        with st.container():
            fig_scatter, ax_scatter = plt.subplots(figsize=(12, 6))
            scatter = ax_scatter.scatter(
                df['year'], 
                df['selling_price'], 
                c=df['km_driven'], 
                norm=LogNorm(vmin=df['km_driven'].min(), vmax=df['km_driven'].max()),
                s=60
            )
            cbar = plt.colorbar(scatter, ax=ax_scatter)
            cbar.set_label('km_driven(log)')
            ax_scatter.set_xlabel('year')
            ax_scatter.set_ylabel('price')
            plt.tight_layout()
            st.pyplot(fig_scatter)
    
    st.subheader("Relplot: Year vs Price by Fuel, Owner, Seller Type")
    if all(col in df.columns for col in ['year', 'selling_price', 'fuel', 'owner', 'seller_type']):
        fig_rel = sns.relplot(
            data=df,
            x='year',
            y='selling_price',
            col='fuel',
            hue='owner',
            style='seller_type',
            col_wrap=2,
            palette='viridis',
            alpha=0.8,
        )
        st.pyplot(fig_rel)

def display_predictions_and_metrics(pipeline, df):
    predictions = pipeline.predict(df)
    st.subheader("Predictions")
    st.write(pd.DataFrame({'Predictions': predictions}, index=df.index))
    
    if 'selling_price' in df.columns:
        y_true = df['selling_price']
        st.subheader("Metrics")
        st.write(f'MSE: {MSE(y_true, predictions)}')
        st.write(f'R^2: {r2_score(y_true, predictions)}')
        st.write(f"Business Metric: {np.mean((np.abs(predictions - y_true) / y_true) <= 0.10)}")

# Main app
st.title("Car Price Prediction App with EDA")

pipeline = load_pipeline()
display_model_details(pipeline)

uploaded_file = st.file_uploader("Upload CSV for Predictions/EDA", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    df_viz = prepare_viz_data(df)
    
    display_data_overview(df)
    display_visualizations(df, df_viz)
    display_predictions_and_metrics(pipeline, df)
else:

    st.write("Upload a CSV file to see EDA, visualizations, and predictions.")




