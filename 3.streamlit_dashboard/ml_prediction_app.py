import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import requests
import gdown

# Page configuration
st.set_page_config(
    page_title="AI Diet Program Predictions",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stAlert {
        background-color: #e8f4fd;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def download_file_from_google_drive(file_id, dest_path):
    if os.path.exists(dest_path):
        return dest_path
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
    # Check if file is HTML (error page)
    with open(dest_path, 'rb') as f:
        start = f.read(10)
        if start.startswith(b'<html>'):
            raise RuntimeError("Downloaded file is not a valid model file (HTML received). Check Google Drive permissions or quota.")
    return dest_path

# Mapping of model/preprocessor names to Google Drive file IDs
GDRIVE_FILES = {
    'preprocessor.pkl': '1d0BywODBgH5optIg-ZvRtBnBhgJegINS',
    'linear_regression_model.pkl': '1q1hNJz8nuGlbnDfUfBa8n2FKJ_ShEvEV',
    'program_name_mapping.pkl': '1f1PWNbU6_XEdZqoyLnVMvRLinNoWFU9N',
    'program_preprocessor.pkl': '19hNE2brdFQxMj4Et1cV21OiEADBpLfQz',
    'random_forest_classifier_program_model.pkl': '16ZLUGx6gkb2-NmiAvWnIDxUaRH9AOY8d',
    'random_forest_regressor_model.pkl': '1Iomh9fe-Z9O8ifMvFfZWZBbHJMVak7yi',
}

class MLModelManager:
    """Manages all machine learning models for predictions"""
    
    def __init__(self):
        self.models = {}
        self.preprocessors = {}
        self.feature_selectors = {}
        self.model_info = {}
        self.program_id_to_name = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all available models from the ML pipeline directory or Google Drive"""
        try:
            ml_dir = Path(__file__).parent.parent / "4.machine_learning_pipeline"
            ml_dir.mkdir(exist_ok=True)

            # Helper to get local path for a model file, downloading if needed
            def get_model_file(filename):
                file_path = ml_dir / filename
                file_id = GDRIVE_FILES.get(filename)
                if file_id:
                    download_file_from_google_drive(file_id, str(file_path))
                return file_path

            # Load Program Prediction Models
            self._load_program_prediction_models(ml_dir, get_model_file)
            # Load Sales Prediction Models
            self._load_sales_prediction_models(ml_dir, get_model_file)
            # Load Enhanced Models (if any)
            self._load_enhanced_models(ml_dir, get_model_file)
            # Load Program ID to Name Mapping
            self._load_program_mapping(ml_dir, get_model_file)

            if self.models:
                st.success(f"✅ Loaded {len(self.models)} ML models successfully")
            else:
                st.warning("⚠️ No ML models found. Please ensure models are trained and saved.")
        except Exception as e:
            st.error(f"❌ Error loading ML models: {e}")
    
    def _load_program_prediction_models(self, ml_dir, get_model_file):
        """Load program prediction models with robust error handling"""
        try:
            rf_model_path = get_model_file("random_forest_classifier_program_model.pkl")
            if rf_model_path.exists():
                try:
                    self.models['program_prediction_rf'] = joblib.load(rf_model_path)
                except:
                    with open(rf_model_path, 'rb') as f:
                        self.models['program_prediction_rf'] = pickle.load(f)
                self.model_info['program_prediction_rf'] = {
                    'name': 'Random Forest Program Predictor',
                    'type': 'classification',
                    'target': 'diet_program_subscription',
                    'description': 'Predicts which diet program a customer is likely to subscribe to',
                    'accuracy': '85-90%',
                    'features': 'Age, Weight, Height, BMI, Gender (based on actual data structure)'
                }
                st.success(f"✅ Loaded Random Forest model ({rf_model_path.stat().st_size / (1024*1024):.1f} MB)")
            
            # Load program preprocessor
            preprocessor_path = get_model_file("program_preprocessor.pkl")
            if preprocessor_path.exists():
                try:
                    self.preprocessors['program_prediction'] = joblib.load(preprocessor_path)
                except:
                    with open(preprocessor_path, 'rb') as f:
                        self.preprocessors['program_prediction'] = pickle.load(f)
                st.success("✅ Loaded program preprocessor")
            
        except Exception as e:
            st.warning(f"⚠️ Could not load program prediction models: {e}")
    
    def _load_sales_prediction_models(self, ml_dir, get_model_file):
        """Load sales prediction models"""
        try:
            # Define the exact filenames to look for
            model_filenames = ["linear_regression_model.pkl", "random_forest_regressor_model.pkl"]
            
            for model_filename in model_filenames:
                model_path = get_model_file(model_filename)
                if model_path.exists():
                    try:
                        # Use joblib first for efficiency and robustness
                        model = joblib.load(model_path)
                    except Exception as e:
                        try:
                            # Fallback to pickle
                            with open(model_path, 'rb') as f:
                                model = pickle.load(f)
                        except Exception as e:
                            st.warning(f"⚠️ Could not load {model_filename}: {e}")
                            continue

                    model_name_base = model_filename.replace('.pkl', '')
                    self.models[f'sales_prediction_{model_name_base}'] = model
                    self.model_info[f'sales_prediction_{model_name_base}'] = {
                        'name': f'Sales Predictor ({model_name_base})',
                        'type': 'regression',
                        'target': 'sales_amount',
                        'description': 'Predicts sales amounts for customers',
                        'accuracy': 'R²: 0.75-0.85',
                        'features': 'Customer profile and behavior data'
                    }
                    st.success(f"✅ Loaded sales prediction model: {model_name_base}")

            # Load sales preprocessor
            sales_preprocessor_path = get_model_file("preprocessor.pkl")
            if sales_preprocessor_path.exists():
                try:
                    self.preprocessors['sales_prediction'] = joblib.load(sales_preprocessor_path)
                    st.success("✅ Loaded sales preprocessor")
                except Exception as e:
                    st.warning(f"⚠️ Could not load sales preprocessor: {e}")
            else:
                st.warning("⚠️ Sales preprocessor not found. Please ensure 'preprocessor.pkl' is in the ML pipeline directory.")
        
        except Exception as e:
            st.warning(f"⚠️ Could not load sales prediction models: {e}")
    
    def _load_enhanced_models(self, ml_dir, get_model_file):
        """Load enhanced models with robust error handling"""
        try:
            # Load enhanced feature selector
            feature_selector_path = get_model_file("enhanced_feature_selector.pkl")
            if feature_selector_path.exists():
                try:
                    self.feature_selectors['enhanced'] = joblib.load(feature_selector_path)
                except:
                    try:
                        with open(feature_selector_path, 'rb') as f:
                            self.feature_selectors['enhanced'] = pickle.load(f)
                    except Exception as e:
                        st.warning(f"⚠️ Could not load feature selector: {e}")
                        return
                st.success("✅ Loaded enhanced feature selector")
            
            # Load SMOTE preprocessor
            smote_path = get_model_file("smote_preprocessor.pkl")
            if smote_path.exists():
                try:
                    self.preprocessors['smote'] = joblib.load(smote_path)
                except:
                    try:
                        with open(smote_path, 'rb') as f:
                            self.preprocessors['smote'] = pickle.load(f)
                    except Exception as e:
                        st.warning(f"⚠️ Could not load SMOTE preprocessor: {e}")
                        return
                st.success("✅ Loaded SMOTE preprocessor")
                    
        except Exception as e:
            st.warning(f"⚠️ Could not load enhanced models: {e}")

    def _load_program_mapping(self, ml_dir, get_model_file):
        """Loads the program_id to program_name mapping"""
        try:
            mapping_path = get_model_file("program_name_mapping.pkl")
            if mapping_path.exists():
                self.program_id_to_name = joblib.load(mapping_path)
                st.success("✅ Loaded program ID to Name mapping")
            else:
                st.warning("⚠️ Program ID to Name mapping not found. Prediction results will show IDs.")
        except Exception as e:
            st.warning(f"⚠️ Could not load program mapping: {e}")
    
    def _get_feature_names(self, preprocessor):
        """A helper method to get feature names from a preprocessor"""
        feature_names = []
        try:
            for name, transformer, cols in preprocessor.transformers_:
                if name == 'num':
                    feature_names.extend(cols)
                elif name == 'cat':
                    if hasattr(transformer, 'get_feature_names_out'):
                        feature_names.extend(transformer.get_feature_names_out(cols))
                    elif isinstance(transformer, Pipeline):
                        onehot_encoder = transformer.named_steps.get('onehot', None)
                        if onehot_encoder:
                            feature_names.extend(onehot_encoder.get_feature_names_out(cols))
        except:
            pass # Fails gracefully if preprocessor format is unexpected
        return feature_names

    def get_feature_importance(self, model_key):
        """Returns a DataFrame of feature importances for a given model"""
        model = self.models.get(model_key)
        
        if model is None:
            return None, f"Model '{model_key}' not found."
        
        if not hasattr(model, 'feature_importances_'):
            return None, f"Model '{model_key}' does not have feature importances."

        preprocessor = None
        if 'program_prediction_rf' in model_key:
            preprocessor = self.preprocessors.get('program_prediction')
        elif 'sales_prediction_random_forest' in model_key:
            preprocessor = self.preprocessors.get('sales_prediction')

        if preprocessor is None:
            return None, f"Preprocessor for model '{model_key}' not found."

        try:
            feature_names = self._get_feature_names(preprocessor)
            if not feature_names:
                return None, "Could not get feature names from preprocessor."

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df, None
        
        except Exception as e:
            return None, f"Error getting feature importances: {e}"

    def get_available_models(self):
        """Get list of available models"""
        return list(self.models.keys())
    
    def get_model_info(self, model_key):
        """Get information about a specific model"""
        return self.model_info.get(model_key, {})
    
    def predict_program_subscription(self, customer_data):
        """Predict diet program subscription for a customer"""
        try:
            if 'program_prediction_rf' not in self.models:
                return None, "Program prediction model not available"
            
            # Convert dictionary to DataFrame if needed
            if isinstance(customer_data, dict):
                df = pd.DataFrame([customer_data])
            else:
                df = customer_data

            # Add required columns with placeholder values
            required_cols = ['paid', 'paid_amount', 'total_days', 'free_days', 'subscribe_year', 'subscribe_month',
                             'subscribe_day', 'subscribe_quarter', 'delivery_duration_days', 'id_customer',
                             'age', 'height', 'weight', 'birth_year', 'birth_month', 'birth_day',
                             'birth_quarter', 'created_year', 'created_month', 'created_day',
                             'created_quarter', 'gender_encoded', 'bmi', 'status', 'diet_program_name',
                             'master_plan_name', 'subscribe_weekday', 'username', 'email', 'nationality',
                             'gender', 'birth_weekday', 'created_month_name', 'created_weekday',
                             'email_domain']
            
            # Fill missing columns with default/placeholder values
            for col in required_cols:
                if col not in df.columns:
                    if col in ['paid', 'paid_amount', 'total_days', 'free_days', 'delivery_duration_days', 'id_customer']:
                        df[col] = 0
                    elif col in ['subscribe_year', 'subscribe_month', 'subscribe_day', 'subscribe_quarter']:
                        df[col] = 2024
                    elif col in ['birth_year', 'birth_month', 'birth_day', 'birth_quarter',
                                 'created_year', 'created_month', 'created_day', 'created_quarter']:
                        df[col] = 1
                    elif col in ['gender_encoded']:
                        gender_mapping = {'male': 1, 'female': 0, 'unknown': 2}
                        df['gender_encoded'] = df['gender'].map(gender_mapping).fillna(2).astype(int)
                    else:
                        df[col] = 'unknown'

            # Ensure data types are correct before preprocessing
            numeric_cols = ['age', 'weight', 'height', 'bmi', 'paid', 'paid_amount', 'total_days', 'free_days',
                            'delivery_duration_days', 'id_customer', 'subscribe_year', 'subscribe_month',
                            'subscribe_day', 'subscribe_quarter', 'birth_year', 'birth_month', 'birth_day',
                            'birth_quarter', 'created_year', 'created_month', 'created_day',
                            'created_quarter', 'gender_encoded']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            st.info(f"🔍 **Debug Info**: Sending data with shape {df.shape} and columns: {list(df.columns)}")
            st.info(f"🔍 **Debug Info**: Data types: {df.dtypes.to_dict()}")
            st.info(f"🔍 **Debug Info**: Data values: {df.iloc[0].to_dict()}")
            
            # Preprocess the input data
            if 'program_prediction' in self.preprocessors:
                try:
                    st.info("🔄 Attempting to preprocess data...")
                    preprocessed_data = self.preprocessors['program_prediction'].transform(df)
                    st.success("✅ Preprocessing successful")
                except Exception as preprocess_error:
                    st.warning(f"⚠️ Preprocessing failed: {preprocess_error}")
                    return None, f"Preprocessing error: {preprocess_error}"
            else:
                st.info("ℹ️ No preprocessor available. Prediction may fail.")
                preprocessed_data = df
            
            # Make prediction
            st.info("🎯 Making prediction...")
            prediction = self.models['program_prediction_rf'].predict(preprocessed_data)
            probabilities = self.models['program_prediction_rf'].predict_proba(preprocessed_data)
            
            st.success("✅ Prediction successful!")
            
            # Use the mapping to get the program name
            predicted_program_id = int(prediction[0])
            predicted_program_name = self.program_id_to_name.get(predicted_program_id, f"Program {predicted_program_id}")
            
            st.info(f"✅ Predicted Program ID: {predicted_program_id}")
            st.info(f"✅ Predicted Program Name: {predicted_program_name}")

            return predicted_program_name, probabilities[0]
            
        except Exception as e:
            st.error(f"❌ Prediction error details: {e}")
            return None, f"Prediction error: {e}"

    def predict_sales(self, customer_data):
        """Predict sales amount for a customer"""
        try:
            sales_models = [k for k in self.models.keys() if k.startswith('sales_prediction')]
            if not sales_models:
                return None, "Sales prediction models not available"
            
            # Convert dictionary to DataFrame if needed
            if isinstance(customer_data, dict):
                df = pd.DataFrame([customer_data])
            else:
                df = customer_data

            # Add required columns with placeholder values
            required_cols = ['paid', 'total_days', 'free_days', 'subscribe_year', 'subscribe_month',
                             'subscribe_day', 'subscribe_quarter', 'delivery_duration_days', 'id_customer',
                             'age', 'height', 'weight', 'birth_year', 'birth_month', 'birth_day',
                             'birth_quarter', 'created_year', 'created_month', 'created_day',
                             'created_quarter', 'gender_encoded', 'bmi', 'status', 'diet_program_name',
                             'master_plan_name', 'subscribe_weekday', 'username', 'email', 'nationality',
                             'gender', 'birth_weekday', 'created_month_name', 'created_weekday',
                             'email_domain']
            
            # Fill missing columns with default/placeholder values
            for col in required_cols:
                if col not in df.columns:
                    if col in ['paid', 'total_days', 'free_days', 'delivery_duration_days', 'id_customer']:
                        df[col] = 0
                    elif col in ['subscribe_year', 'subscribe_month', 'subscribe_day', 'subscribe_quarter']:
                        df[col] = 2024
                    elif col in ['birth_year', 'birth_month', 'birth_day', 'birth_quarter',
                                 'created_year', 'created_month', 'created_day', 'created_quarter']:
                        df[col] = 1
                    elif col in ['gender_encoded']:
                        gender_mapping = {'male': 1, 'female': 0, 'unknown': 2}
                        df['gender_encoded'] = df['gender'].map(gender_mapping).fillna(2).astype(int)
                    else:
                        df[col] = 'unknown'

            st.info(f"🔍 **Debug Info**: Sending data with shape {df.shape} and columns: {list(df.columns)}")
            st.info(f"🔍 **Debug Info**: Data types: {df.dtypes.to_dict()}")
            st.info(f"🔍 **Debug Info**: Data values: {df.iloc[0].to_dict()}")
            
            # Preprocess the data using the sales preprocessor
            if 'sales_prediction' in self.preprocessors:
                try:
                    st.info("🔄 Attempting to preprocess sales data...")
                    preprocessed_data = self.preprocessors['sales_prediction'].transform(df)
                    st.success("✅ Sales preprocessing successful")
                except Exception as preprocess_error:
                    st.warning(f"⚠️ Sales preprocessing failed: {preprocess_error}")
                    return None, "Sales preprocessing failed"
            else:
                st.warning("⚠️ Sales preprocessor not available. Cannot make prediction.")
                return None, "Sales preprocessor not available"

            # Use the first available sales model
            model_key = sales_models[0]
            prediction = self.models[model_key].predict(preprocessed_data)
            
            return prediction[0], None
            
        except Exception as e:
            st.error(f"❌ Sales prediction error details: {e}")
            return None, f"Sales prediction error: {e}"

def create_customer_input_form():
    """Create a comprehensive form for customer input data based on actual data"""
    st.subheader("📝 Customer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Personal Information**")
        age = st.number_input("Age", min_value=0, max_value=125, value=30, help="Customer's age in years")
        weight = st.number_input("Weight (kg)", min_value=37.0, max_value=128.0, value=70.0, help="Customer's weight in kilograms")
        height = st.number_input("Height (cm)", min_value=139.0, max_value=196.0, value=170.0, help="Customer's height in centimeters")
        gender = st.selectbox("Gender", ["male", "female", "unknown"], help="Customer's gender")
        
    with col2:
        st.write("**Calculated Metrics**")
        # Calculate BMI
        bmi = weight / ((height / 100) ** 2)
        
        # Display calculated BMI
        st.info(f"📊 **Calculated BMI**: {bmi:.1f}")
        
        # BMI Category
        if bmi < 18.5:
            bmi_category = "Underweight"
            color = "🔵"
        elif bmi < 25:
            bmi_category = "Normal"
            color = "🟢"
        elif bmi < 30:
            bmi_category = "Overweight"
            color = "🟡"
        else:
            bmi_category = "Obese"
            color = "🔴"
        
        st.info(f"{color} **BMI Category**: {bmi_category}")
        
        # Additional info
        st.info("💡 **Note**: This form uses only the fields available in your actual data")
    
    # Create customer data dictionary matching your real data structure
    customer_data = {
        'age': age,
        'weight': weight,
        'height': height,
        'bmi': bmi,
        'gender': gender
    }
    
    return customer_data

def display_prediction_results(prediction, probabilities, model_info, program_id_to_name):
    """Display prediction results in a beautiful format"""
    st.subheader("🎯 AI Prediction Results")
    
    if prediction is not None:
        # Create metrics row
        col1, col2, col3 = st.columns(3)
        
        # Get all program IDs from the mapping and sort them
        all_program_ids = sorted(list(program_id_to_name.keys()))
        
        with col1:
            st.metric(
                label="🤖 Predicted Program",
                value=prediction,
                delta="AI Recommendation"
            )
        
        with col2:
            if probabilities is not None:
                # Create a list of tuples (program_id, probability)
                prob_with_labels = [(all_program_ids[i], prob) for i, prob in enumerate(probabilities)]
                prob_with_labels.sort(key=lambda x: x[1], reverse=True)
                
                max_prob = prob_with_labels[0][1]
                confidence = max_prob * 100
                st.metric(
                    label="🎯 Confidence",
                    value=f"{confidence:.1f}%",
                    delta="Prediction Confidence"
                )
        
        with col3:
            if probabilities is not None:
                # Use the sorted list from the previous step
                second_choice_id = prob_with_labels[1][0]
                second_choice_name = program_id_to_name.get(second_choice_id, f"Program {second_choice_id}")
                second_prob = prob_with_labels[1][1] * 100
                st.metric(
                    label="🥈 Second Choice",
                    value=second_choice_name,
                    delta=f"{second_prob:.1f}% confidence"
                )
        
        # Display probabilities chart
        if probabilities is not None:
            st.subheader("📊 Prediction Probabilities")
            
            # Create probability dataframe
            data = []
            for i, prob in enumerate(probabilities):
                program_id = all_program_ids[i]
                program_name = program_id_to_name.get(program_id, f"Program {program_id}")
                data.append({
                    'Program': program_name,
                    'Probability': prob,
                    'Percentage': f"{prob*100:.1f}%"
                })

            prob_df = pd.DataFrame(data).sort_values('Probability', ascending=False)
            
            # Create a beautiful bar chart
            fig = px.bar(prob_df, x='Program', y='Probability', 
                         title="Program Subscription Probabilities",
                         color='Probability',
                         color_continuous_scale='viridis',
                         text='Percentage',
                         height=400)
            
            fig.update_traces(textposition='outside')
            fig.update_layout(
                xaxis_title="Diet Programs",
                yaxis_title="Probability",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display detailed breakdown
            st.subheader("📋 Detailed Breakdown")
            for i, row in prob_df.iterrows():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{row['Program']}**")
                with col2:
                    st.write(f"{row['Percentage']}")
                with col3:
                    # Progress bar
                    st.progress(row['Probability'])
    
    else:
        st.error("❌ Unable to make prediction. Please check your input data.")

def display_sales_prediction(sales_prediction):
    """Display sales prediction results"""
    if sales_prediction is not None:
        st.subheader("💰 Sales Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="💵 Predicted Sales",
                value=f"${sales_prediction:.2f}",
                delta="Expected Revenue"
            )
        
        with col2:
            # Determine customer segment
            if sales_prediction > 100:
                segment = "High Value"
                delta_color = "normal"
            elif sales_prediction > 50:
                segment = "Medium Value"
                delta_color = "normal"
            else:
                segment = "Low Value"
                delta_color = "inverse"
            
            st.metric(
                label="👤 Customer Segment",
                value=segment,
                delta="Value Classification"
            )
        
        with col3:
            # Calculate potential monthly revenue
            monthly_potential = sales_prediction * 4  # Assuming weekly programs
            st.metric(
                label="📈 Monthly Potential",
                value=f"${monthly_potential:.2f}",
                delta="Revenue Potential"
            )
        
        # Create a gauge chart for sales prediction
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = sales_prediction,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Sales Prediction Gauge"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 200]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "yellow"},
                    {'range': [100, 200], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 150
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("❌ Unable to predict sales amount.")


def main():
    """Main application function"""
    st.markdown('<h1 class="main-header">🤖 AI Diet Program Predictions</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="prediction-card">
    <h3>🎯 Welcome to AI-Powered Diet Program Predictions</h3>
    <p>This application uses advanced machine learning models to predict:</p>
    <ul>
        <li><strong>Diet Program Recommendations:</strong> Which program a customer is most likely to subscribe to</li>
        <li><strong>Sales Predictions:</strong> Expected sales amounts for customers</li>
        <li><strong>Customer Insights:</strong> Detailed analysis and feature importance</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize ML models
    with st.spinner("🤖 Loading AI models..."):
        ml_manager = MLModelManager()
    
    # Check if models are available
    if not ml_manager.get_available_models():
        st.error("❌ No ML models found. Please ensure models are trained and saved in the 4.machine_learning_pipeline directory.")
        st.info("💡 Required files:")
        st.markdown("""
        - `random_forest_classifier_program_model.pkl` (Program prediction model)
        - `program_preprocessor.pkl` (Data preprocessor)
        - `enhanced_feature_selector.pkl` (Feature selector)
        - `smote_preprocessor.pkl` (SMOTE preprocessor)
        """)
        return
    
    # Sidebar navigation
    st.sidebar.title("🤖 AI Predictions")
    page = st.sidebar.selectbox(
        "Choose Prediction Type:",
        ["🎯 Program Prediction", "💰 Sales Prediction"]
    )
    
    # Program Prediction Page
    if page == "🎯 Program Prediction":
        st.subheader("🎯 Diet Program Subscription Prediction")
        st.write("Enter customer information to predict which diet program they're most likely to subscribe to.")
        
        # Get customer input
        customer_data = create_customer_input_form()
        
        # Prediction button
        if st.button("🔮 Predict Program Subscription", type="primary", use_container_width=True):
            with st.spinner("🤖 AI is analyzing customer profile..."):
                # Convert to DataFrame for prediction
                customer_df = pd.DataFrame([customer_data])
                
                # Make prediction
                prediction, probabilities = ml_manager.predict_program_subscription(customer_df)
                
                # Display results
                display_prediction_results(prediction, probabilities, ml_manager.get_model_info('program_prediction_rf'), ml_manager.program_id_to_name)
    
    # Sales Prediction Page
    elif page == "💰 Sales Prediction":
        st.subheader("💰 Sales Amount Prediction")
        st.write("Predict the expected sales amount for a customer based on their profile.")
        
        # Get customer input for sales prediction
        sales_customer_data = create_customer_input_form()
        
        # Prediction button
        if st.button("💰 Predict Sales Amount", type="primary", use_container_width=True):
            with st.spinner("🤖 AI is calculating sales potential..."):
                # Convert to DataFrame for prediction
                sales_customer_df = pd.DataFrame([sales_customer_data])
                
                # Make prediction
                sales_prediction, _ = ml_manager.predict_sales(sales_customer_df)
                
                # Display results
                display_sales_prediction(sales_prediction)
    
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**AI Features:**")
    st.sidebar.markdown("• 🎯 Program predictions")
    st.sidebar.markdown("• 💰 Sales forecasting")

if __name__ == "__main__":
    main() 