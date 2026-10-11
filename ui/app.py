"""
Streamlit UI for Disease Triage Demo.
Fixed symptom-count mismatch and loads encoders from checkpoints.
"""
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import yaml
import pickle
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import RGBEncoder, ThermalEncoder, TabularEncoder, FusionEncoder, FusionModel
from utils import get_device


st.set_page_config(
    page_title="Disease Triage Demo",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.disclaimer {
    background-color: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 5px;
    padding: 15px;
    margin: 10px 0;
    color: #856404;
}
.result-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 5px;
    padding: 15px;
    margin: 10px 0;
}
.warning-box {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    border-radius: 5px;
    padding: 15px;
    margin: 10px 0;
}
.info-box {
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
    border-radius: 5px;
    padding: 15px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_encoders():
    """Load symptom and disease encoders from tabular checkpoint."""
    try:
        with open("checkpoints/tabular/disease_encoder.pkl", 'rb') as f:
            disease_encoder = pickle.load(f)
        with open("checkpoints/tabular/symptom_encoder.pkl", 'rb') as f:
            symptom_encoder = pickle.load(f)
        return disease_encoder, symptom_encoder
    except Exception as e:
        st.warning(f"Could not load encoders: {e}")
        return None, None


@st.cache_resource
def load_model(config_path: str, checkpoint_path: str, disease_encoder, symptom_encoder):
    """Load the fusion model with proper dimensions from encoders."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = get_device()
    
    # Get dimensions from encoders
    num_visual_classes = 7  # HAM10000 classes
    num_triage_classes = 3
    num_symptoms = len(symptom_encoder.classes_) if symptom_encoder else 131
    
    # Load encoders info from saved checkpoints
    rgb_encoder = RGBEncoder(
        pretrained=False,
        frozen=False,
        output_dim=config['model']['rgb_output_dim'],
        num_classes=num_visual_classes
    )
    
    thermal_encoder = ThermalEncoder(
        pretrained=False,
        frozen=False,
        output_dim=config['model']['thermal_output_dim'],
        in_channels=1
    )
    
    tabular_encoder = TabularEncoder(
        input_dim=num_symptoms,
        hidden_dims=config['model']['tabular_hidden_dims'],
        output_dim=config['model']['tabular_output_dim'],
        dropout=config['model'].get('dropout', 0.3)
    )
    
    fusion_encoder = FusionEncoder(
        rgb_dim=config['model']['rgb_output_dim'],
        thermal_dim=config['model']['thermal_output_dim'],
        tabular_dim=config['model']['tabular_output_dim'],
        fusion_dim=config['model']['fusion_dim'],
        num_heads=config['model'].get('num_heads', 4),
        dropout=config['model'].get('dropout', 0.3)
    )
    
    model = FusionModel(
        rgb_encoder=rgb_encoder,
        thermal_encoder=thermal_encoder,
        tabular_encoder=tabular_encoder,
        fusion_encoder=fusion_encoder,
        num_visual_classes=num_visual_classes,
        num_triage_classes=3
    ).to(device)
    
    # Try to load checkpoint
    if Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        st.success(f"Loaded model from {checkpoint_path}")
    else:
        st.warning(f"Checkpoint not found at {checkpoint_path}. Using random weights.")
    
    model.eval()
    return model, device, config, num_visual_classes


def preprocess_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Preprocess RGB image."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


def preprocess_thermal(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Preprocess thermal image (grayscale)."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    # Convert to grayscale if RGB
    if image.mode == 'RGB':
        image = image.convert('L')
    return transform(image).unsqueeze(0)


def preprocess_symptoms(symptoms: list, symptom_encoder) -> torch.Tensor:
    """Preprocess symptoms to multi-hot vector."""
    if symptom_encoder is None:
        return torch.zeros(1, 131)
    features = symptom_encoder.transform([symptoms]).astype(np.float32)
    return torch.tensor(features)


# App Header
st.title("🏥 Disease Triage — Multimodal Demo")
st.markdown("""
This demo uses a multimodal model (RGB image + Thermal image + Symptoms) to predict:
1. **Visual Condition** - Skin lesion classification (7 classes)
2. **Fever Flag** - Binary fever indicator
3. **Triage Level** - Mild / Moderate / Refer
""")

# DISCLAIMER - Always visible
st.markdown("""
<div class="disclaimer">
<strong>⚠️ MEDICAL DISCLAIMER</strong><br>
This is a <strong>decision-support demo</strong> for research/educational purposes only. 
It is <strong>NOT a medical diagnosis</strong>. The model was trained on limited datasets 
with known limitations (especially thermal data uses a proxy). 
<strong>Always consult a qualified healthcare professional</strong> for medical advice.
</div>
""", unsafe_allow_html=True)

# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")
config_path = st.sidebar.text_input("Config Path", "configs/fusion_fast.yaml")
checkpoint_path = st.sidebar.text_input("Checkpoint Path", "checkpoints/fusion/best_model.pt")

# Load encoders first
disease_encoder, symptom_encoder = load_encoders()

# Load model
model, device, config, num_visual_classes = load_model(config_path, checkpoint_path, disease_encoder, symptom_encoder)

# Get visual classes from config or use default
visual_classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
triage_classes = ['mild', 'moderate', 'refer']

# Main input area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Visual Input")
    
    # RGB Image
    st.subheader("RGB Image (Skin/Lesion Photo)")
    rgb_file = st.file_uploader("Upload RGB image", type=['jpg', 'jpeg', 'png'])
    
    if rgb_file:
        rgb_image = Image.open(rgb_file).convert('RGB')
        st.image(rgb_image, caption="Uploaded RGB Image", width=300)
        rgb_tensor = preprocess_image(rgb_image, config['data']['image_size']).to(device)
    else:
        # Use zero tensor (missing modality will be handled by masking)
        rgb_tensor = torch.zeros(1, 3, config['data']['image_size'], config['data']['image_size']).to(device)
        st.info("No RGB image uploaded. Modality will be masked.")
    
    # Thermal Image
    st.subheader("Thermal Image (Face)")
    thermal_file = st.file_uploader("Upload thermal image", type=['jpg', 'jpeg', 'png'])
    
    if thermal_file:
        thermal_image = Image.open(thermal_file)
        st.image(thermal_image, caption="Uploaded Thermal Image", width=300)
        thermal_tensor = preprocess_thermal(thermal_image, config['data']['image_size']).to(device)
    else:
        thermal_tensor = torch.zeros(1, 1, config['data']['image_size'], config['data']['image_size']).to(device)
        st.info("No thermal image uploaded. Modality will be masked.")

with col2:
    st.header("📋 Symptoms & Vitals")
    
    # Temperature
    temperature = st.number_input("Body Temperature (°F)", min_value=90.0, max_value=110.0, value=98.6, step=0.1)
    
    # Fever checkboxes
    st.subheader("Fever Symptoms")
    has_high_fever = st.checkbox("High Fever")
    has_mild_fever = st.checkbox("Mild Fever")
    has_chills = st.checkbox("Chills")
    has_shivering = st.checkbox("Shivering")
    
    # Other symptoms - load from encoder
    if symptom_encoder is not None:
        all_symptoms = symptom_encoder.classes_.tolist()
    else:
        # Fallback
        all_symptoms = [
            'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering',
            'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting',
            'vomiting', 'burning_micturition', 'spotting_urination', 'fatigue', 'weight_gain',
            'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
            'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever',
            'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache',
            'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
            'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
            'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
            'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm',
            'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion',
            'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements',
            'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness',
            'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels',
            'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
            'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
            'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck',
            'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance',
            'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort',
            'foul_smell_ofurine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
            'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium',
            'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic_patches',
            'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history',
            'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
            'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma',
            'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption',
            'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking',
            'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting',
            'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
            'yellow_crust_ooze', 'prognosis'
        ]
    
    # Fever checkboxes
    st.subheader("Fever Symptoms")
    has_high_fever = st.checkbox("High Fever")
    has_mild_fever = st.checkbox("Mild Fever")
    has_chills = st.checkbox("Chills")
    has_shivering = st.checkbox("Shivering")
    
    # Auto-add fever symptoms based on checkboxes
    selected_symptoms = []
    if has_high_fever:
        selected_symptoms.append('high_fever')
    if has_mild_fever:
        selected_symptoms.append('mild_fever')
    if has_chills:
        selected_symptoms.append('chills')
    if has_shivering:
        selected_symptoms.append('shivering')
    
    # Multi-select for other symptoms
    other_symptoms = st.multiselect(
        "Select other symptoms",
        [s for s in all_symptoms if s not in selected_symptoms],
        default=[]
    )
    
    selected_symptoms.extend(other_symptoms)
    
    # Preprocess tabular
    tabular_tensor = preprocess_symptoms(selected_symptoms, symptom_encoder).to(device)

# Run inference
if st.button("🔍 Run Triage", type="primary", use_container_width=True):
    with st.spinner("Running inference..."):
        with torch.no_grad():
            outputs = model(rgb_tensor, thermal_tensor, tabular_tensor)
            
            # Get predictions
            fever_prob = torch.sigmoid(outputs['fever']).item()
            fever_pred = "Fever Detected" if fever_prob > 0.5 else "No Fever"
            
            visual_probs = torch.softmax(outputs['visual'], dim=1)[0]
            visual_idx = visual_probs.argmax().item()
            visual_conf = visual_probs[visual_idx].item()
            visual_pred = visual_classes[visual_idx]
            
            triage_probs = torch.softmax(outputs['triage'], dim=1)[0]
            triage_idx = triage_probs.argmax().item()
            triage_conf = triage_probs[triage_idx].item()
            triage_pred = triage_classes[triage_idx]
    
    # Display results
    st.header("📊 Results")
    
    # Fever
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        if fever_prob > 0.5:
            st.markdown(f"""
            <div class="warning-box">
                <strong>🌡️ Fever Flag: {fever_pred}</strong><br>
                Probability: {fever_prob:.1%}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box">
                <strong>🌡️ Fever Flag: {fever_pred}</strong><br>
                Probability: {fever_prob:.1%}
            </div>
            """, unsafe_allow_html=True)
    
    # Visual Condition
    with col_f2:
        st.markdown(f"""
        <div class="info-box">
            <strong>🔬 Visual Condition: {visual_pred.upper()}</strong><br>
            Confidence: {visual_conf:.1%}
        </div>
        """, unsafe_allow_html=True)
    
    # Triage
    st.subheader("🚑 Overall Triage")
    triage_colors = {'mild': 'result-box', 'moderate': 'warning-box', 'refer': 'warning-box'}
    triage_icons = {'mild': '🟢', 'moderate': '🟡', 'refer': '🔴'}
    
    st.markdown(f"""
    <div class="{triage_colors[triage_pred]}">
        <strong>{triage_icons[triage_pred]} Triage Level: {triage_pred.upper()}</strong><br>
        Confidence: {triage_conf:.1%}
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed probabilities
    with st.expander("📈 Detailed Probabilities"):
        col_p1, col_p2, col_p3 = st.columns(3)
        
        with col_p1:
            st.write("**Fever:**")
            st.write(f"No Fever: {1-fever_prob:.1%}")
            st.write(f"Fever: {fever_prob:.1%}")
        
        with col_p2:
            st.write("**Visual Condition:**")
            for i, (cls, prob) in enumerate(zip(visual_classes, visual_probs)):
                st.write(f"{cls}: {prob:.1%}")
        
        with col_p3:
            st.write("**Triage:**")
            for i, (cls, prob) in enumerate(zip(triage_classes, triage_probs)):
                st.write(f"{cls}: {prob:.1%}")

# Footer
st.markdown("---")
st.markdown("""
<div class="disclaimer">
<strong>⚠️ REMINDER:</strong> This is a research demo only. 
The thermal encoder uses a public face dataset (Terravic) without fever labels. 
The fever prediction is based on symptom proxy from tabular data. 
<strong>Do not use for clinical decision-making.</strong>
</div>
""", unsafe_allow_html=True)