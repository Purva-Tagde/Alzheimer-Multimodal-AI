
import os
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import pydicom

from PIL import Image
from torchvision.models import resnet18
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AlzheimerAI | MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEVICE = torch.device("cpu")

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "adni_resnet18_5slice_baseline.pth"
)


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */
    .stApp {
        background:
            radial-gradient(circle at 5% 5%, rgba(236,72,153,0.10), transparent 25%),
            radial-gradient(circle at 95% 10%, rgba(99,102,241,0.12), transparent 25%),
            linear-gradient(180deg, #f8fbff 0%, #ffffff 45%, #f8fafc 100%);
    }

    .main {
        padding-top: 0.8rem;
    }

    /* ---------- HERO ---------- */
    .hero {
        padding: 2.2rem 2.4rem;
        border-radius: 24px;
        background:
            radial-gradient(circle at 85% 20%, rgba(255,255,255,0.24), transparent 20%),
            linear-gradient(135deg, #4f46e5 0%, #7c3aed 35%, #db2777 72%, #f97316 100%);
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 12px 35px rgba(99,102,241,0.22);
        position: relative;
        overflow: hidden;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 180px;
        height: 180px;
        right: -45px;
        bottom: -80px;
        border-radius: 50%;
        background: rgba(255,255,255,0.12);
    }

    .hero h1 {
        font-size: 2.7rem;
        margin-bottom: 0.35rem;
        letter-spacing: -0.03em;
    }

    .hero p {
        color: #fdf4ff;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    /* ---------- SECTION HEADINGS ---------- */
    .section-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #312e81;
        margin-top: 1.4rem;
        margin-bottom: 0.65rem;
    }

    /* ---------- METRIC CARDS ---------- */
    .metric-card {
        padding: 1.25rem;
        border-radius: 18px;
        background: linear-gradient(145deg, #ffffff, #f5f3ff);
        border: 1px solid #ddd6fe;
        text-align: center;
        min-height: 125px;
        box-shadow: 0 8px 22px rgba(79,70,229,0.08);
    }

    .metric-card:nth-child(1) {
        border-top: 4px solid #6366f1;
    }

    .metric-title {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 0.35rem;
        font-weight: 600;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #312e81;
    }

    /* ---------- RESULT ---------- */
    .result-card {
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid #c4b5fd;
        background: linear-gradient(
            135deg,
            #eef2ff,
            #f5f3ff,
            #fce7f3
        );
        text-align: center;
        margin: 0.5rem 0 1.1rem 0;
        box-shadow: 0 10px 28px rgba(124,58,237,0.10);
    }

    .result-label {
        color: #6b21a8;
        font-size: 0.9rem;
        font-weight: 600;
    }

    .result-value {
        font-size: 2.5rem;
        font-weight: 900;
        color: #4338ca;
    }

    /* ---------- UPLOAD AREA ---------- */
    [data-testid="stFileUploader"] {
        border: 2px dashed #a78bfa;
        border-radius: 18px;
        padding: 0.4rem;
        background: linear-gradient(
            135deg,
            rgba(238,242,255,0.8),
            rgba(253,242,248,0.8)
        );
    }

    /* ---------- ALERT / DISCLAIMER ---------- */
    .disclaimer {
        padding: 1rem 1.1rem;
        border-radius: 14px;
        background: linear-gradient(
            90deg,
            #fff7ed,
            #fef3c7
        );
        border: 1px solid #fdba74;
        color: #9a3412;
        font-size: 0.88rem;
        box-shadow: 0 5px 18px rgba(249,115,22,0.08);
    }

    /* ---------- SIDEBAR ---------- */
    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #eef2ff 0%,
                #f5f3ff 45%,
                #fdf2f8 100%
            );
        border-right: 1px solid #ddd6fe;
    }

    [data-testid="stSidebar"] h2 {
        color: #4338ca;
    }

    /* ---------- TABS ---------- */
    button[data-baseweb="tab"] {
        font-weight: 700;
        color: #6366f1;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #db2777;
    }

    /* ---------- INFO BOX ---------- */
    [data-testid="stAlert"] {
        border-radius: 14px;
    }

    /* ---------- DATAFRAME ---------- */
    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #ddd6fe;
    }

    /* ---------- EXPANDER ---------- */
    [data-testid="stExpander"] {
        border: 1px solid #ddd6fe;
        border-radius: 14px;
        background: rgba(255,255,255,0.75);
    }

    /* ---------- BUTTONS ---------- */
    .stButton > button {
        border-radius: 12px;
        border: 1px solid #a78bfa;
        color: #4338ca;
        background: linear-gradient(
            90deg,
            #eef2ff,
            #fce7f3
        );
        font-weight: 700;
    }

    .stButton > button:hover {
        border-color: #db2777;
        color: #be185d;
        box-shadow: 0 5px 15px rgba(219,39,119,0.15);
    }

    .small-note {
        color: #64748b;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        in_channels=5,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# PREPROCESSING
# ============================================================

def normalize_array(array):

    array = array.astype(np.float32)

    low = np.percentile(array, 1)
    high = np.percentile(array, 99)

    if high <= low:
        low = array.min()
        high = array.max()

    array = np.clip(array, low, high)

    if high > low:
        array = (array - low) / (high - low)
    else:
        array = np.zeros_like(array)

    return array


def prepare_single_image(uploaded_file):

    image = Image.open(uploaded_file).convert("L")

    image = image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    array = normalize_array(
        np.array(image, dtype=np.float32)
    )

    five_slices = np.stack(
        [array] * 5,
        axis=0
    )

    tensor = torch.tensor(
        five_slices,
        dtype=torch.float32
    ).unsqueeze(0)

    return (
        tensor,
        five_slices,
        None,
        "Single 2D image"
    )


def prepare_multiple_images(uploaded_files):

    arrays = []

    for uploaded_file in uploaded_files:

        image = Image.open(
            uploaded_file
        ).convert("L")

        image = image.resize(
            (224, 224),
            Image.Resampling.BILINEAR
        )

        array = normalize_array(
            np.array(
                image,
                dtype=np.float32
            )
        )

        arrays.append(array)

    if len(arrays) != 5:
        raise ValueError(
            "Upload exactly 5 MRI slice images."
        )

    five_slices = np.stack(
        arrays,
        axis=0
    )

    tensor = torch.tensor(
        five_slices,
        dtype=torch.float32
    ).unsqueeze(0)

    return (
        tensor,
        five_slices,
        None,
        "Five 2D MRI slices"
    )


# ============================================================
# DICOM
# ============================================================

def read_dicom_files(file_paths):

    slices = []

    for path in file_paths:

        try:

            ds = pydicom.dcmread(
                str(path),
                force=True
            )

            if not hasattr(ds, "PixelData"):
                continue

            arr = ds.pixel_array.astype(
                np.float32
            )

            if arr.ndim != 2:
                continue

            instance = getattr(
                ds,
                "InstanceNumber",
                0
            )

            try:
                instance = int(instance)
            except Exception:
                instance = 0

            slices.append(
                {
                    "ds": ds,
                    "array": arr,
                    "instance": instance
                }
            )

        except Exception:
            continue

    slices.sort(
        key=lambda x: x["instance"]
    )

    return slices


def prepare_dicom_series(dicom_paths):

    slices = read_dicom_files(
        dicom_paths
    )

    if len(slices) < 5:
        raise ValueError(
            f"Only {len(slices)} valid DICOM slices "
            "were found. At least 5 are required."
        )

    volume = np.stack(
        [x["array"] for x in slices]
    )

    volume = normalize_array(volume)

    n = volume.shape[0]
    center = n // 2

    indices = [
        min(
            max(center + offset, 0),
            n - 1
        )
        for offset in [-2, -1, 0, 1, 2]
    ]

    selected = volume[indices]

    resized = []

    for array in selected:

        image = Image.fromarray(
            (array * 255).astype(np.uint8)
        )

        image = image.resize(
            (224, 224),
            Image.Resampling.BILINEAR
        )

        resized.append(
            np.array(
                image,
                dtype=np.float32
            ) / 255.0
        )

    five_slices = np.stack(
        resized
    )

    tensor = torch.tensor(
        five_slices,
        dtype=torch.float32
    ).unsqueeze(0)

    metadata = slices[
        n // 2
    ]["ds"]

    return (
        tensor,
        five_slices,
        metadata,
        f"DICOM series ({n} slices)"
    )


def extract_zip(uploaded_file):

    temp_dir = tempfile.TemporaryDirectory()

    zip_path = (
        Path(temp_dir.name)
        / "uploaded_mri.zip"
    )

    with open(zip_path, "wb") as f:
        f.write(
            uploaded_file.getbuffer()
        )

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as z:
        z.extractall(
            temp_dir.name
        )

    paths = []

    for root, _, files in os.walk(
        temp_dir.name
    ):

        for filename in files:

            if filename.startswith("."):
                continue

            paths.append(
                Path(root) / filename
            )

    return temp_dir, paths


def prepare_single_dicom(uploaded_file):

    temp_dir = tempfile.TemporaryDirectory()

    path = (
        Path(temp_dir.name)
        / uploaded_file.name
    )

    with open(path, "wb") as f:
        f.write(
            uploaded_file.getbuffer()
        )

    slices = read_dicom_files([path])

    if len(slices) != 1:
        raise ValueError(
            "The uploaded DICOM file could not be read as one MRI slice."
        )

    array = normalize_array(
        slices[0]["array"]
    )

    image = Image.fromarray(
        (array * 255).astype(np.uint8)
    )

    image = image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    array = (
        np.array(
            image,
            dtype=np.float32
        ) / 255.0
    )

    five_slices = np.stack(
        [array] * 5,
        axis=0
    )

    tensor = torch.tensor(
        five_slices,
        dtype=torch.float32
    ).unsqueeze(0)

    return (
        tensor,
        five_slices,
        slices[0]["ds"],
        "Single DICOM slice"
    )


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    input_tensor,
    predicted_class
):

    cam = GradCAM(
        model=model,
        target_layers=[
            model.layer4[-1]
        ]
    )

    return cam(
        input_tensor=input_tensor,
        targets=[
            ClassifierOutputTarget(
                predicted_class
            )
        ]
    )[0]



# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🧠 AlzheimerAI</h1>
        <p>
            Brain MRI analysis with deep learning and
            explainable AI
        </p>
        <div style="margin-top:18px;">
            <span style="
                background:rgba(255,255,255,0.18);
                padding:7px 14px;
                border-radius:999px;
                margin-right:8px;
                font-size:0.82rem;
                font-weight:700;
            ">MRI Analysis</span>
            <span style="
                background:rgba(255,255,255,0.18);
                padding:7px 14px;
                border-radius:999px;
                margin-right:8px;
                font-size:0.82rem;
                font-weight:700;
            ">Deep Learning</span>
            <span style="
                background:rgba(255,255,255,0.18);
                padding:7px 14px;
                border-radius:999px;
                font-size:0.82rem;
                font-weight:700;
            ">Explainable AI</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="disclaimer">
    <b>Research prototype:</b> This application is intended for
    academic and educational use only. Its predictions are not
    medical diagnoses and must not be used for clinical decisions.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MRI IMAGE DEMONSTRATION
# ============================================================

st.markdown(
    '<div class="section-title">MRI Image Demonstration</div>',
    unsafe_allow_html=True
)

st.info(
    "Upload an MRI image, DICOM file, or supported ZIP to demonstrate "
    "the image-analysis pipeline. This is a supporting ResNet18 "
    "demonstration with Grad-CAM; it is separate from the two final "
    "research models and does not determine the reported 87.88% ADNI "
    "or 96.34% OASIS-3 results."
)

# UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">1. Upload Brain MRI</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a brain MRI image or MRI series for analysis."
)

uploaded_files = st.file_uploader(
    "Choose MRI file(s)",
    type=[
        "png",
        "jpg",
        "jpeg",
        "dcm",
        "zip"
    ],
    accept_multiple_files=True,
    help=(
        "Supported: PNG/JPG, DICOM, DICOM ZIP, "
        "or exactly five MRI slice images."
    )
)


if uploaded_files:

    try:

        model = load_model()

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        if len(uploaded_files) == 1:

            uploaded = uploaded_files[0]

            extension = (
                Path(
                    uploaded.name
                ).suffix.lower()
            )

            if extension == ".zip":

                temp_dir, dicom_paths = (
                    extract_zip(
                        uploaded
                    )
                )

                (
                    input_tensor,
                    five_slices,
                    metadata,
                    mode
                ) = prepare_dicom_series(
                    dicom_paths
                )

            elif extension == ".dcm":

                temp_dir = None

                (
                    input_tensor,
                    five_slices,
                    metadata,
                    mode
                ) = prepare_single_dicom(
                    uploaded
                )

            else:

                temp_dir = None

                (
                    input_tensor,
                    five_slices,
                    metadata,
                    mode
                ) = prepare_single_image(
                    uploaded
                )

        else:

            temp_dir = None

            if not all(
                Path(f.name).suffix.lower()
                in [".png", ".jpg", ".jpeg"]
                for f in uploaded_files
            ):
                raise ValueError(
                    "Multiple upload mode supports PNG/JPG/JPEG only."
                )

            if len(uploaded_files) != 5:
                raise ValueError(
                    "Upload exactly 5 MRI slice images."
                )

            (
                input_tensor,
                five_slices,
                metadata,
                mode
            ) = prepare_multiple_images(
                uploaded_files
            )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        input_tensor = input_tensor.to(
            DEVICE
        )

        with torch.no_grad():

            outputs = model(
                input_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[0]

        predicted_class = torch.argmax(
            probabilities
        ).item()

        cn_probability = probabilities[0].item()
        ad_probability = probabilities[1].item()

        prediction = (
            "AD"
            if predicted_class == 1
            else "CN"
        )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">2. Prediction</div>',
            unsafe_allow_html=True
        )

        result_color = (
            "#fff7ed"
            if prediction == "AD"
            else "#eff6ff"
        )

        result_text = (
            "Model output: AD"
            if prediction == "AD"
            else "Model output: CN"
        )

        # Clear, user-friendly Alzheimer's result
        if prediction == "AD":
            result_title = "Alzheimer's Disease pattern predicted"
            result_icon = "🔴"
            result_subtitle = "The research model classified this MRI as AD."
            result_color = "#fff1f2"
            result_border = "#fda4af"
            result_text_color = "#be123c"
        else:
            result_title = "No Alzheimer's Disease pattern predicted"
            result_icon = "🟢"
            result_subtitle = (
                "The research model classified this MRI as CN "
                "(Cognitively Normal)."
            )
            result_color = "#ecfdf5"
            result_border = "#86efac"
            result_text_color = "#047857"

        st.markdown(
            f"""
            <div class="result-card"
                 style="background:linear-gradient(135deg,{result_color},#ffffff);border:2px solid {result_border};">
                <div style="font-size:2.2rem;margin-bottom:0.3rem;">{result_icon}</div>
                <div class="result-label" style="color:{result_text_color};">
                    Alzheimer's Disease Prediction
                </div>
                <div class="result-value" style="color:{result_text_color};font-size:2rem;">
                    {result_title}
                </div>
                <div style="color:#475569;margin-top:0.45rem;font-size:0.95rem;">
                    {result_subtitle}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">
                        Predicted Class
                    </div>
                    <div class="metric-value">
                        {prediction}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">
                        AD Probability
                    </div>
                    <div class="metric-value">
                        {ad_probability * 100:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">
                        CN Probability
                    </div>
                    <div class="metric-value">
                        {cn_probability * 100:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            f"""
            <div style="margin-top:0.8rem;padding:0.85rem 1rem;border-radius:12px;background:#f8fafc;border:1px solid #e2e8f0;color:#475569;font-size:0.9rem;">
                <b>How to read this result:</b> The model selects the class with the higher probability.
                Current result: <b>{prediction}</b> ({max(cn_probability, ad_probability) * 100:.2f}% model probability).
                This is a research prediction, not a confirmed medical diagnosis.
            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # INPUT INFO
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">3. MRI Information</div>',
            unsafe_allow_html=True
        )

        info = {
            "Processing mode": mode,
            "Uploaded file(s)": len(uploaded_files)
        }

        if metadata is not None:

            info["Patient Sex"] = str(
                getattr(
                    metadata,
                    "PatientSex",
                    "Not available"
                )
            )

            info["Patient Age"] = str(
                getattr(
                    metadata,
                    "PatientAge",
                    "Not available"
                )
            )

        st.dataframe(
            pd.DataFrame(
                {
                    "Property": list(info.keys()),
                    "Value": list(info.values())
                }
            ),
            hide_index=True,
            use_container_width=True
        )


        # ----------------------------------------------------
        # GRAD-CAM
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">4. Explainable AI — Grad-CAM</div>',
            unsafe_allow_html=True
        )

        grayscale_cam = generate_gradcam(
            model,
            input_tensor,
            predicted_class
        )

        middle_slice = five_slices[2]

        rgb_slice = np.stack(
            [
                middle_slice,
                middle_slice,
                middle_slice
            ],
            axis=-1
        )

        visualization = show_cam_on_image(
            rgb_slice,
            grayscale_cam,
            use_rgb=True
        )

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                middle_slice,
                caption="Processed MRI",
                use_container_width=True,
                clamp=True
            )

        with col2:

            st.image(
                visualization,
                caption="Grad-CAM explanation",
                use_container_width=True
            )

        st.info(
            "Grad-CAM identifies image regions that influenced "
            "the model prediction. Highlighted regions are not "
            "clinically validated Alzheimer's biomarkers."
        )


        # ----------------------------------------------------
        # MODEL DETAILS
        # ----------------------------------------------------

        with st.expander(
            "View model details"
        ):

            details = pd.DataFrame(
                {
                    "Component": [
                        "Primary OASIS-3 model",
                        "Primary ADNI model",
                        "OASIS-3 accuracy",
                        "ADNI accuracy",
                        "ADNI ROC-AUC",
                        "Image demonstration model",
                        "Explainability"
                    ],
                    "Value": [
                        "Weighted LR + RF + XGBoost ensemble",
                        "Optimized Random Forest",
                        "96.34%",
                        "87.88%",
                        "0.9269",
                        "ResNet18",
                        "Grad-CAM + SHAP"
                    ]
                }
            )

            st.dataframe(
                details,
                hide_index=True,
                use_container_width=True
            )

        if temp_dir is not None:
            temp_dir.cleanup()


    except Exception as e:

        st.error(
            "The uploaded MRI could not be processed."
        )

        st.exception(e)


# ============================================================
# RESEARCH OVERVIEW
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">Final Research Models</div>',
    unsafe_allow_html=True
)

st.write(
    "The final research framework contains two validated model experiments. "
    "The MRI image section above is retained only as a supporting demonstration."
)

tab1, tab2 = st.tabs(
    [
        "OASIS-3 Multimodal Ensemble",
        "ADNI Random Forest"
    ]
)

with tab1:
    st.markdown(
        """
        **Primary multimodal experiment**

        **Input:** MRI/FreeSurfer-derived features + cognitive features

        **Method:** Mutual-information feature selection followed by a
        weighted ensemble of Logistic Regression, Random Forest and XGBoost.

        **Held-out accuracy: 96.34%**

        **Precision: 77.78%**

        **Recall/Sensitivity: 63.64%**

        **Specificity: 98.69%**

        **F1-score: 70.00%**

        **ROC-AUC: 0.8990**

        **AUPRC: 0.5555**
        """
    )

with tab2:
    st.markdown(
        """
        **Independent ADNI validation experiment**

        **Input:** 37 FreeSurfer-derived structural MRI features

        **Method:** Optimized Random Forest for CN versus AD classification.

        **Held-out accuracy: 87.88%**

        **Precision: 85.19%**

        **Recall/Sensitivity: 85.19%**

        **Specificity: 89.74%**

        **F1-score: 85.19%**

        **ROC-AUC: 0.9269**

        **AUPRC: 0.9178**
        """
    )

st.markdown(
    '<p class="small-note">'
    'The 96.34% OASIS-3 and 87.88% ADNI results are the two final '
    'research models. The raw-MRI ResNet18 image pipeline is a separate '
    'supporting demonstration and is not used to report either final '
    'research accuracy.'
    '</p>',
    unsafe_allow_html=True
)

st.divider()



