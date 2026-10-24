# Alzheimer's MRI Dashboard

## Files

Place `app.py` inside:

`Alzheimer_Multimodal_AI/dashboard/`

The trained model should already exist at:

`Alzheimer_Multimodal_AI/models/adni_resnet18_5slice_baseline.pth`

## Run

From the project root:

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

Upload an ADNI-style DICOM ZIP. The app extracts the series, selects five central slices, predicts CN/AD, and displays Grad-CAM.

This is a research/educational prototype and is not a clinical diagnostic system.
