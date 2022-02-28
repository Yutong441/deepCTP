# Predict thrombolysis outcome using CT perfusion imaging

```bash
python3.8 -m venv CTPM
source CTPM/bin/activate
pip3 install numpy pandas matplotlib seaborn scikit-learn 
pip3 install nibabel pydicom
pip3 install torch torchvision
pip3 install medcam
deactivate
```

To use the script, enter the configurations such as the files in the `train` directory. Then,
```bash
source CTPM/bin/activate
# for example:
python train/CTP.py --mode='train'
python train/CTP.py --mode='test'
deactivate
```
