# ISLES 2016
login to [ISLES 2016](https://www.smir.ch/ISLES/Start2016)
download training coregistered, training native etc

# CT intracranial hemorrhage
```bash
mkdir data/CTblood
cd data/CTblood
kaggle competitions download -c rsna-intracranial-hemorrhage-detection
unzip rsna-intracranial-hemorrhage-detection.zip
```

# OASIS3
Go to Oasis, then Browse > Data > MR Sessions 
Options > Edit columns (remove 'Date' and 'Project' columns)
Options > Spreadsheet
Save the downloaded csv to `data/OASIS3/labels/MR_session.csv`
Change the 'label' in the colnames into 'experiment_id'
Remove all the fields that are not OASIS3

Go to Oasis, then Browse > Data > ADRC Clinical Data > Options > Spreadsheet
Save the downloaded csv to `data/OASIS3/labels/clinical_data.csv`


```bash
Rscript data_raw/oasis_data_matchup.R data/OASIS3/labels/MR_session.csv \
        data/OASIS3/labels/clinical_data.csv 1095 0 \
        data/OASIS3/labels/all.csv

./data_raw/download_oasis_scans.sh data/OASIS3/labels/MR_ID.csv \
        data/OASIS3 YutongChen T1w

source CTPM/bin/activate
python data_raw/OASIS3.py
deactivate
```

# CTP
```bash
source CTPM/bin/activate
python data_raw/CTP.py --dir='data/CTP/CTP_raw' \
        --mode='CTP,CTA,NCCT' 
python data_raw/CT_post.py --dir='data/CTP'
deactivate
```
