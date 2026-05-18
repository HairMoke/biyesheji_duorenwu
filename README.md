


## Installation

```txt
python >= 3.6
torch >= 1.7.0
numpy >= 1.20
tqdm >= 4.59.0
scipy >= 1.6.2
```

## Usage

1. modified the config file

   ```json
    {
        "DatasetName": ["CAS", "THU", "GIST"],
        "Dataset": {
            
            "CAS": {
                "subject_num": 14
            },
            "THU": {
                "subject_num": 64
            },
            "GIST": {
                "subject_num": 55
            }
        },
        "TrainPara":{
            "batch_size": 32,
            "epoch": 60
        },
        "TestPara":{
            "batch_size": 32,
            "epoch": 1
        }
    }
   
   ```
2. Train model

   ```cmd
   python main.py
   ```



## Citation



数据集地址：[EEG-RSVP 数据集（THU，CAS，DPN） --- EEG-RSVP dataset(THU,CAS,DPN)](https://www.kaggle.com/datasets/hairmonk/eeg-rsvp-datasetthucasdpn)

