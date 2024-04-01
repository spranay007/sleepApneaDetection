import os
import wfdb
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from subprocess import check_output

files = os.listdir('./')
#print (files)
#print(check_output(["ls", "E:\College\Project\apnea-ecg-database"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
os.listdir("./apnea-ecg-database")

#record = wfdb.rdrecord('./apnea-ecg-database/setOne/a01')
record = wfdb.rdrecord(record_name='./apnea-ecg-database/setOne/a01er', sampfrom=1200, sampto=19400,channels=None, physical=True, pn_dir=None, m2s=True, smooth_frames=True, ignore_skew=False, return_res=16, force_channels=True, channel_names=None, warn_empty=False)
wfdb.plot_wfdb(record, title='Record a01 from Physionet Kaggle Apnea ECG') 
#display(record.__dict__)


recordname = "./apnea-ecg-database/setOne/a01r"
record1 = wfdb.rdsamp(recordname)

annotation = wfdb.rdann(recordname,
                        extension="apn",
                        sampfrom=1200,
                        sampto=None,
                        shift_samps=False,
                        pn_dir=None,
                        return_label_elements=["symbol"],
                        summarize_labels=False)

print("Contained Labels:", annotation.contained_labels)
print("Label Fields:", annotation.get_label_fields())
print("First 50 Symbols:", annotation.symbol[12:100])
print("Unique Symbols and Counts:", np.unique(annotation.symbol, return_counts=True))

