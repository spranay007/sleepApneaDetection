import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import wfdb
import matplotlib.pyplot as plt
import os


# for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

from subprocess import check_output
files = os.listdir('./')
print (files)
#print(check_output(["ls", "E:\College\Project\apnea-ecg-database"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
os.listdir("./apnea-ecg-database")


#record = wfdb.rdrecord('./apnea-ecg-database/setOne/a01')
record = wfdb.rdrecord(record_name='./apnea-ecg-database/setOne/a01er', sampfrom=1200, sampto=4200,channels=None, physical=True, pn_dir=None, m2s=True, smooth_frames=True, ignore_skew=False, return_res=16, force_channels=True, channel_names=None, warn_empty=False)

wfdb.plot_wfdb(record, title='Record a01 from Physionet Kaggle Apnea ECG') 
#display(record.__dict__)


#record2 = wfdb.rdrecord('./apnea-ecg-database/a05') 
#wfdb.plot_wfdb(record, title='Record a05 from Physionet Kaggle Apnea ECG') 
#display(record2.__dict__)

'''
recordname = "./apnea-ecg-database/a01"
record3 = wfdb.rdsamp(recordname)
annotation = wfdb.rdann(recordname, extension="apn")

annotation.contained_labels
annotation.get_label_fields()
annotation.symbol[:10]
np.unique(annotation.symbol, return_counts=True)
'''

recordname = "./apnea-ecg-database/setOne/a01er"
record3 = wfdb.rdsamp(recordname)
annotation = wfdb.rdann(recordname, extension="apn")

annotation.contained_labels
annotation.get_label_fields()
annotation.symbol[:10]
np.unique(annotation.symbol, return_counts=True)