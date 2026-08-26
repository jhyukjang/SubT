import torch
from torch.utils.data import Dataset
import pandas as pd
import os



class FewShotDataset(Dataset):
    def __init__(self, root, split=None, num_shots=-1, repeat=False, process_audio_fn=None, resample=True, is_half_labels=False, train_csv=None, test_csv=None, args=None): 
        """
        Args:
            root (str): path to the dataset.
            num_shots (int): number of shots per class.
            repeat (bool): repeat samples if needed (default: False).
            process_audio_fn (function): function to process audio samples.
            resample (bool): resample audio samples (default: True).
            is_half_labels (bool): use half labels (default: False).
            train_csv (str): path to the train csv file.
            test_csv (str): path to the test csv file.
        """

        assert split is not None, "'split' cannot be None. Choose from ['train', 'test']"
        assert train_csv is not None, "'train_csv' cannot be None"
        assert test_csv is not None, "'test_csv' cannot be None"
        
        self.root = root
        self.split = split
        self.num_shots = num_shots
        self.repeat = repeat
        self.resample = resample
        self.is_half_labels = is_half_labels
        if split == 'train':
            df = pd.read_csv(os.path.join(root, train_csv))
        else:
            df = pd.read_csv(os.path.join(root, test_csv))
        
        self.classnames = df['classname'].unique().tolist()
        self.classnames.sort()
        if self.is_half_labels:
            self.classnames = self.classnames[:(len(self.classnames)+1)//2]
        self.label2classname = {i: classname for i, classname in enumerate(self.classnames)}
        self.classname2label = {classname: i for i, classname in enumerate(self.classnames)}
        
        self.data = self.generate_fewshot_dataset(df, num_shots=num_shots, repeat=repeat)

        self.process_audio_fn = process_audio_fn

        print("\n\n################## Dataset Information ##################")
        if num_shots>0: print("FewShot Dataset")
        print(f"{'Root':<25} : {root}")
        print(f"{'Split':<25} : {split}")
        print(f"{'Number of Classes':<25} : {len(self.classnames)}")
        print(f"{'Number of Shots':<25} : {num_shots}")
        print(f"{'Total Number of Samples':<25} : {len(self.data)}")
        print(f"{'Classnames':<25} : {self.classnames}")
        print(f"{'Label to Classname':<25} : {self.label2classname}")
        print(f"{'Classname to Label':<25} : {self.classname2label}")
        print("########################################################\n\n")

    def generate_fewshot_dataset(self, df, num_shots=-1, repeat=False):
        """
        Generate a few-shot dataset.
        Args:
            df (pd.DataFrame): dataframe containing the dataset.
            num_shots (int): number of shots per class.
            repeat (bool): repeat samples if needed.
        """

        if num_shots == -1:
            return df

        print(f"Creating a {num_shots}-shot dataset ...")
        df_subset = pd.DataFrame(columns=df.columns)

        for classname in self.classnames:

            df_class = df[df['classname'] == classname]

            if len(df_class) >= num_shots:
                df_subset = pd.concat([df_subset, df_class.sample(num_shots)])
            else:
                if repeat:
                    df_subset = pd.concat([df_subset, df_class.sample(num_shots, replace=True)])
                else:
                    df_subset = pd.concat([df_subset,df_class])


        df_subset = df_subset.reset_index(drop=True)

        return df_subset


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = os.path.join(self.root, row['path'])
        audio = self.process_audio_fn([audio_path], self.resample)
        label = self.classname2label[row['classname']]
        return audio, label
    

    

