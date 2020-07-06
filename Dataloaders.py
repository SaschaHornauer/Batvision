import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import random

class raw_to_depth(Dataset):
    """waveform to depth maps."""

    def __init__(self, csv_file, output=128):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            
        """
        self.df = pd.read_csv(csv_file, sep="\t")
        self.output = (output,output)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        start_id = random.randint(0,970)
        left_raw_path = self.df["raw_l"].loc[idx]
        left_raw = np.load(left_raw_path)[start_id:start_id+3200]
        left_raw = np.expand_dims(np.expand_dims(left_raw,axis=0),axis=0)

        right_raw_path = self.df["raw_r"].loc[idx]
        right_raw = np.load(right_raw_path)[start_id:start_id+3200]
        right_raw = np.expand_dims(np.expand_dims(right_raw,axis=0),axis=0)

        if left_raw.shape != right_raw.shape:
            print(left_raw.shape, right_raw.shape, left_raw_path,right_raw_path )

        depth_meas_path = self.df["depth"].loc[idx]
        depth_meas = np.load(depth_meas_path)

        depth_meas = cv2.resize(depth_meas,self.output)
        depth_meas[depth_meas > 12e3] = 12e3
        depth_meas = depth_meas/12e3
        depth = np.expand_dims(depth_meas,axis=0)
                
        return torch.from_numpy(left_raw).float(),torch.from_numpy(right_raw).float(), torch.from_numpy(depth).float()

class raw_spec_to_depth(Dataset):
    """waveform to depth maps."""

    def __init__(self, csv_file, output=128):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            
        """
        self.df = pd.read_csv(csv_file, sep="\t")
        self.output = (output,output)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        start_id = random.randint(0,970)
        left_raw_path = self.df["raw_l"].loc[idx]
        left_raw = np.load(left_raw_path)[start_id:start_id+3200]
        left_raw = np.expand_dims(np.expand_dims(left_raw,axis=0),axis=0)

        right_raw_path = self.df["raw_r"].loc[idx]
        right_raw = np.load(right_raw_path)[start_id:start_id+3200]
        right_raw = np.expand_dims(np.expand_dims(right_raw,axis=0),axis=0)

        if left_raw.shape != right_raw.shape:
            print(left_raw.shape, right_raw.shape, left_raw_path,right_raw_path )

        left_spec_path = self.df["spec_l"].loc[idx]
        left_spec = Image.open(left_spec_path).crop((54, 35, 388, 251))
        left_spec = left_spec.convert('RGB')#.resize((256,256), Image.ANTIALIAS)
        left_spec = (np.array(left_spec)*1/255).transpose((2,0,1))
        
        
        right_spec_path = self.df["spec_r"].loc[idx]
        right_spec = Image.open(right_spec_path).crop((54, 35, 388, 251))
        right_spec = right_spec.convert('RGB')#.resize((256,256), Image.ANTIALIAS)
        right_spec = (np.array(right_spec)*1/255).transpose((2,0,1))

        depth_meas_path = self.df["depth"].loc[idx]
        depth_meas = np.load(depth_meas_path)

        depth_meas = cv2.resize(depth_meas,self.output)
        depth_meas[depth_meas > 12e3] = 12e3
        depth_meas = depth_meas/12e3
        depth = np.expand_dims(depth_meas,axis=0)
                
        return torch.from_numpy(left_spec).float(),torch.from_numpy(right_spec).float(),torch.from_numpy(left_raw).float(),torch.from_numpy(right_raw).float(), torch.from_numpy(depth).float()


class raw_to_image(Dataset):
    """waveform to grayscale images."""

    def __init__(self, csv_file, output=128):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            
        """
        self.df = pd.read_csv(csv_file, sep="\t")
        self.output = (output,output)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        start_id = random.randint(0,970)

        left_raw_path = self.df["raw_l"].loc[idx]
        left_raw = np.load(left_raw_path)[start_id:start_id+3200]
        left_raw = np.expand_dims(np.expand_dims(left_raw,axis=0),axis=0)

        right_raw_path = self.df["raw_r"].loc[idx]
        right_raw = np.load(right_raw_path)[start_id:start_id+3200]
        right_raw = np.expand_dims(np.expand_dims(right_raw,axis=0),axis=0)

        image_path = self.df["image"].loc[idx]
        image = Image.open(image_path).resize(self.output, Image.ANTIALIAS)
        image = image.convert('L')
        image = np.expand_dims((np.array(image)*1/255),axis=-1).transpose((2,0,1))
        
        return torch.from_numpy(left_raw).float(),torch.from_numpy(right_raw).float(), torch.from_numpy(image).float()

class spec_to_depth(Dataset):
    """Spectrogram to depth maps."""

    def __init__(self, csv_file, output=128):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            
        """
        self.df = pd.read_csv(csv_file, sep="\t")
        self.output = (output,output)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        left_spec_path = self.df["spec_l"].loc[idx]
        left_spec = Image.open(left_spec_path).crop((54, 35, 388, 251))
        left_spec = left_spec.convert('RGB')#.resize((256,256), Image.ANTIALIAS)
        left_spec = (np.array(left_spec)*1/255).transpose((2,0,1))
        
        
        right_spec_path = self.df["spec_r"].loc[idx]
        right_spec = Image.open(right_spec_path).crop((54, 35, 388, 251))
        right_spec = right_spec.convert('RGB')#.resize((256,256), Image.ANTIALIAS)
        right_spec = (np.array(right_spec)*1/255).transpose((2,0,1))


        depth_meas_path = self.df["depth"].loc[idx]
        depth_meas = np.load(depth_meas_path)

        depth_meas = cv2.resize(depth_meas,self.output)
        depth_meas[depth_meas > 12e3] = 12e3
        depth_meas = depth_meas/12e3
        depth = np.expand_dims(depth_meas,axis=0)

        return torch.from_numpy(left_spec).float(),torch.from_numpy(right_spec).float(), torch.from_numpy(depth).float()

class spec_to_image(Dataset):
    """Spectrogram to grayscale images."""

    def __init__(self, csv_file, output=128):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            
        """
        self.df = pd.read_csv(csv_file, sep="\t")
        self.output = (output,output)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        left_spec_path = self.df["spec_l"].loc[idx]
        left_spec = Image.open(left_spec_path).crop((54, 35, 388, 251))
        left_spec = left_spec.convert('RGB')#.resize((256,256), Image.ANTIALIAS)
        left_spec = (np.array(left_spec)*1/255).transpose((2,0,1))
        
        
        right_spec_path = self.df["spec_r"].loc[idx]
        right_spec = Image.open(right_spec_path).crop((54, 35, 388, 251))
        right_spec = right_spec.convert('RGB')#.resize((256,256), Image.ANTIALIAS)
        right_spec = (np.array(right_spec)*1/255).transpose((2,0,1))


        image_path = self.df["image"].loc[idx]
        image = Image.open(image_path).resize(self.output, Image.ANTIALIAS)
        image = image.convert('L')
        image = np.expand_dims((np.array(image)*1/255),axis=-1).transpose((2,0,1))

        return torch.from_numpy(left_spec).float(),torch.from_numpy(right_spec).float(), torch.from_numpy(image).float()
