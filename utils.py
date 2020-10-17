import os
import random

import librosa
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations import (
    Compose,
    Normalize,
    OneOf)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from model import ModelForward


class config:
    sampling_rate = 22050
    n_fft = int(0.02 * sampling_rate)
    n_mels = 80
    hop_length = int(n_fft * 0.5)


def audio_to_spectrogram(audio):
    spectrogram = librosa.stft(audio,
                               sr=config.sampling_rate,
                               hop_length=config.hop_length,
                               n_fft=config.n_fft)
    spectrogram = librosa.amplitude_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


def seed(seed=2020):
    """Reproducer for pytorch experiment.

    Parameters
    ----------
    seed: int, optional (default = 2020)
        Radnom seed.

    Example
    -------
    seed_reproducer(seed=2020).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True


seed()


class PadToSize:
    def __init__(self, mode='constant'):
        assert mode in ['constant', 'wrap']
        self.size = 277
        self.mode = mode

    def __call__(self, signal):
        if signal.shape[1] < self.size:
            padding = self.size - signal.shape[1]
            offset = padding // 2
            pad_width = ((0, 0), (offset, padding - offset))
            if self.mode == 'constant':
                signal = np.pad(signal, pad_width,
                                'constant', constant_values=signal.min())
            else:
                signal = np.pad(signal, pad_width, 'wrap')
        return signal


data_augmentation = Compose(
    [
        OneOf([
            PadToSize(mode='wrap'),
            PadToSize(mode='constant'),
        ], p=[0.5, 0.5]),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
    ]
)
data_augmentation_test = Compose(
    [
        OneOf([
            PadToSize(mode='wrap'),
            PadToSize(mode='constant'),
        ], p=[0.5, 0.5]),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
    ]
)


class DatasetPreparer:
    def __init__(self, train_path, audio_path):
        self.train_path = train_path
        self.audio_path = audio_path
        self.label_encoder = LabelEncoder()

    def get_train_test(self):
        train = pd.read_csv(self.train_path)
        list_audio = [self.audio_path + '/' + x for x in os.listdir(self.audio_path)]
        test = [fn for fn in list_audio if fn not in train['fn'].tolist()]
        train['label'] = self.label_encoder.fit_transform(train['label'])
        return train, pd.Series(test, name='fn')

    def submission(self, prediction, index):
        data = {'fn': index}
        data = pd.DataFrame(data=data)
        prediction = pd.DataFrame(data=prediction, columns=np.arange(prediction.shape[1]))
        prediction.columns = self.label_encoder.inverse_transform(prediction.columns)
        submission = pd.concat([data, prediction], axis=1)
        return submission


class Data(Dataset):
    def __init__(self, df, data_augmentation=False, test=False):
        self.df = df
        self.data_augmentation = data_augmentation
        self.test = test

    def __len__(self):
        return len(self.df)

    def prepare_data(self, path):
        data = librosa.load(path)
        data = audio_to_spectrogram(data)
        data = data_augmentation(image=data)['image'] if self.data_augmentation else data_augmentation_test(image=data)[
            'image']
        data = np.expand_dims(data, axis=0)
        return data

    def __getitem__(self, idx):
        if not self.test:
            fn, label, quality = self.df.iloc[idx]
        else:
            fn = self.df.iloc[idx]

        data = self.prepare_data(fn)
        if self.test:
            return data
        else:
            return data, np.array([label], dtype=np.int64)


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model_forward = ModelForward()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model_forward(x)

    def predict(self, data_loader):
        outputs = []
        self.eval()
        with torch.no_grad():
            for x in data_loader:
                x = x.cuda()
                out = self(x)
                outputs.append(out)
        return torch.cat(outputs, dim=0).cpu().detach().numpy()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        result = pl.TrainResult(loss)
        result.log('E_train', loss, prog_bar=True, on_epoch=True, on_step=False, logger=True)
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        result = pl.EvalResult()
        result.log('E_val', loss, prog_bar=False, on_epoch=True, on_step=False, logger=True)
        return result

    def validation_epoch_end(self, outputs):
        metric = outputs['E_val'].mean()
        result = pl.EvalResult(checkpoint_on=metric)
        result.log('CE_val', metric, prog_bar=True, logger=True)
        return result

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        self.scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=8),
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'val_loss'
        }
        return [self.optimizer], [self.scheduler]
