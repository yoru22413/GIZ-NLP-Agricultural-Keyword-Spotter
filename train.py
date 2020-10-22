import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils import seed, DatasetPreparer, Data, Model

seed()

dp = DatasetPreparer('Train.csv', 'audio_files')
train, test = dp.get_train_test()

X_train, X_val = train_test_split(train, test_size=0.2, stratify=train['label'], shuffle=True,
                                  random_state=4738)

batch_size = 32

train_loader = DataLoader(Data(X_train, data_augmentation=True),
                          batch_size=batch_size)
val_loader = DataLoader(Data(X_val, data_augmentation=False), batch_size=batch_size)

lr_monitor = LearningRateMonitor(logging_interval='epoch')
mc = pl.callbacks.ModelCheckpoint(filepath='{epoch}-{CE_val:.5f}',
                                  save_top_k=3,
                                  save_weights_only=True)

model = Model()
trainer = pl.Trainer(gpus=None, precision=32, checkpoint_callback=mc, callbacks=[lr_monitor],
                     progress_bar_refresh_rate=5, max_epochs=120)
trainer.fit(model, train_loader, val_loader)
