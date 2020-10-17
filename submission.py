import sys

import torch
from torch.utils.data import DataLoader

from utils import Model, DatasetPreparer, Data, seed

seed()
model_path = sys.argv[1]
model = Model().cuda()
model.load_state_dict(torch.load(model_path)['state_dict'])
dp = DatasetPreparer('Train.csv', 'audio_files')
train, test = dp.get_train_test()
test_loader = DataLoader(Data(test, data_augmentation=False, test=True), batch_size=32)
predictions = model.predict(test_loader)
submission = dp.submission(predictions, test)
submission.to_csv('submission.csv', index=False)
