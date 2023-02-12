### 根据训练好的模型去识别

import os
import torch
from PIL import Image
from torchvision import transforms
from model import Model
import warnings
import pandas as pd

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path_to_checkpoint_file = r'./logs/model-66000.pth'
    file_name = []
    house_number = []
    res = pd.DataFrame()
    num_to_predict = len(os.listdir(r'./data/test')) - 2
    model = Model()
    model.restore(path_to_checkpoint_file)
    model.to(device)

    for i in range(num_to_predict):
        print('******************************{}/{}******************************'.format(i + 1, num_to_predict))
        img_name = ('0000' + str(i + 1) + '.png')[-9:]
        img_path = os.path.join(r'./data/test', str(i + 1) + '.png')
        with torch.no_grad():
            transform = transforms.Compose([
                transforms.Resize([64, 64]),
                transforms.CenterCrop([54, 54]),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            image = Image.open(img_path)
            image = image.convert('RGB')
            image = transform(image)
            images = image.unsqueeze(dim=0).to(device)
            length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits = model.eval()(images)
            length_prediction = length_logits.max(1)[1]
            digit1_prediction = digit1_logits.max(1)[1]
            digit2_prediction = digit2_logits.max(1)[1]
            digit3_prediction = digit3_logits.max(1)[1]
            digit4_prediction = digit4_logits.max(1)[1]
            digit5_prediction = digit5_logits.max(1)[1]
            digit_list=[digit1_prediction,digit2_prediction,digit3_prediction,digit4_prediction,digit5_prediction]
        file_name.append(img_name)
        house_number.append(''.join([str(x.item()) for x in digit_list[:length_prediction]]))
    res['file_name'] = file_name
    res['house_number'] = house_number
    res.sort_values(by='file_name')
    res.to_csv(r'./res.csv', index=False)