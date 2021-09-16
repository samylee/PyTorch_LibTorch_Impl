from __future__ import print_function, division

import torch
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image
import os

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load model
model = torch.load('model.pkl')
model = model.to(device)
model.eval()

classes = ['cat','dog']

test_path = "data/val/"
true_count = 0
all_count = 0

for test_dir in os.listdir(test_path):
    test_dir_path = test_path + test_dir + "/"
    for img_names in os.walk(test_dir_path):
        for img_name in img_names[2]:
            img_path = test_dir_path + img_name
            print(img_path)
    
            image = Image.open(img_path)
            transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            image_transformed = transform(image)
            image_transformed = image_transformed.unsqueeze(0)
            
            # forward
            output = model(image_transformed.to(device))
            
            output = F.softmax(output, dim=1)
            predict_value, predict_idx = torch.max(output, 1)
            
            if(classes[predict_idx.cpu().data[0].numpy()] == test_dir):
                true_count += 1
            
            all_count += 1
        
print("acc: {}/{}={}".format(true_count,all_count,float(true_count)/float(all_count)))
#acc: 1966/2000=0.983
