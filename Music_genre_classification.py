#!/usr/bin/env python
# coding: utf-8

# ### Lots of the code was taken from our CNN assignment (to utilize the summary)

# In[34]:


import librosa
import pathlib
import csv
import torch.utils.tensorboard as tb
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import random 
import os, math


#STUDENT's TODO
NOTEBOOK=1 #turn to zero before submitting

# Object labels used in this programming homework 
LABEL_NAMES = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock':9 }

LABEL_=['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']


# In[10]:


# get spectrograms

cmap = plt.get_cmap('magma')

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

for g in genres:
    pathlib.Path(f'img_data_graph/{g}').mkdir(parents=True, exist_ok=True)     
    for filename in os.listdir(f'./genres/{g}'):
        songname = f'./genres/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=5)
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.xlabel('Time')
        plt.ylabel('Hz')
        cl = plt.colorbar()
        cl.ax.set_title('dB')
        plt.savefig(f'img_data_graph/{g}/{filename[:-3].replace(".", "")}.png')
        plt.clf()



# In[28]:


# write the label csv for valid and train
header = ['file', 'label']
file = open('labels.csv', 'w', newline='')

with file:
    writer = csv.writer(file)
    writer.writerow(header)
    
for g in genres:
    for filename in os.listdir(f'img_data/valid/{g}'):
        row = f'{filename} {g}'
        file = open('labels.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(row.split())
            
for g in genres:
    for filename in os.listdir(f'img_data/train/{g}'):
        row = f'{filename} {g}'
        file = open('labels.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(row.split())
            
# then I moved every img_data out of the genre folder (prob could have done this programmatically)


# In[32]:


class MusicDataset(Dataset):
    def __init__(self, image_path,data_transforms=None):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv
        """
        #parse csv file and initialize data_transform
        self.image_path = image_path
        filename = 'labels.csv'
        csv = os.path.join(self.image_path, filename)
        data = pd.read_csv(csv)
        new_labels = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock':9 }
        data.label = [new_labels[item] for item in data.label]
        
        self.data = data
        
        self.transforms = data_transforms


    def __len__(self):
        """
        Your code here
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """

        image_name = os.path.join(self.image_path, self.data.iloc[idx, 0])
        image = Image.open(image_name)
        image_arr = np.asanyarray(image)
        image_tensor = torchvision.transforms.ToTensor()(image_arr)

        label = self.data.iloc[idx, 1]
        return (image_tensor, label)


# In[36]:


def visualize_data():
    # Student's TODO. Replace ' ' by your path to training data folder
    Path_to_your_data= 'img_data/train'
    
    dataset = MusicDataset(image_path=Path_to_your_data)

    f, axes = plt.subplots(3, len(LABEL_NAMES))

    counts = [0]*len(LABEL_NAMES)

    for img, label in dataset:
        c = counts[label]

        if c < 3:
            ax = axes[c][label]
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.axis('off')
            ax.set_title(LABEL_[label])
            counts[label] += 1
        
        if sum(counts) >= 3 * len(LABEL_NAMES):
            break

    plt.show()
    
if NOTEBOOK:
    visualize_data()


# In[ ]:


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Your code here
        Compute mean(-log(softmax(input)_label))
        @input:  torch.Tensor((B,C)), where B = batch size, C = number of classes
        @target: torch.Tensor((B,), dtype=torch.int64)
        @return:  torch.Tensor((,))
        Hint: use torch.nn.functional.nll_loss and torch.nn.functional.log_softmax
        More details: https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.nll_loss).
        """
        
        m = torch.nn.functional.log_softmax(input)
        nll_loss = torch.nn.functional.nll_loss(m, target)
        return nll_loss

def test_ClassificationLoss():
    # Cross entropy loss
    base_loss_obj    = torch.nn.CrossEntropyLoss(reduction='mean')
    # Your loss 
    student_loss_obj = ClassificationLoss()
    
    # Run 100 tests 
    for i in range(100):
      dummy_logit  = torch.randn([10,6])
      dummy_target = torch.randint(0, 6, [10])
      student_loss = student_loss_obj(dummy_logit, dummy_target)
      base_loss    = base_loss_obj(dummy_logit, dummy_target)

      # Check type
      if not torch.is_tensor(student_loss):
          print(f"[Fail!] ClassificationLoss.forward(...) must return a tensor!, but yours returns: {type(student_loss)}")
          return 
      # Check size
      if student_loss.size():
          print(f"[Fail!] ClassificationLoss.forward(...) must return a tensor of size torch.Size([])!, but your return's size: {student_loss.size()}")
          return
      # Check value
      if torch.abs(student_loss - base_loss) > 1e-6:
          print(f"[Fail!] ClassificationLoss.forward(...) returned value is not correct! It should be {base_loss}, but yours returns: {student_loss}")
          return
    # If you pass all, congrats!
    print("[SUCCESSFUL!] Congrats! Your implementation of ClassificationLoss is correct!")


# In[ ]:


# DO NOT TOUCH THIS CELL!
def init_weights(net):
    """
    Usage: net = Model()
           net.apply(init_weights)
    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 stdv = 1. / math.sqrt(m.weight.size(1))
#                 nn.init.uniform_(m.bias, -stdv, stdv)


# In[37]:


# DO NOT TOUCH THIS CELL!
import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


def summary(model, input_size, batch_size=-1, device=torch.device('cpu'), dtypes=None, verbose=True):
    result, params_info = summary_string(
        model, input_size, batch_size, device, dtypes)
    if verbose:
        print(result)

    return params_info


def summary_string(model, input_size, batch_size=-1, device=torch.device('cpu'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params
            

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    total_conv2d = 0
    total_linear = 0 
    for layer in summary:
        if 'conv2d' in layer.lower():
            total_conv2d += 1
        if 'linear' in layer.lower():
            total_linear += 1 

        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total Conv2d layers: {0:,}".format(total_conv2d) + "\n"
    summary_str += "Total Linear layers: {0:,}".format(total_linear) + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    return summary_str, {'total_params': total_params, 
                         'total_trainable_params': trainable_params,
                         'total_conv2d': total_conv2d,
                         'total_linear': total_linear}


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
class BasicCNNClassifier(torch.nn.Module):
    def __init__(self):
        """
        Task: create a model with:
          2 convolutional layers, followed by 2 linear layers, the last of which should output the logits for each class.
          Check the table given above (section 3.3.2) for more details in the specification.
        """
        # Don't remove the following line. Otherwise, it would raise ```AttributeError: cannot assign module before Module.__init__() call``` exception ERROR!
        super(BasicCNNClassifier, self).__init__() 
        
        # YOUR CODE HERE
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.linear1 = nn.Linear(128 *7*11,4428)
        self.linear2 = nn.Linear(4428,2214)
        self.linear = nn.Linear(2214,10)
        self.relu = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        """
        Your code here
        @Brief: This function takes as input a tensor x of size Bx3x64x64 
        and outputs a "logit" tensor of size Bx6. Do not include a softmax layer 
        here because most of Pytorch's loss functions take "logit" as one of the inputs
        while integrating log() with softmax() into a log_softmax() function.    
        @Inputs: 
          x: torch.Tensor((B,3,64,64)) 
        @return: torch.Tensor((B,6))
        @Note: After the 2nd Conv2d (and ReLU), the intermediate feature has the shape 
               ```B x 16 x 14 x 14```. Before putting it into layer 3 (Linear), 
               make sure to reshape this intermediate feature to ```B x 3136```.
        """
        # YOUR CODE HERE 
        x = self.maxpool2d(self.relu(self.conv1(x)))
        x = self.maxpool2d(self.relu(self.conv2(x)))
        x = self.maxpool2d(self.relu(self.conv3(x)))
        x = self.maxpool2d(self.relu(self.conv4(x)))
        x = self.maxpool2d(self.relu(self.conv5(x)))
#         print(x.shape)
        x = x.view(x.size(0), 128 *7*11)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        
        x = self.linear(x)
        return x
        
def test_BasicCNNClassifier():
    # Run randomseed here to ensure results are consistent
    runRamdomSeed()
    student_net = BasicCNNClassifier()

    # Investigate your network's layers
    # Compare the printed shape with what expected in the specification
    print("\n========= Model summarization ============ ") 
    student_net_info = summary(student_net, (4, 288, 432), device='cpu')

  
    # Check the number of Conv2d layers and Linear layers
    total_conv2d = student_net_info['total_conv2d']
    total_linear = student_net_info['total_linear']
#     if total_conv2d != 2:
#         print(f"[FAIL!] BasicCNNClassifier.forward(...) must contains exactly 2 Conv2d layers!, but yours consists of: {total_conv2d} Conv2d layers")
#         return 
#     if total_linear != 2:
#         print(f"[FAIL!] BasicCNNClassifier.forward(...) must contains exactly 2 Linear layers!, but yours consists of: {total_linear} Linear layers")
#         return 

    # Check total number of parameters
    total_parmas = student_net_info['total_params']
#     if total_parmas != 404766:
#         print(f"[FAIL!] BasicCNNClassifier.forward(...) must contains exactly 404,766 parameters!, but yours consists of: {total_parmas} parameters")
#         print(f"Check kernel size, stride, no of input features, no of output featurs of your Conv2d layer,")
#         print(f"\t and no of input features, no of output featurs of your Linear layer")
#         return 

    # Initialize weights for this network
    student_net.apply(init_weights)

    # Give the network a dummy input of size 2x3x64x64
    batch_size  = 2
    dummy_x  = torch.randn([batch_size, 4, 288, 432])
    logit    = student_net(dummy_x) 

    print("========= Run Model with Dummy Input ============\n") 
#     # Check type of the output logit
#     if not torch.is_tensor(logit):
#         print(f"[FAIL!] BasicCNNClassifier.forward(...) must return a tensor!, but yours returns: {type(logit)}")
#         return 

#     # Check size of the output logit
#     if logit.size() != torch.Size([2,6]):
#         print(f"[FAIL!] BasicCNNClassifier.forward(...) must return a tensor of size torch.Size([2,6])!, but your return's size: {logit.size()}")
#         print(f"Make sure that the number of layers, the layers, number of features, kernel size ... all are correct!")
#         return
  
    # Check value
    expected_logit = torch.tensor([[-1.7501, -2.2915,  0.2061, -0.9353,  0.5314,  1.1240],
                                   [-1.3631, -1.3827,  1.7212, -0.1081, -1.3976,  2.3925]])

#     if torch.norm(logit - expected_logit) > 1e-3:
#         print(f"[FAIL!] BasicCNNClassifier.forward(...) must return \n {expected_logit}")
#         print(f"However, yours returns: \n{logit}")
#         print(f"Make sure that the number of layers, the layers, number of features, kernel size ... all are correct!")
#         return

    # Check to see if we can do backpropagation
    base_loss_obj  = torch.nn.CrossEntropyLoss(reduction='mean')  
    dummy_target   = torch.randint(0, 6, [2])
    loss           =  base_loss_obj(logit, dummy_target)
    loss.backward()
    print(f"Target: {dummy_target}")
    print(f"Logit: {logit}")
    print(f"Loss: {loss.item()}")



    print("----------------------------------------------------------------------------------") 
    print("\n[SUCCESSFUL!] Congrats! Your implementation of ClassificationLoss looks correct!")
    
if NOTEBOOK:
    test_BasicCNNClassifier()


# In[ ]:


# DO NOT TOUCH THIS CELL
from torch import save
from torch import load
from os import path

def check_model_exist_by_name(model_name):
    if os.path.exists(model_name + ".pth") and os.path.isfile(model_name + ".pth"):
        print(f"[Successly trained and saved!] Your model named {model_name}.pth has been saved! DO NOT change the file's name! Just include it in your submission files.")
    else:
        print(f"[Successly trained but failed to save!] Your model named {model_name}.pth has not been saved or saved in a diffrent name from what we expect!")
        import glob
        pt_files = glob.glob("*.pth")
        if pt_files:
            print(f"---> Somehow you've saved models as, {pt_files} which is not what autograder expects!")
            print(f"---> We expect that the trained model to be saved exactly as {model_name}.pth")
            print(f"---> What you can do is to manually rename the trained model to be {model_name}.pth")
        else:
            print(f"---> We found no saved models in *.pth format at all! Manually check if your saved the models in other formats or they are not saved at all!")


def save_model(model, name):
    if isinstance(model, BasicCNNClassifier) or isinstance(model, MyBestCNNClassifier):
        return save(model.state_dict(), name + ".pth")
    
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model(name, device_name='cpu'):
    """
    @Brief: load a model saved in the ".pth" or ".pt" formats
    @Inputs:
        name (str): name of the model (without the extension)
        device_name (str): name of the device i.e: 'cpu', 'cuda:0', that you would want to run the model on.
    @Outputs:
        r (nn.Module): a Pytorch model of either "BasicCNNClassifier" or "MyBestCNNClassifier" (depend on "name" input) 
            with pretrained wieghts.
    """
    # In case students set input name = "*.pth" 
    if "." in name:
        name = name.split('.')[0]
        
    if name == "BasicCNNClassifier":
        r = BasicCNNClassifier()
    elif name == "MyBestCNNClassifier":
        r = MyBestCNNClassifier()
    else:
        raise ValueError(f"model {name} has not been supported! Check the spelling!")
    r.load_state_dict(load(name + ".pth", map_location=device_name))
    return r


# In[ ]:


# DO NOT TOUCH THIS CELL!
def dummy_logging(train_logger, valid_logger):

    """
    @Brief: given two Tensorboard writers (train_logger and valid_logger), write 
    some simple lines of code to dump ```dummy_train_loss``` and ```dummy_train_accuracy```
    as well as ```dummy_valid_loss``` and ```dummy_valid_accuracy``` in each interation
    """

    global_step = 0
    # This is a strongly simplified training loop
    for epoch in range(10):
        torch.manual_seed(epoch)
        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            dummy_train_accuracy = epoch/10. + torch.randn(10)
            global_step += 1
            train_logger.add_scalar('loss',dummy_train_loss,global_step)
            
        train_logger.add_scalar('accuracy',torch.mean(dummy_train_accuracy).item(),epoch)
     
        torch.manual_seed(epoch)
        for iteration in range(10):
            dummy_validation_accuracy = epoch / 10. + torch.randn(10)
        valid_logger.add_scalar('accuracy',torch.mean(dummy_validation_accuracy).item(),epoch)


# In[ ]:


# DO NOT TOUCH THIS CELL!
import numpy as np
from torchvision.transforms import functional as TF

def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()

def predict(model, inputs, device='cpu'):
    inputs = inputs.to(device)
    logits = model(inputs)
    return F.softmax(logits, -1)

def draw_bar(axis, preds, labels=None):
    y_pos = np.arange(6)
    axis.barh(y_pos, preds, align='center', alpha=0.5)
    axis.set_xticks(np.linspace(0, 1, 10))
    
    if labels:
        axis.set_yticks(y_pos)
        axis.set_yticklabels(labels)
    else:
        axis.get_yaxis().set_visible(False)
    
    axis.get_xaxis().set_visible(False)

def visualize_predictions(model=None, model_name=None, device_name='cpu'):
  
    if model is not None:
        model.eval()
    else:
        model = load_model(model_name, device_name)
    
    # Get the device 
    if device_name is not None:
        device = torch.device(device_name)
    model = model.to(device)

    validation_image_path='./img_data/valid' #enter the path 

    dataset = MusicDataset(image_path=validation_image_path)

    f, axes = plt.subplots(2, 10)

    idxes = np.random.randint(0, len(dataset), size=10)

    for i, idx in enumerate(idxes):
        img, label = dataset[idx]
        preds = predict(model, img[None], device=device).detach().cpu().numpy()

        axes[0, i].imshow(TF.to_pil_image(img))
        axes[0, i].axis('off')
        draw_bar(axes[1, i], preds[0], LABEL_ if i == 0 else None)

    plt.show()


# In[ ]:


def load_data(dataset_path, data_transforms=None, num_workers=0, batch_size=128):
    dataset = MusicDataset(dataset_path,data_transforms)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)


# In[ ]:


class Args(object):
  def __init__(self):
    self.learning_rate = 0.0001
    self.log_dir = './my_tensorboard_log_directory'

args = Args();
# Add attributes to args here, such as:
# args.learning_rate = 0.0001
# args.log_dir = './my_tensorboard_log_directory' 


# In[ ]:


from torch import optim

def train(args, model_name="MyBestCNNClassifier"):
    """
    @Brief: training your model. This should include the following items:
        - Initialize the model (already given). Only need to map the model to the device on which you would want to run the model on 
                using the following syntax: 
                model = model.to(device) 
                where device = torch.device(<device_name>), 
                i.e: device = torch.device("cuda:0") or device = torech.device("cpu")
                    
        - Initialize tensorboard summarizers (already given)
        - Initialize data loaders (you need to code up)
        - Initialize the optimizer (you need to code up. Type is of your choice)
        - Initialize the loss function (you should have coded up above)
        - A for loop to iterate through many epochs (up to your choice). In each epoch:
                - Iterate through every mini-batches (remember to map data and labels to the device that you would want to run the model on)
                        - Run the forward path
                        - Get loss
                        - Calculate gradients 
                        - Update the model's parameters
                - Evaluate your model on the validation set
                - Save the model if the performance on the validation set is better using exactly the following line:
                        save_model(model, model_name) 
                 
    @Inputs: 
        Args: object of your choice to carry arguments that you want to use within your training function. 
    @Output: 
        No return is necessary here. 
    """
    # Do not touch the following lines
    # Initialize the model 
    if model_name == "MyBestCNNClassifier":
        model = MyBestCNNClassifier()
    else:
        model = BasicCNNClassifier()
    # Initialize tensorboard loggers
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))
    
    # Create subfolders to save the tensorboard log files
    if not os.path.exists(path.join(args.log_dir, f'train/{model_name}')):
        os.makedirs(path.join(args.log_dir, f'train/{model_name}'))
    if not os.path.exists(path.join(args.log_dir, f'valid/{model_name}')):
        os.makedirs(path.join(args.log_dir, f'valid/{model_name}'))  
    #----------------------------------------
    
    # YOUR CODE HERE
    train_loader = load_data('img_data/train')
    criterion = ClassificationLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 16
    train_loss = []
    for epoch in range(epochs):
      running_loss = 0
      for itr, (image, label) in enumerate(train_loader):
        optimizer.zero_grad()
        y_predicted = model(image)
        loss = criterion(y_predicted, label)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
      train_loss.append(running_loss)
      # if (epoch+1) % 2 == 0:
      #   print(f'epoch: {epoch+1}, loss: {running_loss:.4f}')



    # Don't touch the following lines
    #You are not returning a model, but rather saving it to a file, which you will upload along with the homework4.py file
    save_model(model, model_name) 
    # Make sure the file has been saved 
    assert os.path.exists(model_name + ".pth") and os.path.isfile(model_name + ".pth"), f"[Fail to save your model named {model_name}.pth!"


# In[ ]:


# Train BasicCNNClassifier and save the model
if NOTEBOOK:
    model_name="BasicCNNClassifier"
    train(args, model_name=model_name)
    
    # Make sure that the model you've trained above has already been saved! 
    check_model_exist_by_name(model_name)


# In[ ]:


def accuracy_labels(preds, labels):
    return np.sum(preds == labels)/len(preds)

def get_model_accuracy(model_name, device_name='cpu'):
    """
    @Brief: evaluate the model's accuracy on the test set. In this function, instead of using the model that is already  
            available in the current running session, we attempt to retrieve it using a Pytorch function called 
            torch.load(...). 
            This step is gonna be executed with device_name='cpu' to evaluate your code's submission on gradescope so if it fails here, your submission would likely fail.
    @Inputs: 
        model_name (str): name of the model (without its exitension)
        device_name (str): name of the device on which the model is run i.e: 'cpu', 'cuda:0' ...
    """
    if "." in model_name:
        model_name = model_name.split(".")[0]
        
    data = load_data("./img_data/valid")
    
    device = torch.device(device_name)
    
    model = load_model(model_name, device_name)
    model = model.to(device)
    batch_size = 2
    preds = []
    labels = []
    for (X, Y) in data:
        X = X.to(device)
        Y = Y.to(device)
        y_pred = torch.argmax(model(X), dim = 1).tolist()
        y_pred = map(int, y_pred)
        preds.extend(list(y_pred))
        labels.extend(Y.tolist())
    return accuracy_labels(np.array(preds), np.array(labels))  


# In[ ]:


# Evaluate the accuracy of the two models on the test set.

def test_accuracy_models():
    """
    @Brief: evaluate the accuracy of your models on the validation set. 
    @Inputs: empty 
    @Outputs: empty
    @Note:
        device_name (str): name of the device on which your model is run i.e: 'cpu', 'cuda:0'. To speed up, 
        change it to your cuda device (if you're running using colab or your machine has NVIDIA GPU with all 
        drivers and CUDA installed correctly). 
    For example, 
                device_name = 'cuda:0'

    Before your submission, make sure to set device_name = 'cpu' and run again to make sure that your trained model
    could be loaded and run successfully on CPU on gradescope. 
    """
    
    # Student's TODO
    device_name = 'cpu'

    b = get_model_accuracy("BasicCNNClassifier", device_name=device_name)
    print(b)
#     if (b >= BASIC_ACC_THRESH):
#         print(f"*\tBasic model is successful with accuracy {b} >= {BASIC_ACC_THRESH}")
#     else:
#         print(f"*\tBasic model is NOT successful as accuracy {b} < {BASIC_ACC_THRESH}")

#     a = get_model_accuracy("MyBestCNNClassifier", device_name=device_name)

#     if (a >= BEST_ACC_THRESH):
#         print(f"\n*\tYour best model seems to be alright as it has accuracy {a} >= {BEST_ACC_THRESH}!")
#         print("*\tKeep tuning the architecture/hyperparameters to step up on the leaderboard!")
#     else:
#         print(f"\n*\tYour best model seems to be not good enough to get full credit as it has accuracy {a} < {BEST_ACC_THRESH}!")
#         print("*\tIf your model is already more complicated than the base model, check if it's converged or NOT!")

if NOTEBOOK==1 :
    test_accuracy_models()

