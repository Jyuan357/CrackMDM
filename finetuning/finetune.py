# import imp
import shutil
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
from utils.utils import *
from utils.Validator import *
from utils.Crackloader import *
from utils.lossFunctions import *

from utils.loadpre import load_from, load_raw
from nets import *

import time

device_id = 0
torch.cuda.set_device(device_id)


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = torch.sigmoid(inputs)       

        inputs = inputs.view(-1)
        targets = targets.view(-1)
 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU
IoUloss=IoULoss()
def calculate_loss(outputs, labels):
    loss = 0
    loss = cross_entropy_loss_RCF(outputs, labels)
    return loss
def calculate_IoU_loss(outputs, labels):
    loss = 0
    loss = IoUloss(outputs, labels)
    return loss
def trainer(net, total_epoch, lr_init, batch_size,train_img_dir, valid_img_dir, valid_lab_dir,
            valid_result_dir, valid_log_dir, best_model_dir, image_format, lable_format, datasetName, size, pretrain_dir=None):

    img_data = Crackloader(txt_path=train_img_dir, size=size, normalize=False)

    img_batch = data.DataLoader(img_data, batch_size=batch_size, shuffle=True, num_workers=2)


    if pretrain_dir is not None:
        model_dict = load_from(pretrain_dir, net)
        net.load_state_dict(model_dict)


    crack = net.cuda()  

    validator = Validator(valid_img_dir, valid_lab_dir,
                          valid_result_dir, valid_log_dir, best_model_dir, crack,  image_format, lable_format,datasetName, size=size)

    log_dir=valid_log_dir+'/'+datasetName+'_loss.txt'
    
    net_str=str(net)
    f=open(log_dir,'a')
    f.writelines(net_str)
    f.close()
    for epoch in range(1, total_epoch):

        losses = Averagvalue()

        crack.train() 

        count = 0 

        new_lr = updateLR(lr_init, epoch, total_epoch)

        print('Learning Rate: {:.9f}'.format(new_lr))

        # optimizer
        lr = new_lr
        # optimizer = torch.optim.Adam(crack.parameters(), lr=lr)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, crack.parameters()), lr=lr)


        for (images, labels) in img_batch:
            f=open(log_dir,'a')
            count += 1
            loss = 0

            images = Variable(images).cuda()
            labels = Variable (labels.float()).cuda()
            
            output = crack.forward(images)
         
            loss += calculate_loss(torch.squeeze(output, dim=1), labels)
            

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            losses.update(loss.item(), images.size(0))
            lr = optimizer.param_groups[0]['lr']
            if count%20==0:
                info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, total_epoch, count, len(img_batch)) + \
                       'Loss {loss.val:f} (avg:{loss.avg:f} lr {lr:.10f}) '.format(
                           loss=losses, lr=lr)
                print(info)
                f.writelines(time.strftime('%m-%d %X')+" "+info+'\n')
            f.close()
      
        if epoch < 4 or (epoch+1) % 5 == 0: 
            print("test.txt valid")
            validator.validate(epoch)
        if (epoch + 1) == total_epoch:
            torch.save(net.state_dict(), best_model_dir + str(epoch) + ".pth")

def check_dir(path):
    if os.path.exists(path)==False:
        os.makedirs(path)



if __name__ == '__main__':

    datasetName="YCD"   
    size = 448 
    dataName= "train"
    netName='afin_448px_bs8_epoch500' # 要改
    image_format = "jpg"
    lable_format = "jpg"

    total_epoch = 500
    lr_init=0.0001
    batch_size = 8 

    net = CrackMiM()

    train_img_dir = ""
    valid_img_dir = ""
    valid_lab_dir = ""
    
    
    valid_result_dir = "./valid_result/"+netName+"/"+datasetName+"/"
    valid_log_dir = "./log/" + netName 
    best_model_dir = "./model/" + datasetName +"/"+netName+'/'
    
    check_dir(valid_log_dir)
    check_dir(valid_result_dir)

    pretrain_dir = '.pth'

    trainer(net, total_epoch, lr_init, batch_size, train_img_dir, valid_img_dir, valid_lab_dir,
            valid_result_dir, valid_log_dir, best_model_dir,  image_format, lable_format,datasetName, pretrain_dir=pretrain_dir, size=size) #

    print("训练结束")

