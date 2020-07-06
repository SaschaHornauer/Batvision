from Dataloaders import *
from Models import *
from torch.utils.data import DataLoader
from time import time
import torch
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import argparse
import time
import setproctitle as spt
import matplotlib.pyplot as plt
import math
import pickle


ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=int, required=True,
    help="number of epochs")
ap.add_argument("-l", "--learningrate", type=float, required=True,
    help="learning rate")
ap.add_argument("-v", "--version", type=str, required=True,
    help="something to remember this model by")
ap.add_argument("-c", "--checkpoint", type=int, required=False,
    help="checkpoint for continue training")
ap.add_argument("-g", "--generator", type=str, required=True,
    help="unet | direct")
ap.add_argument("-a","--audioencoder", type=str, required=True,
    help="waveform | spectrogram")
ap.add_argument("-r","--resolution",type=int,required=True,
    help="output resolution - 32 | 64 | 128")
ap.add_argument("-o","--objective",type=str,required=True,
    help="Training objective - grayscale | depth")

args = vars(ap.parse_args())

spt.setproctitle(args["version"])

torch.backends.cudnn.benchmark = True

CSV_TRAIN = "../data/2019.08.14/specCW_raw_long_image_meas_train2.csv"
CSV_VAL = "../data/2019.08.14/specCW_raw_long_image_meas_val2.csv"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_GPU = torch.cuda.device_count()
print("{} {} device is used".format(n_GPU,device))

batch_size = 16*n_GPU

output_res = args["resolution"]
if output_res != 32 and output_res != 64 and output_res != 128:
    raise Exception("Output resolution: use 32 | 64 | 128")

if args["audioencoder"] == "waveform":
    if args["objective"] == "grayscale":
        dataset_train = raw_to_image(csv_file=CSV_TRAIN,output=output_res)
        dataset_val = raw_to_image(csv_file=CSV_VAL,output=output_res)
    elif args["objective"] == "depth":
        dataset_train = raw_to_depth(csv_file=CSV_TRAIN,output=output_res)
        dataset_val = raw_to_depth(csv_file=CSV_VAL,output=output_res)
    else:
        raise Exception("Objective: use grayscale or depth")

    model = WaveformNet(generator=args["generator"],output=output_res)

elif args["audioencoder"] == "spectrogram":
    if args["objective"] == "grayscale":
        dataset_train = spec_to_image(csv_file=CSV_TRAIN,output=output_res)
        dataset_val = spec_to_image(csv_file=CSV_VAL,output=output_res)
    elif args["objective"] == "depth":
        dataset_train = spec_to_depth(csv_file=CSV_TRAIN,output=output_res)
        dataset_val = spec_to_depth(csv_file=CSV_VAL,output=output_res)
    else:
        raise Exception("Objective: use grayscale or depth")  

    model = SpectrogramNet(generator=args["generator"],output=output_res)

else:
    raise Exception("Audio Encoder: use waveform or spectrogram")

print("Number of train samples: {}".format(len(dataset_train)))
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=(6 if n_GPU==1 else n_GPU*8),
    drop_last = True)

print("Number of validation samples: {}".format(len(dataset_val)))
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=(6 if n_GPU==1 else n_GPU*6),
    drop_last=True)


model = model.float()
if n_GPU > 1:
    model = nn.DataParallel(model)
model.to(device)

criterion = nn.L1Loss(reduction='none')

writer = SummaryWriter('../logs/'+args["version"]+'/')

file = open("../logs/"+args["version"]+'/'+"architecture.txt","w")
file.write("Batch size: {}\n".format(batch_size))
file.write("Learning rate: {}\n".format(args["learningrate"]))
file.write("Output resolution: {}\n".format(output_res))
file.write(str(model))
file.close()

learning_rate = args["learningrate"]
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if args["checkpoint"] is None:
    checkpoint_epoch=0
       
else:
    checkpoint = torch.load("../checkpoints/checkpoint_"+args["version"]+"_"+str(args["checkpoint"])+".pth")
    model.load_state_dict(checkpoint["state_dict"])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint_epoch = checkpoint["epoch"]

epochs = args["epochs"]


for param_group in optimizer.param_groups:
    print("Learning rate used: {}".format(param_group['lr']))


train_iter = 0

for epoch in range(checkpoint_epoch,epochs):

    print('* Epoch %d/%d' % (epoch, epochs))

    t0 = time.time()

    avg_loss = 0
    avg_val_loss = 0

    avg_acc = 0
    avg_val_acc = 0
    
    # ------ TRAINING ---------
    
    model.train()  # train mode
    

    for i,(x1_batch, x2_batch, y_batch) in enumerate(train_loader):
        
        x1_batch = x1_batch.to(device)
        x2_batch = x2_batch.to(device)

        
        y_batch = y_batch.to(device)    
           
        # set parameter gradients to zero
        optimizer.zero_grad()

        # forward
        y_pred = model(x1_batch,x2_batch)

        loss = criterion(y_pred,y_batch)
        loss = loss[y_batch!=0].mean()

        loss.backward()  # backward-pass
        optimizer.step()  # update weights

        acc = ((1-torch.abs(y_pred-y_batch))*100)
        acc = acc[y_batch!=0].mean()

        
        # calculate metrics 
        avg_loss += loss / len(train_loader)
        avg_acc += acc / len(train_loader)
        
        print('[{}/{}] - loss: {:.4f} - acc: {:.3f}'.format(i,len(train_loader)-1,loss,acc), end="\r",flush=True)

        for param_group in optimizer.param_groups:
            writer.add_scalar('Learning_rate', param_group['lr'], train_iter)
        train_iter +=1
        break
       
    print('[{}/{}] - loss: {:.4f} - acc: {:.3f}'.format(i,len(train_loader)-1,avg_loss, avg_acc))
    
    writer.add_scalar('Train/Loss', avg_loss, epoch)
    writer.add_scalar('Train/Accuracy', avg_acc, epoch)
    
    
    
    # ------- VALIDATION ------------
    
    model.eval()  # val mode

    with torch.no_grad():
    
        for x1_val, x2_val, y_val in val_loader:
            
            x1_val = x1_val.to(device)
            x2_val = x2_val.to(device)

            y_val = y_val.to(device)        
            
            y_pred_val = model(x1_val,x2_val)

            loss_val = criterion(y_pred_val,y_val)
            loss_val = loss_val[y_val!=0].mean()

            acc_val = ((1-torch.abs(y_pred_val-y_val))*100)
            acc_val = acc_val[y_val!=0].mean()
            
            avg_val_loss += loss_val / len(val_loader)
            avg_val_acc += acc_val / len(val_loader)
            break
        
    print(' - val loss: {:.4f} - acc: {:.3f}'.format(avg_val_loss,avg_val_acc))

    epoch_time = time.time()-t0
    print(' - epoch time: {:.1f}'.format(epoch_time))


    writer.add_scalar('Val/Loss', avg_val_loss, epoch)
    writer.add_scalar('Val/Accuracy', avg_val_acc, epoch)
    writer.add_scalar('Epoch_time',epoch_time,epoch)
    
    
    
    
    # ------- IMAGES TO TENSORBOARD ------------
        
    
    if epoch % 1 == 0:

        images = vutils.make_grid(y_pred[:,-1,:,:].unsqueeze(1), normalize=True, scale_each=True)
        writer.add_image('Train/Pred', images, epoch)
        
        images = vutils.make_grid(y_batch[:,-1,:,:].unsqueeze(1), normalize=True, scale_each=True)
        writer.add_image('Train/True', images, epoch)
        
        images = vutils.make_grid(y_pred_val[:,-1,:,:].unsqueeze(1), normalize=True, scale_each=True)
        writer.add_image('Val/Pred', images, epoch)
        
        images = vutils.make_grid(y_val[:,-1,:,:].unsqueeze(1), normalize=True, scale_each=True)
        writer.add_image('Val/True', images, epoch)
        
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, "../checkpoints/checkpoint_"+args["version"]+"_"+str(epoch)+".pth")

        

        #for name, param in model.named_parameters():
        #    if 'bn' not in name:
        #        writer.add_histogram('Params/'+name, param, epoch)
        #        writer.add_histogram('Grads/'+name, param.grad, epoch)
        #        writer.add_scalar('Params_L2norm/'+name,(param**2).mean().sqrt(),epoch)
        #        writer.add_scalar('Gradients_L2norm/'+name,(param.grad**2).mean().sqrt(),epoch)

   
