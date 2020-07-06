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
val_loader = DataLoader(dataset_val, batch_size=8, shuffle=True, num_workers=(6 if n_GPU==1 else n_GPU*6),
    drop_last=True)

netG = netG.float()
if n_GPU > 1:
    netG = nn.DataParallel(netG)
netG.to(device)

netD = Discriminator(output=output_res)
netD = netD.float()
if n_GPU > 1:
    netD = nn.DataParallel(netD)
netD.to(device)

criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)

writer = SummaryWriter('../logs/'+args["version"]+'/')

file = open("../logs/"+args["version"]+'/'+"architecture.txt","w")
file.write("Batch size: {}\n".format(batch_size))
file.write("Learning rate: {}\n".format(args["learningrate"]))
file.write("Output resolution: {}\n".format(output_res))
file.write(str(netD))
file.write(str(netG))
file.close()

learning_rate = args["learningrate"]
optimizer_d = torch.optim.Adam(netD.parameters(),lr=learning_rate,betas=(0.5,0.999))
optimizer_g = torch.optim.Adam(netG.parameters(),lr=learning_rate,betas=(0.5,0.999))

if args["checkpoint"] is None:
    checkpoint_epoch=0
else:
    raise Exception("Resume from checkpoint not implemented")
    #checkpoint = torch.load("../checkpoints/checkpoint_"+args["version"]+"_"+str(args["checkpoint"])+".pth")
    #model.load_state_dict(checkpoint["state_dict"])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    #checkpoint_epoch = checkpoint["epoch"]

epochs = args["epochs"]

for param_group in optimizer_d.param_groups:
    print("Learning rate used: {}".format(param_group['lr']))

train_iter = 0

for epoch in range(checkpoint_epoch,epochs):

    print('* Epoch %d/%d' % (epoch, epochs))

    t0 = time.time()

    avgG_loss = 0
    avgG_GAN_loss = 0
    avgG_L1_loss = 0
    avgG_val_loss = 0

    avgG_acc = 0
    avgG_gan_acc = 0
    avgG_val_acc = 0

    avgD_loss = 0
    avgD_val_loss = 0

    avgD_real_acc = 0
    avgD_fake_acc = 0
    
    # ------ TRAINING ---------
    
    netD.train()
    netG.train()

    for i,(x1_batch, x2_batch, y_batch) in enumerate(train_loader):
        
        x1_batch = x1_batch.to(device)
        x2_batch = x2_batch.to(device)
        y_batch  = y_batch.to(device)

        fake_b = netG(x1_batch,x2_batch)

        ######################
        # (1) Update D network
        ######################

        optimizer_d.zero_grad()

        # train with fake
        pred_fake = netD(fake_b.detach())
        loss_d_fake = criterionGAN(pred_fake,False)

        label = torch.full((pred_fake.size(0),pred_fake.size(-1),pred_fake.size(-1)), 0, device=device)  
        acc_d_fake = ((1-torch.abs(pred_fake-label))*100).mean()

        #train with real
        pred_real = netD(y_batch)
        loss_d_real = criterionGAN(pred_real,True)

        label.fill_(1)
        acc_d_real = ((1-torch.abs(pred_real-label))*100).mean()

        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5
        loss_d.backward()

        optimizer_d.step()


        ######################
        # (2) Update G network
        ######################

        optimizer_g.zero_grad()

        # First try to fake the discriminator
        pred_fake = netD(fake_b)
        loss_g_gan = criterionGAN(pred_fake,True)

        label.fill_(1)
        acc_g_gan = ((1-torch.abs(pred_fake-label))*100).mean()

        loss_g_l1 = criterionL1(fake_b[y_batch!=0],y_batch[y_batch!=0]) * 100

        loss_g = loss_g_gan + loss_g_l1
        loss_g.backward()

        optimizer_g.step()


        ######################
        # (3) Write metrics
        ######################

        acc = ((1-torch.abs(fake_b-y_batch))*100)
        acc = acc[y_batch!=0].mean()
        
        avgG_loss += loss_g / len(train_loader)
        avgG_GAN_loss += loss_g_gan / len(train_loader)
        avgG_L1_loss += loss_g_l1 / len(train_loader)
        avgD_loss += loss_d / len(train_loader)

        avgG_acc += acc / len(train_loader)
        avgG_gan_acc += acc_g_gan / len(train_loader)
        avgD_fake_acc += acc_d_fake / len(train_loader)
        avgD_real_acc += acc_d_real / len(train_loader)

        
        print('[{}/{}] - Loss_D: {:.4f} - Loss_G: {:.4f} - Loss_G_gan: {:.4f} - Loss_G_L1: {:.4f} - Acc_G_L1: {:.3f} - Acc_G_gan: {:.3f} - Acc_D_real: {:.3f} - Acc_D_fake: {:.3f}'.format(i,
            len(train_loader)-1,loss_d,loss_g,loss_g_gan,loss_g_l1,acc,acc_g_gan,acc_d_real,acc_d_fake), end="\r",flush=True)

        for param_group in optimizer_d.param_groups:
            writer.add_scalar('Learning_rate', param_group['lr'], train_iter)
        train_iter +=1
        
        
       
    print('[{}/{}] - Loss_D: {:.4f} - Loss_G: {:.4f} - Loss_G_gan: {:.4f} - Loss_G_L1: {:.4f} - Acc_G_L1: {:.3f} - Acc_G_gan: {:.3f} - Acc_D_real: {:.3f} - Acc_D_fake: {:.3f}'.format(i,
            len(train_loader)-1,avgD_loss,avgG_loss,avgG_GAN_loss,avgG_L1_loss,avgG_acc,avgG_gan_acc,avgD_real_acc,avgD_fake_acc),flush=True)
    
    writer.add_scalar('Train/Loss_D', avgD_loss, epoch)
    writer.add_scalar('Train/Loss_G', avgG_loss, epoch)
    writer.add_scalar('Train/Loss_G_GAN', avgG_GAN_loss, epoch)
    writer.add_scalar('Train/Loss_G_L1', avgG_L1_loss, epoch)
    writer.add_scalar('Train/Accuracy_G_L1', avgG_acc, epoch)
    writer.add_scalar('Train/Accuracy_G_gan', avgG_gan_acc, epoch)
    writer.add_scalar('Train/Accuracy_D_real', avgD_real_acc, epoch)
    writer.add_scalar('Train/Accuracy_D_fake', avgD_fake_acc, epoch)
    
    
    
    # ------- VALIDATION ------------
    netG.eval()

    with torch.no_grad():
    
        for x1_val, x2_val, y_val in val_loader:
            
            x1_val = x1_val.to(device)
            x2_val = x2_val.to(device)

            y_val = y_val.to(device)        
            
            y_pred_val = netG(x1_val,x2_val)

            loss_val = criterionL1(y_pred_val[y_val!=0],y_val[y_val!=0])
            
            acc_val = ((1-torch.abs(y_pred_val-y_val))*100)
            acc_val = acc_val[y_val!=0].mean()
            
            avgG_val_loss += loss_val / len(val_loader)
            avgG_val_acc += acc_val / len(val_loader)
            

        
    print(" - val loss: {:.4f} - acc: {:.3f}".format(avgG_val_loss,avgG_val_acc))

    epoch_time = time.time()-t0
    print(' - epoch time: {:.1f}'.format(epoch_time))


    writer.add_scalar('Val/Loss', avgG_val_loss, epoch)
    writer.add_scalar('Val/Accuracy', avgG_val_acc, epoch)
    writer.add_scalar('Epoch_time',epoch_time,epoch)
    
    
    
    
    # ------- IMAGES TO TENSORBOARD ------------
    images = vutils.make_grid(fake_b[:,-1,:,:].unsqueeze(1), normalize=True, scale_each=True)
    writer.add_image('Train/Pred', images, epoch)
    
    images = vutils.make_grid(y_batch[:,-1,:,:].unsqueeze(1), normalize=True, scale_each=True)
    writer.add_image('Train/True', images, epoch)
    
    images = vutils.make_grid(y_pred_val[:,-1,:,:].unsqueeze(1), normalize=True, scale_each=True)
    writer.add_image('Val/Pred', images, epoch)
    
    images = vutils.make_grid(y_val[:,-1,:,:].unsqueeze(1), normalize=True, scale_each=True)
    writer.add_image('Val/True', images, epoch)
    
    if epoch % 5 == 0:

        stateD = {
            'epoch': epoch,
            'state_dict': netD.state_dict(),
            'optimizer': optimizer_d.state_dict()
        }
        torch.save(stateD, "../checkpoints/checkpoint_"+args["version"]+"_D_"+str(epoch)+".pth")

        stateG = {
            'epoch': epoch,
            'state_dict': netG.state_dict(),
            'optimizer': optimizer_g.state_dict()
        }
        torch.save(stateG, "../checkpoints/checkpoint_"+args["version"]+"_G_"+str(epoch)+".pth")
        

        #for name, param in model.named_parameters():
        #    if 'bn' not in name:
        #        writer.add_histogram('Params/'+name, param, epoch)
        #        writer.add_histogram('Grads/'+name, param.grad, epoch)
        #        writer.add_scalar('Params_L2norm/'+name,(param**2).mean().sqrt(),epoch)
        #        writer.add_scalar('Gradients_L2norm/'+name,(param.grad**2).mean().sqrt(),epoch)

   
