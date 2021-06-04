# Imports 
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from torchvision.utils import make_grid
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from src.model import Discrimiator, Generator, initialize_wieghts
import datetime

import argparse
import os
import shutil
from IPython import get_ipython
import wandb
from src import Data



"""
python src/train.py \

--wandbkey=89cd42a1a18e81da82539c61e2fc34054bdf2627 \
--projectname=AnimeGAN \
--wandbentity=rohitkuk \
--tensorboard=True \
--dataset=anime \
--kaggle_user=rohitkuk \
--kaggle_key=45cd756a5a4449406b323ae680ac9332 \
--batch_size=32 \
--epoch=5 \
--load_checkpoints=True \
"""


# Read more about this.
os.environ['KMP_DUPLICATE_LIB_OK']='True'


parser = argparse.ArgumentParser(description='######DC GAN PAPER IMPLEMENTATION TRAINING MODULE ########')


parser.add_argument('--wandbkey', metavar='wandbkey', default=None,
                    help='Key for Weight and Biases Integration')


parser.add_argument('--projectname', metavar='projectname', default="DC_GAN",
                    help='Key for Weight and Biases Integration')


parser.add_argument('--wandbentity', metavar='wandbentity',
                    help='Entity for Weight and Biases Integration')


parser.add_argument('--tensorboard', metavar='tensorboard', type=bool, default=True,
                    help='Tensorboard Integration')


parser.add_argument('--dataset', choices= ['mnist', 'pokemon', 'anime'], type=str.lower, required=True,
                    help = "Choose the Dataset From MNIST or Pokemon")


parser.add_argument('--kaggle_user',default = None,
                    help = "Kaggle API creds Required to Download Kaggle Dataset")


parser.add_argument('--kaggle_key', default = None,
                    help = "Kaggle API creds Required to Download Kaggle Dataset")


parser.add_argument('--batch_size', metavar='batch_size', type=int , default = 32,
                    help = "Batch_Size")
                    

parser.add_argument('--epoch', metavar='epoch', type=int ,default = 5,
                    help = "Kaggle API creds Required to Download Kaggle Dataset")


parser.add_argument('--load_checkpoints', metavar='load_checkpoints',    default = False,
                    help = "Kaggle API creds Required to Download Kaggle Dataset")
                    
args = parser.parse_args()
    

# Initiating Wiegh and biases for logging



shutil.rmtree("logs") if os.path.isdir("logs") else ""

# Hyper Paramerts
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS   = args.epoch
NOISE_DIM    = 100
IMG_DIM      = 64
lr           = 2e-4
BATCH_SIZE   = args.batch_size
MAPS_GEN     = 64
MAPS_DISC    = 64
IMG_CHANNELS = 3
FIXED_NOISE  = torch.randn(64, NOISE_DIM, 1, 1).to(DEVICE)
GEN_CHECKPOINT = '{}_Generator.pt'.format(args.projectname)
DISC_CHECKPOINT = '{}_Discremenator.pt'.format(args.projectname)


# Transforms
Trasforms = transforms.Compose([
    transforms.Resize(IMG_DIM),
    transforms.CenterCrop(IMG_DIM),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5))
    ])


# Prepare Data

# PokeMon Data
if args.dataset !="MNIST":
    Data.kaggle_dataset(args)

# MNIST Still to Implemet in the Data Module
if args.dataset =="MNIST":
    Data.MNIST(args)



    


# Data Loaders
train_dataset = datasets.ImageFolder(root = 'data', transform=Trasforms)
train_loader   = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last=True)

if args.wandbkey :
    wandb_integration = True
    wandb.login(key = args.wandbkey)
    wandb.init(project = args.projectname,  entity=args.wandbentity, resume=True)

print(wandb.run.name)

# Loading Generator
if os.path.isdir(os.path.join(wandb.run.dir, GEN_CHECKPOINT)) and args.load_checkpoints:
    generator = torch.load(wandb.restore(GEN_CHECKPOINT).name)
else:
    generator = Generator(noise_channels=NOISE_DIM, img_channels=IMG_CHANNELS, maps=MAPS_GEN).to(DEVICE)

# Loading Discremenator
if os.path.isdir(os.path.join(wandb.run.dir, DISC_CHECKPOINT)) and args.load_checkpoints:
    discremenator = torch.load(wandb.restore(DISC_CHECKPOINT).name)
else:
    discremenator = Discrimiator(num_channels=IMG_CHANNELS, maps=MAPS_DISC).to(DEVICE)


# weights Initialize
initialize_wieghts(generator)
initialize_wieghts(discremenator)


# Loss and Optimizers
gen_optim = optim.Adam(params = generator.parameters(), lr=lr, betas=(0.5, 0.999))
disc_optim = optim.Adam(params = discremenator.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()


# Tensorboard Implementation
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")

    
    
if args.wandbkey :
    wandb.watch(generator)
    wandb.watch(discremenator)


# Code for COLLAB TENSORBOARD VIEW
try:
    get_ipython().magic("%load_ext tensorboard")
    get_ipython().magic("%tensorboard --logdir logs")
except:
    pass

# training
discremenator.train()
generator.train()
step = 0
images = []
for epoch in range(1, NUM_EPOCHS+1):
    tqdm_iter = tqdm(enumerate(train_loader), total = len(train_loader), leave = False)

    for batch_idx, (data, _) in tqdm_iter:
        data = data.to(DEVICE)
        batch_size = data.shape[0]
        
        # ====================== Training the Discremnator===============
        latent_noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(DEVICE)
        fake_img = generator(latent_noise)
        
        disc_fake = discremenator(fake_img.detach()).reshape(-1)
        disc_real = discremenator(data).reshape(-1)

        disc_fake_loss = criterion(disc_fake, torch.zeros_like(disc_fake))
        disc_real_loss = criterion(disc_real, torch.ones_like(disc_real))
        disc_loss = (disc_fake_loss+disc_real_loss)/2

        discremenator.zero_grad()
        disc_loss.backward()
        disc_optim.step()

        # ====================== Training the Generator===============
        # gen_img  = generator(latent_noise)
        output = discremenator(fake_img).reshape(-1)
        gen_loss = criterion(output, torch.ones_like(output))
        generator.zero_grad()
        gen_loss.backward()
        gen_optim.step()
        
        # Logger
        tqdm_iter.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        tqdm_iter.set_postfix(disc_loss = "{0:.4f}".format(disc_loss.item()), gen_loss = "{0:.4f}".format(gen_loss.item()))

        # for Tensorboard

        if batch_idx % 30 == 0 :
            torch.save(generator.state_dict(), os.path.join("model", GEN_CHECKPOINT))
            torch.save(discremenator.state_dict(), os.path.join("model", DISC_CHECKPOINT))
            if args.tensorboard:
                GAN_gen = generator(FIXED_NOISE)
                img_grid_real = make_grid(data[:32], normalize=True)
                img_grid_fake = make_grid(GAN_gen[:32], normalize=True)
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                images.append(img_grid_fake.cpu().detach().numpy())
                step +=1

            if args.wandbkey:
                wandb.log({"Discremenator Loss": disc_loss.item(), "Generator Loss": gen_loss.item()})
                wandb.log({"img": [wandb.Image(img_grid_fake, caption=step)]})
                torch.save(generator.state_dict(), os.path.join(wandb.run.dir, GEN_CHECKPOINT))
                torch.save(discremenator.state_dict(), os.path.join(wandb.run.dir, DISC_CHECKPOINT))





import numpy as np
from IPython.display import HTML
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
matplotlib.rcParams['animation.embed_limit'] = 2**64
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = []


from matplotlib import animation
for j,i in tqdm(enumerate(images)):
    ims.append([plt.imshow(np.transpose(i,(1,2,0)), animated=True)]) 
    

ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
HTML(ani.to_jshtml())
f = "animation{}.gif".format(datetime.datetime.now()).replace(":","")

from matplotlib.animation import PillowWriter

ani.save(os.path.join(wandb.run.dir,f), writer=PillowWriter(fps=20)) 
ani.save(f, writer=PillowWriter(fps=20)) 
