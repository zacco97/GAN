from dataset import DatasetMonet
import albumentations as A
from albumentations.pytorch import ToTensorV2 
from model import testDisc, testGen, Discriminator, Generator, train, save_chk
from torch import optim, nn
from torch.utils.data import DataLoader
import torch

MONET_DIR = r"D:\ProjectData\gan-getting-started\monet_jpg"
PHOTO_DIR = r"D:\ProjectData\gan-getting-started\photo_jpg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run() -> None: 
    
    # create dataset
    transform = A.Compose([A.Resize(256, 256), A.Normalize(0.5,0.5), ToTensorV2()], additional_targets= {"image0": "image"})
    
    dataset = DatasetMonet(root_dir = PHOTO_DIR, target_dir=MONET_DIR, transform=transform)
    
    # dataset.visualize(100)
    
    print(DEVICE)
    # create model
    disc_photo = Discriminator(3).to(DEVICE)
    disc_monet = Discriminator(3).to(DEVICE)
    gen_photo = Generator(3, 9).to(DEVICE)
    gen_monet = Generator(3, 9).to(DEVICE)
    
    opt_disc = optim.Adam(list(disc_photo.parameters()) + list(disc_monet.parameters()), 
                          lr=1e-4)
    
    opt_gen = optim.Adam(list(gen_photo.parameters()) + list(gen_monet.parameters()), 
                          lr=1e-4)
    
    L1 = nn.L1Loss()
    mse = nn.MSELoss()
    
    loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
    
    # t, t1 = next(iter(loader))
    
    g_scaler = torch.amp.GradScaler()
    d_scaler = torch.amp.GradScaler()
    
    for epoch in range(5):
        train(gen_monet=gen_monet, disc_monet=disc_monet, gen_photo=gen_photo,
              disc_photo=disc_photo, loader=loader, device=DEVICE, opt_disc=opt_disc, 
              opt_gen=opt_gen, l1=L1, mse=mse, d_scaler=d_scaler, g_scaler=g_scaler,
              epoch=epoch)
        
        save_chk(gen_monet, opt_gen, 
                 filename="../checkpoints/genM.pth.tar")
        save_chk(gen_photo, opt_gen, 
                 filename="../checkpoints/genP.pth.tar")
        save_chk(disc_monet, opt_disc, 
                 filename="../checkpoints/discM.pth.tar")
        save_chk(disc_monet, opt_disc, 
                 filename="../checkpoints/discP.pth.tar")