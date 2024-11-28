from torch import nn
import torch 
from tqdm import tqdm
from torchvision.utils import save_image

class Discriminator(nn.Module):
    def __init__(self, i_cha = 3):
        super(Discriminator, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(i_cha, 64, kernel_size=4, stride=2, padding=1, padding_mode="reflect"), 
            nn.LeakyReLU(0.2)
        )
        
        self.model = nn.Sequential(
            nn.Conv2d(64, 128, stride=2, kernel_size=4, padding=1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, stride=2, kernel_size=4, padding=1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, stride=1,  kernel_size=4, padding=1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        )
        
    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))   


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super(ResidualBlock, self).__init__()
        
        self.seq = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(ch),
            nn.Identity()
        )


    def forward(self, x):
        return x + self.seq(x)
    
class Generator(nn.Module):
    def __init__(self, i_ch, num_f=64, num_r=9):
        """Create the generator

        Args:
            i_ch (_type_): number of channels
            num_f (int, optional): number of features. Defaults to 64.
            num_r (int, optional): number of residuals. Defaults to 9.
        """
        super(Generator, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(i_ch, num_f, kernel_size=7, stride= 1, padding=3, padding_mode = "reflect"), 
            nn.ReLU(inplace=True))
        
        # nn.ModuleList is just an array store
        self.down = nn.Sequential(
            nn.Conv2d(num_f, num_f*2, padding_mode="reflect", kernel_size=3, stride=2, padding =1),
            nn.InstanceNorm2d(num_f*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_f*2, num_f*4, padding_mode="reflect", kernel_size=3, stride=2, padding =1),
            nn.InstanceNorm2d(num_f*4),
            nn.ReLU(inplace=True)
            
        )
        self.residuals = nn.Sequential(
            *[ResidualBlock(num_f*4) for _ in range(num_r)]
        )
        
        self.up = nn.Sequential(
            nn.ConvTranspose2d(num_f*4, num_f*2, kernel_size=3, stride= 2, padding=1, output_padding=1),
            nn.InstanceNorm2d(num_f*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_f*2, num_f, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(num_f),
            nn.ReLU(inplace=True)
        )
        
        self.last = nn.Conv2d(num_f, i_ch, kernel_size=7, stride=1, padding=3, padding_mode="reflect")
        
    def forward(self, x):
        x = self.initial(x)
        x = self.down(x)
        x = self.residuals(x)
        x = self.up(x)
        return torch.tanh(self.last(x))

def testGen():
    i_cha = 3
    img_size = 256
    x = torch.randn((2, i_cha, img_size, img_size))
    gen = Generator(i_cha, 9)
    print(gen)
    print(gen(x).shape)


def testDisc():
        x = torch.randn((5,3,256,256))
        model = Discriminator(i_cha=3)
        preds = model(x)
        print(preds.shape)   


def save_chk(model, opt, filename="../checkpoints/chk.pth.tar"):
    print("Saving checkpoits...")
    chk = {
        "weights": model.state_dict(),
        "opt": opt.state_dict()
    }
    torch.save(chk, filename)

def load_chk(chk_file, model, opt, lr, device):
    print("Loading checkpoints")
    chk = torch.load(chk_file, map_location=device)
    model.load_state_dict(chk["weights"])
    opt.load_state_dict(chk["opt"])
    
    for param in opt.param_group():
        param["lr"] = lr

def train(gen_monet, disc_monet, gen_photo, disc_photo, loader, device,
          opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch):
    
    for idx, (photo, monet) in enumerate(tqdm(loader, leave=True)):
        monet = monet.to(device)
        photo = photo.to(device)
        
        # allows to run training of different precision
        with torch.amp.autocast("cuda"):
            fake_photo = gen_photo(monet)
            disc_photo_real = disc_photo(photo)
            disc_photo_fake = disc_photo(fake_photo.detach())
            disc_photo_real_loss = mse(disc_photo_real, torch.ones_like(disc_photo_real))
            disc_photo_fake_loss = mse(disc_photo_fake, torch.zeros_like(disc_photo_fake))
            disc_photo_loss = disc_photo_real_loss + disc_photo_fake_loss
            
            fake_monet = gen_monet(photo)
            disc_monet_real = disc_monet(monet)
            disc_monet_fake = disc_monet(fake_monet.detach())
            disc_monet_real_loss = mse(disc_monet_real, torch.ones_like(disc_monet_real))
            disc_monet_fake_loss = mse(disc_monet_fake, torch.ones_like(disc_monet_fake))
            disc_monet_loss = disc_monet_real_loss + disc_monet_fake_loss
            
            disc_loss = (disc_photo_loss + disc_monet_loss)/2
            
        opt_disc.zero_grad()
        d_scaler.scale(disc_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        
        with torch.amp.autocast("cuda"):
            disc_photo_fake = disc_photo(fake_photo)
            disc_monet_fake = disc_monet(fake_monet)
            loss_gen_mon = mse(disc_monet_fake, torch.ones_like(disc_monet_fake))
            loss_gen_photo = mse(disc_photo_fake, torch.ones_like(disc_photo_fake))
            
            cylce_monet = gen_monet(fake_photo)
            cylce_photo = gen_photo(fake_monet)
            cylce_monet_loss = l1(monet, cylce_monet)
            cylce_photo_loss = l1(photo, cylce_photo)
            
            idt_monet = gen_monet(monet)
            idt_photo = gen_photo(photo)
            idt_monet_loss = l1(monet, idt_monet)
            idt_photo_loss = l1(photo, idt_photo)
            
            gen_loss =  ((loss_gen_mon + loss_gen_photo) + 
                         (cylce_monet_loss + cylce_photo_loss)*10 + 
                         (idt_monet_loss + idt_photo_loss)*0)
        
        opt_gen.zero_grad()
        g_scaler.scale(gen_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        
        if idx % 200 == 0:
            save_image(fake_photo*0.5 + 0.5, f"../saved_images/photo_{epoch}_idx.jpg")
            save_image(fake_monet*0.5 + 0.5, f"../saved_images/monet_{epoch}_idx.jpg")
            
            
            
    