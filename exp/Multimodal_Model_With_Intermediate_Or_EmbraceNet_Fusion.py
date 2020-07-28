from exp.Required_Modules_And_Packages import *

# Multimodal Model with Intermediate Fusion

class IntermediateConcatModel(nn.Module):
    def __init__(self, mod_img, mod_clas, layers, drops,with_transformer=False):
        super().__init__()
        self.mod_img = mod_img
        self.mod_clas = mod_clas
        self.with_transformer = with_transformer
        #layers of terminal network
        terminal_layers = bn_drop_lin(layers[0],layers[-1], p=drops, actn=None)
       
        self.layers = nn.Sequential(*terminal_layers)

    def forward(self,*x):
        x_img = self.mod_img(x[0])
        if self.with_transformer: x_clas = self.mod_clas(x[1].permute(1,0))[0]
        else: x_clas = self.mod_clas(x[1])[0]
        #Intermediate Fusion
        x = torch.cat([x_img, x_clas], dim=1)
        x = self.layers(x)
        return x

# Multimodal Model with EmbraceNet Fusion

class EmbraceNetConcatModel(nn.Module):
    def __init__(self, mod_img, mod_clas, layers, drops,embrace_layer_size,with_transformer=False):
        super().__init__()
        self.mod_img = mod_img
        self.mod_clas = mod_clas
        self.embrace_size = embrace_layer_size
        self.with_transformer = with_transformer
        self.lin_img = nn.Linear(512,self.embrace_size)
        self.lin_clas = nn.Linear(layers[0]-512,self.embrace_size)
        self.relu = nn.ReLU()
        #layers of terminal network
        terminal_layers = bn_drop_lin(self.embrace_size,layers[-1], p=drops, actn=None)
       
        self.layers = nn.Sequential(*terminal_layers)

    def forward(self,*x):
        x_img = self.mod_img(x[0])
        if self.with_transformer: x_clas = self.mod_clas(x[1].permute(1,0))[0]
        else: x_clas = self.mod_clas(x[1])[0]
        #EmbraceNet Fusion
        selection_prob = torch.ones(x_img.shape[0],2,device='cuda')/2.0
        indices = torch.multinomial(-torch.log(selection_prob),self.embrace_size,replacement=True)
        sample = torch.nn.functional.one_hot(indices)
        sample = sample.view(sample.shape[-1],sample.shape[0],sample.shape[1])

        x_img = self.relu(self.lin_img(x_img))
        x_clas = self.relu(self.lin_clas(x_clas))
        docking_output_stack = torch.stack([x_img,x_clas])
        embrace_output = sample*docking_output_stack
        embrace_output = embrace_output.sum(dim=0)
        x = self.layers(embrace_output)
        return x