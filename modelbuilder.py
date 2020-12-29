import torchvision.models as models
from collections import OrderedDict
from torch import nn
from torch import optim
import torch



def set_model(arch):
    if 'resnet18' == arch:
        model = models.resnet18(pretrained=True)
    if 'densenet121' == arch:
        model = models.densenet121(pretrained=True)
    else:    
        model = models.vgg16(pretrained=True)
    return model          

def build_model(in_args, device):
    
    model = set_model(in_args.arch)
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    criterion, optimizer = set_classifier(model,in_args.hidden_units,in_args.arch,in_args.learning_rate )    
    model.to(device)
    return model, criterion, optimizer 
    
def set_classifier(model,hidden_units,arch,learning_rate):
    
    if 'vgg16' in arch:
        feature_num = model.classifier[0].in_features
    elif 'densenet121' in arch:
        feature_num = model.classifier.in_features
    elif 'resnet18' in arch:
        feature_num = model.fc.in_features
        
    if hidden_units>feature_num:
        print('hidden_units cannot be bigger than input units({}) of arch {}'.format(feature_num,arch))
        exit()
    
    
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(feature_num, hidden_units)),
                              ('relu1', nn.ReLU()),
                              ('dropout1',nn.Dropout(0.5)),
                              ('fc2', nn.Linear(hidden_units, int(hidden_units/2))),
                              ('relu2', nn.ReLU()),
                              ('dropout2',nn.Dropout(0.5)),
                              ('fc3', nn.Linear( int(hidden_units/2), 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))  
        
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    #only classifer params, rest of them are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    return criterion, optimizer

def save_model(p_args, p_model, p_train_data,p_optimizer):
    p_model.class_to_idx = p_train_data.class_to_idx

    checkpoint = {'arch': p_args.arch,
                  'features': p_model.features,
                  'learning_rate': p_args.learning_rate,
                  'state_dict': p_model.state_dict(),
                  'epochs': p_args.epochs,
                  'optimizer': p_optimizer.state_dict(),
                  'classifier' : p_model.classifier,
                  'class_to_idx': p_model.class_to_idx}

    torch.save(checkpoint, p_args.save_dir +'/checkpoint_oince_part2.pth') 
    print("Model is saved."+ p_args.save_dir +'/checkpoint_oince_part2.pth') 

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath,map_location=lambda storage, loc:storage)
    model = set_model(checkpoint['arch'])
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.features = checkpoint['features']
    print('Loaded from checkpoint.')
    return model