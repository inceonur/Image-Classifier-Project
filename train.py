import utility
import modelbuilder
import torch

def main():
    in_arg = utility.get_input_args()
    print(in_arg)
    device = utility.set_device_type(in_arg.gpu)
    image_datasets, dataloaders=utility.load_data(in_arg.data_dir)
    cat_to_name = utility.load_category_map()
    model, criterion, optimizer = modelbuilder.build_model(in_arg, device)
    
    train_model(model,dataloaders['train'],dataloaders['valid'],in_arg.epochs,40, criterion, optimizer, device)

    modelbuilder.save_model(in_arg, model, image_datasets['train'], optimizer)
    
def train_model(p_model, p_trainloader,p_validloader, p_epochs, p_print_every, p_criterion, p_optimizer,p_device):
    
    steps = 0
    p_model.train()

    for e in range(p_epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(p_trainloader):
            steps += 1

            inputs, labels = inputs.to(p_device), labels.to(p_device)

            p_optimizer.zero_grad()

            # Forward and backward passes
            outputs = p_model.forward(inputs)
            loss = p_criterion(outputs, labels)
            loss.backward()
            p_optimizer.step()

            running_loss += loss.item()

            if steps % p_print_every == 0:

                p_model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(p_model, p_validloader, p_criterion,p_device)

                print("Epoch: {}/{}.. ".format(e+1, p_epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/p_print_every),
                      "Valid Loss: {:.3f}.. ".format(valid_loss/len(p_validloader)),
                      "Valid Accuracy: {:.3f}".format(accuracy/len(p_validloader)))

                running_loss = 0

                # back to training
                p_model.train()
    print("Training is done")   
    
def validation(p_model, p_validloader, p_criterion,p_device):
    valid_loss = 0
    accuracy = 0
    for images, labels in p_validloader:
    
        images, labels = images.to(p_device), labels.to(p_device)
        output = p_model.forward(images)
        valid_loss += p_criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy    



if __name__ == "__main__":
    main()
                      