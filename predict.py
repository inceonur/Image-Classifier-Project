import utility
import modelbuilder
import torch
import pandas as pd

def main():
    in_arg = utility.get_input_args_predict()
    print(in_arg)
    device = utility.set_device_type(in_arg.gpu)
    model = modelbuilder.load_checkpoint(in_arg.checkpoint)
    cat_to_names=utility.load_category_map(in_arg.category_names)

    probs, classes = predict(in_arg.path_to_image, model, in_arg.top_k)
    probs = probs.cpu().numpy()
    classes = classes.cpu().numpy()
        
    class_names = get_classes_as_names(classes, model.class_to_idx,cat_to_names)
    
    data = pd.DataFrame({ 'Flower': class_names, 'Probability': probs })
    data = data.sort_values('Probability', ascending=False)  

    print("Predicting image:", in_arg.path_to_image)
    print()
    print(data)    
    
def get_classes_as_names(classes, class_to_idx, cat_to_names):
    names = {}

    for k in class_to_idx:
        names[class_to_idx[k]] = cat_to_names[k]
        
    return [names[c] for c in classes]    
    
    
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    img = utility.process_image(image_path)
    im_torch = torch.from_numpy(img)
    im_torch.unsqueeze_(0)
    im_torch.requires_grad_(False)
    
    if torch.cuda.is_available():
        im_torch.cuda()
    else:
        im_torch.cpu()
    
    model.eval()
    
    with torch.no_grad():
        output = model(im_torch)
        results = torch.exp(output).topk(topk)
    
    probs = results[0][0]
    classes = results[1][0]
    
    return probs, classes

if __name__ == "__main__":
    main()