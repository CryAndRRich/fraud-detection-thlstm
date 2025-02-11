import torch

def save_checkpoint(model, optimizers, epoch, losses):
    checkpoint = {
        'epoch' : epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dicts' : [opt.state_dict() for opt in optimizers],
        'losses' : {key: loss.ite() for key, loss in losses.items()},
    }
    filepath = "saved_models/" + model.name + ".pth"
    torch.save(checkpoint, filepath)

def load_checkpoint(model, optimizer = None):
    filepath = "saved_models/" + model.name + ".pth"
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss