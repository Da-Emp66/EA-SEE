import argparse
import os
import warnings
import torch
import torch.nn as nn
import torchfile
from typing import Union
from ea_see.model import FaceEmbeddingModel

def resolve_layer_index(model: Union[nn.Module, FaceEmbeddingModel], relative_index: int, layer_type: type = nn.Conv2d):
    counter = 0

    for index, layer in enumerate(model.model.modules()):

        if counter == relative_index and type(layer) == layer_type:
            return (index - 1, layer)
        
        if type(layer) == layer_type:
            counter += 1
        
    return (-1, None)

def convert_t7_vgg_face_embedding_model_to_pt(t7_filepath: os.PathLike, pt_filepath: os.PathLike):
    torch_model = torchfile.load(t7_filepath)
    embedding_model = FaceEmbeddingModel(device="cpu")

    index = 0

    for layer in torch_model.modules:
        if layer.weight is not None:
            embedding_conv_layer_index, _ = resolve_layer_index(embedding_model, index)
            if embedding_conv_layer_index == -1:
                break

            layer_to_modify = embedding_model.model.pop(embedding_conv_layer_index)
            print(f"Loading parameters into layer ({embedding_conv_layer_index}, {layer_to_modify})")
            layer_to_modify.weight.data[...] = torch.tensor(layer.weight).view_as(layer_to_modify.weight)[...]
            layer_to_modify.bias.data[...] = torch.tensor(layer.bias).view_as(layer_to_modify.bias)[...]
            embedding_model.model.insert(embedding_conv_layer_index, layer_to_modify)
            
            index += 1
            
    torch.save(embedding_model.state_dict(), pt_filepath)

def main(args):
    if args.t7_filepath.endswith(".t7") and args.pt_filepath.endswith(".pt"):
        convert_t7_vgg_face_embedding_model_to_pt(args.t7_filepath, args.pt_filepath)
    elif not args.t7_filepath.endswith(".t7"):
        warnings.warn(f"File {args.t7_filepath} not converted. File is not of `.t7` type.")
    elif not args.pt_filepath.endswith(".pt"):
        warnings.warn(f"File {args.pt_filepath} is not of `.pt` type. Doing nothing.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t7", "--t7_filepath", type=str)
    parser.add_argument("-pt", "--pt_filepath", type=str)
    args = parser.parse_args()
    main(args)
