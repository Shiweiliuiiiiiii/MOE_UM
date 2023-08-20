import torch
from argparse import ArgumentParser
from models_moe import vit_moe_mlp16E4_small, vit_small

def merge_by_average(experts):
    return torch.mean(experts, dim=0)

def merge_by_summation(experts):
    return torch.sum(experts, dim=0)


def merge_moe_to_dense(dense_model, moe_state_dict, merge_fn, reset_bias):
    # TODO: Does this functions is called under no_grad()?
    dense_state_dict = dense_model.state_dict()
    for k, v in moe_state_dict.items():
        if k in dense_state_dict:
            dense_state_dict[k].data.copy_(v)
        elif "f_gate" in k:
            continue
        elif "mlp" in k:
            mapped_key = map_moe2dense(k)
            if 'bias' in mapped_key:
                if reset_bias:
                    mapped_value = torch.zeros_like(v[0])
                else:
                    mapped_value = torch.mean(v, dim=0)
            else:
                assert len(v.shape) == 3, "mlp weight shape is not 3"
                mapped_value = merge_fn(v)
                # Handling the swapping of input and output between dense and moe
                mapped_value = mapped_value.T
            dense_state_dict[mapped_key].data.copy_(mapped_value)
        else:
            print(f"key {k} not in moe state dict.. are you sure keys are mapped correctly?")

    dense_model.load_state_dict(dense_state_dict)

    return dense_model


def map_moe2dense(key):
    # output_experts maps to fc2
    # experts maps to fc1
    mapped_key = ''
    mapped_value = None
    # Checking output_experts first 
    if 'output_experts.w' in key:
        mapped_key = key.replace('output_experts.w', 'fc2.weight')
    elif 'output_experts.b' in key:
        mapped_key = key.replace('output_experts.b', 'fc2.bias')
    elif 'experts.w' in key:
        mapped_key = key.replace('experts.w', 'fc1.weight')
    elif 'experts.b' in key:
        mapped_key = key.replace('experts.b', 'fc1.bias')
    else:
        print(f"Mapping not defined for key {key}")
    return mapped_key


if __name__ == "__main__":
    # TODO: I am not sure if it is better to have a single model class that implements forward pass for both MoE and Dense. This way we can avoid copying the weights from dense class to MoE class and vice versa. 
    parser = ArgumentParser()
    parser.add_argument("--merge_type", type=str, default="average")
    parser.add_argument("--reset_bias", type=bool, default=False)
    args = parser.parse_args()
    moe_model = vit_moe_mlp16E4_small()
    print("=========== MoE Model State =============")
    print("size moe: ", sum(p.numel() for n,p in moe_model.named_parameters()))
    dense_model = vit_small()

    print("loading state_dict")
    if args.merge_type == "average":
        merge_fn = merge_by_average
    elif args.merge_type == "sum":
        merge_fn = merge_by_summation
    else:
        raise NotImplementedError("Merge type not implemented")
    # TODO: How do we deal for optimizer states?
    dense_model = merge_moe_to_dense(dense_model, moe_model.state_dict(), merge_fn, args.reset_bias)
    
    # print(list(model.state_dict().keys()))
    print("Successfully loaded state_dict")
    print("size dense: ", sum(p.numel() for n,p in dense_model.named_parameters()))