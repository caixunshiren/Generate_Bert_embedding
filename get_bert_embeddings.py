import torch
from transformers import BertModel, BertTokenizer


def get_embeddings(text, model_name='bert-base-uncased', tokenizer_name='bert-base-uncased'):
    # import the model, tokenizer, and device
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    model = BertModel.from_pretrained(model_name, output_hidden_states=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get input ids
    ids = torch.LongTensor(tokenizer.encode(text)).unsqueeze(0)

    # setup cuda for faster computation
    model = model.to(device)
    ids = ids.to(device)

    # get hidden states
    with torch.no_grad():
        out = model(input_ids=ids)
    hidden_states = out[2]

    # get last four layers
    last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
    # cast layers to a tuple and concatenate over the last dimension
    cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
    return torch.mean(cat_hidden_states, dim=1)


def get_pca(vec, k, normalize=True, verbose = False):
    if normalize:
        if verbose:
            print("mean:", torch.mean(vec, 0, True))
            print("standard deviation:", torch.std(vec, 0, True))
        vec = (vec - torch.mean(vec, 0, True)) / torch.std(vec, 0, True)
    u, s, v = torch.pca_lowrank(vec)
    return torch.matmul(vec, v[:, :k])

'''
embed = get_embeddings("I fucking hate praxis holy shit.")
print(embed.size())
print(embed)
print(torch.max(embed))
'''
a = torch.randn(5, 10)
print(a)
print(get_pca(a, 3, normalize=True, verbose=True))

