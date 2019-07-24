import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        # Embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(input_size = self.embed_size, 
                            hidden_size = self.hidden_size, 
                            num_layers = self.num_layers, 
                            batch_first=True,
                            dropout=0, # applying dropout only if num_layers >1
                            bidirectional=False, # unidirectional LSTM
                            )
        self.fc = nn.Linear(hidden_size, vocab_size) 
        
        
        
    def init_hidden(self, batch_size):
        '''
        Initialize a hidden state;
        Based on previously seen data.
        The dims: (num_layers, batch_size, hidden_size)
        '''
        #TODO: not working right now doesnt initialize at all from this function
        
        print('in init_hiden')
        return (torch.zeros((1, batch_size, self.hidden_size), device=device), 
                torch.zeros((1, batch_size, self.hidden_size), device=device))
    
    
    
    def forward(self, features, captions):
        #print('features_shape =', features.shape)
        #print('captions_shape =', captions.shape)
        batch_size = features.shape[0]
        #self.hidden_w = init_hidden(self, batch_size)
        self.hidden_w = (torch.zeros((1, batch_size, self.hidden_size), device=device), 
                         torch.zeros((1, batch_size, self.hidden_size), device=device))    
        #print('hidden_w =', self.hidden_w.shape)
        captions = captions[:,:-1] # remove end
        embeddings = self.word_embeddings(captions)
        #print('embeddings =', embeddings.shape)
        # Stack the features and captions
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1) # embeddings new shape : (batch_size, caption length, embed_size)
        #print('embeddings =', embeddings.shape)
        # Get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hidden state
        x, hc = self.lstm(embeddings, self.hidden_w) # lstm_out shape : (batch_size, caption length, hidden_size)
        #print('x_shape =', x.shape)
        #import numpy as np
        #print('hc_shape =', np.ndim(hc))
        # Fully connected layer
        x = self.fc(x)
        return x 
    
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # design wise same as forward
        features = inputs
        batch_size = features.shape[0]
        #print('batch size in inference (should be 1)', batch_size)
        self.hidden_w = (torch.zeros((1, batch_size, self.hidden_size), device=device), 
                         torch.zeros((1, batch_size, self.hidden_size), device=device))
        out = []
        while True:
            x, self.hidden_w = self.lstm(features, self.hidden_w) # x shape : (1, 1, hidden_size)
            #import numpy as np
            #print('x_out =', np.ndim(x))
            #print('hc =', np.ndim(hc))
            #print('x_out.shape, self.hidden_w.shape)
            x = self.fc(x)  # x shape : (1, 1, vocab_size)
            x = x.squeeze(1) # x shape : (1, vocab_size)
            _, max_indice = torch.max(x, dim=1) # predict the most likely next word, max_indice shape : (1)
            
            out.append(max_indice.cpu().numpy()[0].item()) # storing the word predicted
            # if next word is 'end' then sop here.
            if (max_indice == 1):
                #print('out = ', out)
                break
            # else continue
            ## Prepare to embed the last predicted word to be the new input of the lstm
            features = self.word_embeddings(max_indice) # features shape : (1, embed_size)
            features = features.unsqueeze(1) # features shape : (1, 1, embed_size)
        
        #x = forward(inputs, captions)
        
        return out