import pandas as pd
import torch
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import sys
TRAIN_PATH = sys.argv[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMDataset(Dataset):
    def __init__(self, path=TRAIN_PATH, pad_to=50, pad=True, pad_value=0, features=None):
        self.data = []
        self.targets = []
        for file in os.listdir(path):
            df = pd.read_csv(os.path.join(path, file), sep='|')
            sepsis_index = df[df['SepsisLabel'] == 1].first_valid_index()
            df = df.drop('SepsisLabel', axis=1)
            if features is not None:
                df = df[features]
            if sepsis_index is not None:
                df = df.iloc[:sepsis_index+1]
                self.targets.append(1)
            else:
                self.targets.append(0)
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            data = torch.tensor(df.values, dtype=torch.float32)
            self.data.append(data)
            if pad:
                if len(self.data[-1]) < pad_to:
                    self.data[-1] = nn.functional.pad(self.data[-1], (0, 0, 0, pad_to - self.data[-1].shape[0]), value=pad_value)
                else:
                    self.data[-1] = self.data[-1][-pad_to:]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
class LSTMModel(nn.Module):
    def __init__(self, embedding_size=40, in_channels=400, out_channels=40, hidden_size=128, num_layers=2, lstm_dropout=0.1, fc_dropout=0.1, bidirectional=True, conv_dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.maxpool = nn.MaxPool1d(4)
        self.conv = nn.Sequential(nn.Conv1d(in_channels, out_channels, 3, padding=1), self.maxpool, nn.ReLU(), nn.Dropout(conv_dropout))
        self.lstm = nn.LSTM(int(embedding_size/4), hidden_size, num_layers, dropout=lstm_dropout, batch_first=True, bidirectional=bidirectional)
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(out_channels*(int(2*hidden_size) if bidirectional else int(hidden_size)), 2)

    def forward(self, x):
        out = self.conv(x)
        out, _ = self.lstm(out)
        out = self.fc_dropout(out)
        out = self.fc(out.reshape(x.shape[0], -1))
        return out









def main():
    df= pd.DataFrame()
    
    data =LSTMDataset(pad_to=400)
    dataloader = DataLoader(data, batch_size=32, shuffle=False)
    
    df["id"]= list(map(lambda x: x.split(".")[0],os.listdir(TRAIN_PATH)))
    model = torch.load('best_rnn_model.pt',map_location=device)
    model.eval()
    real = []
    pred = []
    with torch.no_grad(): 
        for input, _ in dataloader:
            input = input.to(device)
            out = model(input)
            pred.append(torch.argmax(out, dim=1))
        df["prediction"]= torch.cat(pred).cpu().numpy()
        df = df.sort_values(by=['id'], key=lambda x: x.str.split('_').str[1].astype(int))
        df.to_csv("prediction.csv")      
if __name__ == "__main__":
    main()