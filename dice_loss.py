import torch 
from torch import nn

class DiceLoss(nn.Module):
    """DiceLoss implemented from 'Dice Loss for Data-imbalanced NLP Tasks'
    Useful in dealing with unbalanced data
    Add softmax automatically
    """

    def __init__(self):
        super(DiceLoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        # shape(y_pred) = batch_size, label_num, **
        # shape(y_true) = batch_size, **
        y_pred = torch.softmax(y_pred, dim=1)
        pred_prob = torch.gather(y_pred, dim=1, index=y_true.unsqueeze(1))
        dsc_i = 1 - ((1-pred_prob)*pred_prob) / ((1-pred_prob) * pred_prob + 1)
        dice_loss = dsc_i.mean()
        return dice_loss

#=================================TEST=================================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(4,32)
        self.fc2 = nn.Linear(32,3)
    def forward(self, x):
        return self.fc2(self.fc(x))

def test():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    data = load_iris()
    X = data['data']
    y = data['target']
    model = Net()
    opti = torch.optim.Adam(model.parameters())
    criterion = DiceLoss()
    from torch.utils.data import TensorDataset, DataLoader
    tx, cvx, ty, cvy = train_test_split(X, y, test_size=0.2)
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(tx), torch.LongTensor(ty)), batch_size=32)
    dev_loader = DataLoader(TensorDataset(torch.FloatTensor(cvx), torch.LongTensor(cvy)), batch_size=32)
    for i in range(100):
        for train_x, train_y in train_loader:
            output = model(train_x)
            loss = criterion(output, train_y)
            opti.zero_grad()
            loss.backward()
            opti.step()
        for dev_x, dev_y in train_loader:
            output = model(dev_x)
            true_num = (output.argmax(dim=1) == dev_y).sum()
        print(true_num)