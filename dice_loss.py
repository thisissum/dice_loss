import torch
from torch import nn
import torch.nn.functional as F


class BinaryDiceLoss(nn.Module):
    """DiceLoss implemented from 'Dice Loss for Data-imbalanced NLP Tasks'
    Useful in dealing with unbalanced data
    Add softmax automatically
    """

    def __init__(self, alpha=0.5, gamma=0.5):
        super(BinaryDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true, reduction='mean'):
        """
        :param y_pred: [N, C, ]
        :param y_true: [N, C, ]
        :param reduction: 'mean' or 'sum'
        """
        batch_size = y_true.size(0)
        y_pred = y_pred.contiguous().view(batch_size, -1)
        y_true = y_true.contiguous().view(batch_size, -1)

        numerator = torch.sum(2 * torch.pow((1 - y_pred), self.alpha) * y_pred * y_true, dim=1) + self.gamma
        denominator = torch.sum(torch.pow((1 - y_pred), self.alpha)  * y_pred + y_true, dim=1) + self.gamma
        loss = 1 - (numerator / denominator)
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss

class DiceLoss(nn.Module):
    def __init__(self, alpha=1, gamma=1):
        super(DiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.binary_dice_loss = BinaryDiceLoss(alpha, gamma)

    def forward(self, y_pred, y_true, reduction='mean'):
        """
        :param y_pred: [N, C, ]
        :param y_true: [N, ]
        :param reduction: 'mean' or 'sum'
        """
        shape = y_pred.shape
        num_labels = shape[1]
        dims = [i for i in range(len(y_pred.shape))]
        dims.insert(1, len(dims))
        y_pred = torch.softmax(y_pred, dim=1)
        y_true = F.one_hot(y_true, num_classes=num_labels).permute(*dims)
        loss = self.binary_dice_loss(y_pred, y_true, reduction)
        return loss


# =================================TEST=================================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        return self.fc2(self.fc(x))


if __name__ == '__main__':

    def test():
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        data = load_iris()
        X = data['data']
        y = data['target']
        model = Net()
        opti = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = DiceLoss(0.5,0)
        from torch.utils.data import TensorDataset, DataLoader
        tx, cvx, ty, cvy = train_test_split(X, y, test_size=0.2, random_state=1)
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

    test()