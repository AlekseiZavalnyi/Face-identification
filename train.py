import torch
from torch import nn
import torchvision
from torchvision.transforms import v2
from tqdm import tqdm

from data.dataset import TripletDataset


def train(model, train_dataloader, train_dataset, optimizer, scheduler, criterion, num_epochs):
    for epoch in range(1, num_epochs + 1):
        mean_loss = 0
        for idx, (a, p, n) in tqdm(enumerate(train_dataloader), desc=f'{epoch=}'):
            af = model(a)
            pf = model(p)
            nf = model(n)

            loss = criterion(af, pf, nf)
            mean_loss += loss.item()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if (idx + 1) % 10 == 0:
                train_dataset.update_model(model)
            if (idx + 1) % 100 == 0:
                train_dataset.update_negative_sample_size()
        print(f'EPOCH {epoch}. Train loss: {mean_loss / train_dataloader.batch_size}')

if __name__ == '__main__':
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 128)

    transform = v2.Compose([v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))
                            ])
    data_path = '...'
    test_path = ('test_file.txt', 'test/data/path')
    train_dataset = TripletDataset(data_path, transform, 'semi-hard-negative', model, test_path)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)

    EPOCHS = 10
    criterion = nn.TripletMarginLoss(margin=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-5, steps_per_epoch=len(train_dataloader), epochs=EPOCHS)

    train(model, train_dataloader, train_dataset, optimizer, scheduler, criterion, EPOCHS)