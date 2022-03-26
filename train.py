import pickle
from dataset import ImageDataset
import math
import torch
from torch import nn
from model import ResNetModel
from tqdm import tqdm
from torch.utils.data import DataLoader
from transforms import img_transform, target_transform

TRAIN_IMG_DIR = "data/train/images"
TRAIN_TARGET_DIR = "data/train/targets"
VALIDATION_IMG_DIR = "data/validation/images"
VALIDATION_TARGET_DIR = "data/validation/targets"
TEST_IMG_DIR = "data/test/images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 2e-4
BATCH_SIZE = 64
OPTIM = "sgd"
NUM_EPOCHS = 200

fname = "2e4_adamw_resnet"


def get_dataloader(img_dir, target_dir, img_transform, target_transform, mode):
    "Returns the dataloader after creating a Dataset"
    dataset = ImageDataset(
        TRAIN_IMG_DIR, TRAIN_TARGET_DIR, img_transform, target_transform, mode
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    return dataloader


train_dataloader = get_dataloader(
    TRAIN_IMG_DIR, TRAIN_TARGET_DIR, img_transform, target_transform, "train"
)
validation_dataloader = get_dataloader(
    VALIDATION_IMG_DIR,
    VALIDATION_TARGET_DIR,
    img_transform,
    target_transform,
    "validation",
)

model = ResNetModel()
model.to(DEVICE)
loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
num_training_steps = len(train_dataloader) // BATCH_SIZE
num_warmup_steps = int(num_training_steps * 0.7)
num_cycles = 0.5
num_epochs = 452


def lr_lambda(current_step):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return max(
        0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    )


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)


def train_one_epoch(model, dataloader, loss_fn):
    "Runs training of the model for one epoch (on the entire dataset)"
    model.train()
    running_loss = 0.0
    last_loss = 0.0
    batch_cnt = 0
    for data in tqdm(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
    return model, optimizer, running_loss / 10


@torch.no_grad()
def evaluate(model, dataloader):
    "Method to run evaluation on the desired dataset"
    model.eval()
    score, batch_cnt = 0, 0
    for data in tqdm(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        output = model(inputs)
        score += abs(sum(output - labels))
        batch_cnt += 1
    return score / batch_cnt


def train(num_epochs, model, train_dataloader, validation_dataloader, loss_fn):
    """
    The main training method. It will run the training along with evaluation on the validation set.
    Input:
        num_epochs : number of epochs
        model: nn.Module() model
        train_dataloader: training dataset wrapped inside DataLoader
        validation_dataloader: validation dataset wrapped inside DataLoader
        loss_fn: Loss function (metric)
    """
    all_loss_vals, all_lrs, all_accs = [], [], []
    for epoch in range(num_epochs):
        model, optimizer, loss_val = train_one_epoch(model, train_dataloader, loss_fn)
        all_loss_vals.append(loss_val)
        all_lrs.append(optimizer.param_groups[0]["lr"])
        print(" Epoch {} Loss: {}".format(epoch + 1, loss_val))
        acc = evaluate(model, validation_dataloader).item()
        all_accs.append(acc)
        print(" Epoch {} Accuracy: {}".format(epoch + 1, acc))

    with open(fname + "_loss_vals.txt", "wb") as fp:
        pickle.dump(all_loss_vals, fp)
    with open(fname + "_all_lrs.txt", "wb") as fp:
        pickle.dump(all_lrs, fp)
    with open(fname + "_all_acs.txt", "wb") as fp:
        pickle.dump(all_accs, fp)


train(NUM_EPOCHS, model, train_dataloader, validation_dataloader, loss_fn)
