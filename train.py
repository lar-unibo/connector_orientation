import os, wandb, torch, random
from tqdm import tqdm
import albumentations as aug
import numpy as np
from torch.utils.data import DataLoader

from model import ModelMultiCls
from dataset import DatasetCls


def set_seeds(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


hyperparameter_defaults = dict(
    epochs=50,
    lr=1e-4,
    batchsize=4,
    early_stopping_patience=5,
    early_stopping_min_epochs=20,
    freq_validation_per_epoch=2,
    angle_dim=72,
    type_dim=4,
    feature_dim=256,
    downsample_angle_factor=1,
    backbone="convnext_tiny",
    seed=0,
)

wandb.init(config=hyperparameter_defaults, project="connector_pose", entity="name", mode="disabled")
config = wandb.config

SIZE = 512

# set random seed
set_seeds(config.seed)

transforms_base = aug.Compose(
    [
        aug.RandomBrightnessContrast(contrast_limit=[0, 0.2], brightness_limit=[0.0, 0.2]),
        # aug.Rotate(limit=10, p=1),
        # aug.RandomCrop(SIZE - 48, SIZE - 48, p=1),
        aug.Resize(SIZE, SIZE),
    ],
    p=1,
)

transforms_1 = aug.Compose(
    [
        aug.RandomBrightnessContrast(contrast_limit=[0, 0.2], brightness_limit=[0.0, 0.2]),
        aug.RandomCrop(SIZE - 48, SIZE - 48, p=1),
        aug.Resize(SIZE, SIZE),
    ],
    p=1,
)

transforms_2 = aug.Compose(
    [
        aug.RandomBrightnessContrast(contrast_limit=[0, 0.2], brightness_limit=[0.0, 0.2]),
        aug.Rotate(limit=15, p=1),
        aug.RandomCrop(SIZE - 48, SIZE - 48, p=1),
        aug.Resize(SIZE, SIZE),
    ],
    p=1,
)


class Trainer:
    def __init__(self, config):
        ######################################
        self.config = config

        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device ", self.device)

        self.model = ModelMultiCls(**config)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])
        self.criterion = torch.nn.BCEWithLogitsLoss()

        ######################################
        # Dataset ----------------------------
        self.train_dataset = DatasetCls(
            data_path="data/data_merged_pollock_green_yellow_cubi.pkl",
            transform=transforms_2,
            angle_dim=config["angle_dim"],
            type_dim=config["type_dim"],
            downsample_angle_factor=config["downsample_angle_factor"],
        )
        self.val_dataset = DatasetCls(
            data_path="data/data_merged_pollock2.pkl",
            transform=transforms_2,
            angle_dim=config["angle_dim"],
            type_dim=config["type_dim"],
            downsample_angle_factor=config["downsample_angle_factor"],
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=config["batchsize"], shuffle=True, num_workers=8)
        self.val_loader = DataLoader(self.val_dataset, batch_size=config["batchsize"], shuffle=True, num_workers=8)

        ######################################

    def train(self, save_last_cp=False):
        min_val_loss = np.inf
        global_step = 0

        ### TRAIN
        for epoch in range(config["epochs"]):
            self.model.train()
            train_epoch_loss, val_epoch_loss = 0, 0

            # TRAINING LOOP
            self.model.train()
            for img, label_angle, label_type in tqdm(self.train_loader):
                self.optimizer.zero_grad()

                img = img.to(self.device)
                label_angle = label_angle.to(self.device)
                label_type = label_type.to(self.device)

                pred_angle, pred_type = self.model(img)

                loss_angle = self.criterion(pred_angle.squeeze(), label_angle)
                loss_type = self.criterion(pred_type.squeeze(), label_type)
                loss = loss_angle + loss_type

                ##########
                train_epoch_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "train_loss_angle": loss_angle.item(),
                        "train_loss_type": loss_type.item(),
                    },
                    step=global_step,
                )
                global_step += 1

            # VALIDATION LOOP
            self.model.eval()
            for img, label_angle, label_type in tqdm(self.val_loader):
                img = img.to(self.device)
                label_angle = label_angle.to(self.device)
                label_type = label_type.to(self.device)

                pred_angle, pred_type = self.model(img)

                loss_angle = self.criterion(pred_angle.squeeze(), label_angle)
                loss_type = self.criterion(pred_type.squeeze(), label_type)
                loss = loss_angle + loss_type

                val_epoch_loss += loss.item()

                wandb.log(
                    {
                        "val_loss": loss.item(),
                        "val_loss_angle": loss_angle.item(),
                        "val_loss_type": loss_type.item(),
                    },
                    step=global_step,
                )

            # LOG EPOCH
            print("Epoch: ", epoch + 1)
            print("Train loss: ", train_epoch_loss / len(self.train_loader))
            print("Val loss: ", val_epoch_loss / len(self.val_loader))

            # SAVE CHECKPOINT
            state = dict(self.config).copy()
            state["epoch"] = epoch + 1
            state["step"] = global_step
            state["model_state_dict"] = self.model.state_dict()

            if val_epoch_loss < min_val_loss:
                min_val_loss = val_epoch_loss
                torch.save(state, os.path.join(self.checkpoint_dir, "CP_BEST_{}.pth".format(wandb.run.name)))
                print(f"Best Checkpoint Saved!")

            if save_last_cp:
                torch.save(state, os.path.join(self.checkpoint_dir, "CP_LAST_{}.pth".format(wandb.run.name)))
                print(f"Checkpoint {epoch + 1} saved !")


if __name__ == "__main__":
    print("Starting training:")
    for k, v in config.items():
        print(f"\t{k}:   {v}")
    print("")

    trainer = Trainer(config)
    trainer.train()
