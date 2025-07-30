import numpy as np
import torch, math, os
import pickle
import matplotlib.pyplot as plt


class DatasetAngleOnly(torch.utils.data.Dataset):
    def __init__(self, data_path, angle_dim=72, downsample_angle_factor=None, transform=None):
        self.transform = transform
        self.data = pickle.load(open(os.path.join(data_path), "rb"))
        self.angle_dim = angle_dim

        if downsample_angle_factor is not None:
            # assert (
            #    360 // downsample_angle_factor
            # ) == angle_dim, "downsample_angle_factor must be compatible with angle_dim"

            print("downsampling data...")
            d = len(self.data)
            self.data = [
                (x, y_angle, y_type) for x, y_angle, y_type in self.data if y_angle % downsample_angle_factor == 0
            ]
            print(f"downsampled data: from {d} to {len(self.data)}")

    def __len__(self):
        return len(self.data)

    @classmethod
    def pre_process(self, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img = img.transpose((2, 0, 1))
        if img.max() > 1:
            img = img / 255
        return img

    def gaussian_label(self, label, num_class, u=0, sig=4.0):
        x = np.array(range(math.floor(-num_class / 2), math.ceil(num_class / 2), 1))
        y_sig = np.exp(-((x - u) ** 2) / (2 * sig**2))
        return np.concatenate(
            [y_sig[math.ceil(num_class / 2) - label :], y_sig[: math.ceil(num_class / 2) - label]], axis=0
        )

    def __getitem__(self, i):
        x, y_angle = self.data[i]

        # img crop
        if self.transform is not None:
            x = self.transform(**{"image": x})["image"]
        x = self.pre_process(x)
        x = torch.from_numpy(x).type(torch.FloatTensor)

        # label angle
        scaled_angle = y_angle // (360 // self.angle_dim)
        y_angle = torch.tensor(self.gaussian_label(scaled_angle, self.angle_dim)).type(torch.FloatTensor)

        return x, y_angle


class DatasetCls(torch.utils.data.Dataset):
    def __init__(self, data_path, angle_dim=72, type_dim=2, downsample_angle_factor=None, transform=None):
        self.transform = transform
        self.data = pickle.load(open(os.path.join(data_path), "rb"))
        self.angle_dim = angle_dim
        self.type_dim = type_dim

        if downsample_angle_factor is not None:
            # assert (
            #    360 // downsample_angle_factor
            # ) == angle_dim, "downsample_angle_factor must be compatible with angle_dim"

            print("downsampling data...")
            d = len(self.data)
            self.data = [
                (x, y_angle, y_type) for x, y_angle, y_type in self.data if y_angle % downsample_angle_factor == 0
            ]
            print(f"downsampled data: from {d} to {len(self.data)}")

    def __len__(self):
        return len(self.data)

    @classmethod
    def pre_process(self, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img = img.transpose((2, 0, 1))
        if img.max() > 1:
            img = img / 255
        return img

    def gaussian_label(self, label, num_class, u=0, sig=4.0):
        x = np.array(range(math.floor(-num_class / 2), math.ceil(num_class / 2), 1))
        y_sig = np.exp(-((x - u) ** 2) / (2 * sig**2))
        return np.concatenate(
            [y_sig[math.ceil(num_class / 2) - label :], y_sig[: math.ceil(num_class / 2) - label]], axis=0
        )

    def __getitem__(self, i):
        x, y_angle, y_type = self.data[i]

        # img crop
        if self.transform is not None:
            x = self.transform(**{"image": x})["image"]
        x = self.pre_process(x)
        x = torch.from_numpy(x).type(torch.FloatTensor)

        # label angle
        scaled_angle = y_angle // (360 // self.angle_dim)
        y_angle = torch.tensor(self.gaussian_label(scaled_angle, self.angle_dim)).type(torch.FloatTensor)

        # label type
        y_type = torch.nn.functional.one_hot(torch.tensor(y_type), self.type_dim).type(torch.FloatTensor)

        return x, y_angle, y_type


if __name__ == "__main__":
    dataset = DatasetCls(
        data_path="data/data_merged_green_1deg_crop.pkl", angle_dim=120, type_dim=2, downsample_angle_factor=3
    )

    for i in range(len(dataset)):
        x, y_angle, y_type = dataset[i]
        print(i, x.shape)

        x = x.detach().cpu().numpy().transpose(1, 2, 0)
        angle = np.argmax(y_angle) * 3
        plt.imshow(x)
        plt.title(f"angle: {angle}, type: {y_type}")
        plt.show()
