import sys, torch, pickle, cv2
import numpy as np
from model import ModelMultiCls
from dataset import DatasetCls
import matplotlib.pyplot as plt
import albumentations as aug

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = sys.argv[1]
state = torch.load(checkpoint_path, map_location=torch.device("cpu"))

model = ModelMultiCls(**state)
model.load_state_dict(state["model_state_dict"])
model.to(device)
model.eval()

print("model loaded")
print("angle_dim", state["angle_dim"])
print("type_dim", state["type_dim"])
print("downsample_angle_factor", state["downsample_angle_factor"])


transforms_base = aug.Compose(
    [
        # aug.Compose(
        #    [
        #        # aug.HueSaturationValue(hue_shift_limit=100, sat_shift_limit=100, val_shift_limit=100, p=0.8),
        #        # aug.ChannelShuffle(p=0.8),
        #    ],
        #    p=0.5,
        # ),
        # aug.RandomBrightnessContrast(contrast_limit=[0, 0.1], brightness_limit=[0.0, 0.1]),
        # aug.RandomCrop(512 - 128, 512 - 128, p=1),
        aug.Resize(512, 512),
    ],
    p=1,
)

input("Press enter to start...")

dataset_pickle = "data/data_test.pkl"

dataset = DatasetCls(
    data_path=dataset_pickle,
    transform=transforms_base,
    angle_dim=state["angle_dim"],
    type_dim=state["type_dim"],
    downsample_angle_factor=state["downsample_angle_factor"],
)

print("dataset size: ", len(dataset))

preds = {i: [] for i in range(state["type_dim"])}
for i in range(0, len(dataset)):
    print(i)
    img, label_angle, label_type = dataset[i]

    img = img.to(device).unsqueeze(0)
    pred_angle, pred_type = model(img)

    pred_angle = pred_angle.softmax(-1)
    pred_angle = pred_angle / torch.max(pred_angle)

    pred_type = pred_type.softmax(-1)
    pred_type = pred_type / torch.max(pred_type)

    ############################
    pred_angle = pred_angle.squeeze(0).detach().cpu().numpy()
    pred_type = pred_type.squeeze(0).detach().cpu().numpy()
    img = img.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    label_angle = label_angle.detach().cpu().numpy()
    label_type = label_type.detach().cpu().numpy()

    # angle = np.argmax(pred_angle) * 5
    # plt.imshow(img)
    # plt.title(f"angle: {angle}, type: {label_type}")
    # plt.show()

    label_type_value = np.argmax(label_type)
    label_angle_value = np.argmax(label_angle) * 5  # state["downsample_angle_factor"]
    pred_type_value = np.argmax(pred_type)
    pred_angle_value = np.argmax(pred_angle) * 5  # state["downsample_angle_factor"]  ##############

    preds[label_type_value].append(
        {"pred_type": pred_type, "label_type": label_type, "pred_angle": pred_angle, "label_angle": label_angle}
    )

    print("pred_type", pred_type_value, "label_type", label_type_value)
    print("pred_angle", pred_angle_value, "label_angle", label_angle_value)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("image")
    axs[1].plot(np.arange(len(pred_angle)), pred_angle, label="pred")
    axs[1].plot(np.arange(len(pred_angle)), label_angle, "o-", label="label")
    axs[1].set_title("angle")
    axs[1].legend()
    axs[2].bar(np.arange(len(pred_type)) - 0.2, pred_type, label="pred", width=0.3)
    axs[2].bar(np.arange(len(pred_type)) + 0.2, label_type, label="label", width=0.3)
    axs[2].set_xticks(np.arange(len(pred_type)))
    axs[2].set_title("type")
    axs[2].legend()
    plt.tight_layout()
    plt.show()
