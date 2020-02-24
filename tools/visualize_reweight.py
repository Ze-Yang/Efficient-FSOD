import matplotlib.pyplot as plt


def tensor_display(tensor):
    np_tensor = tensor.cpu().numpy()
    plt.imshow(np_tensor)
    plt.show()

