import matplotlib.pyplot as plt

def loss_plot(train_loss, dev_loss=None):
    plt.figure(figsize=(10,5))
    plt.title("Training and Development Loss")
    # plt.plot(dev_loss_list,label="dev")
    plt.plot(train_loss,label="train")
    if dev_loss:
        plt.plot(dev_loss,label="dev")    
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()