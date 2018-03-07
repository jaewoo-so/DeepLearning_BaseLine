from matplotlib import pyplot as plt

class Plotting():
    def plot_acc(self,history, title="Accuracy"):
        if not isinstance(history, dict):
            history = history.history
        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        if title is not None:
            plt.title(title)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Verification'], loc=0)
        #plt.show()


    def plot_loss(self,history, title="Loss"):
        # summarize history for loss
        if not isinstance(history, dict):
            history = history.history
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        if title is not None:
            plt.title(title)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Verification'], loc=0)
        #plt.show()

