import torch


class Optimizer:
    def __init__(self, model):
        self.model = model
        self.velocities = self.init_velocities()

    def init_velocities(self):
        n = self.get_n_weights()
        return torch.zeros(n)

    def step_velocities(self, accelerations):
        # TODO
        return

    def get_n_weights(self):
        return self.model.fc0.weight.data.numel() + self.model.fc1.weight.data.numel()

    def get_weights(self):
        # 784 ** 2
        w0 = self.model.fc0.weight.data.flatten()
        # print(f"w0 {w0.size()}")
        # 10 * 784
        w1 = self.model.fc1.weight.data.flatten()
        # print(f"w1 {w1.size()}")

        w = torch.cat([w0, w1])
        return w

    def step(self, loss, outputs):

        out_fc0, out_fc1 = outputs

        # print(out_fc0.flatten().size())
        # print(out_fc1.flatten().size())

        w = self.get_weights()
        # self.velocities
        # print(w.size())

        # quit()
        return

