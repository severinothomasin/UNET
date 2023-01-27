import torch as T
import torch.nn as T_nn
import torch.nn.functional as T_nn_functional
from torchsummary import summary as TS_summary

class DynamicUNet(T_nn.Module):

    def __init__(self, filters, input_channels=1, output_channels=1):
        """ Constructor for UNet class.
        Parameters:
            filters(list): Five filter values for the network.
            input_channels(int): Input channels for the network. Default: 1
            output_channels(int): Output channels for the final network. Default: 1
        """
        super(DynamicUNet, self).__init__()

        if len(filters) != 5:
            raise Exception(f"Filter list size {len(filters)}, expected 5!")

        padding = 1
        ks = 3
        # Encoding Part of Network.
        #   Block 1
        self.conv1_1 = T_nn.Conv2d(input_channels, filters[0], kernel_size=ks, padding=padding)
        self.conv1_2 = T_nn.Conv2d(filters[0], filters[0], kernel_size=ks, padding=padding)
        self.maxpool1 = T_nn.MaxPool2d(2)
        #   Block 2
        self.conv2_1 = T_nn.Conv2d(filters[0], filters[1], kernel_size=ks, padding=padding)
        self.conv2_2 = T_nn.Conv2d(filters[1], filters[1], kernel_size=ks, padding=padding)
        self.maxpool2 = T_nn.MaxPool2d(2)
        #   Block 3
        self.conv3_1 = T_nn.Conv2d(filters[1], filters[2], kernel_size=ks, padding=padding)
        self.conv3_2 = T_nn.Conv2d(filters[2], filters[2], kernel_size=ks, padding=padding)
        self.maxpool3 = T_nn.MaxPool2d(2)
        #   Block 4
        self.conv4_1 = T_nn.Conv2d(filters[2], filters[3], kernel_size=ks, padding=padding)
        self.conv4_2 = T_nn.Conv2d(filters[3], filters[3], kernel_size=ks, padding=padding)
        self.maxpool4 = T_nn.MaxPool2d(2)
        
        # Bottleneck Part of Network.
        self.conv5_1 = T_nn.Conv2d(filters[3], filters[4], kernel_size=ks, padding=padding)
        self.conv5_2 = T_nn.Conv2d(filters[4], filters[4], kernel_size=ks, padding=padding)
        self.conv5_t = T_nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)

        # Decoding Part of Network.
        #   Block 4
        self.conv6_1 = T_nn.Conv2d(filters[4], filters[3], kernel_size=ks, padding=padding)
        self.conv6_2 = T_nn.Conv2d(filters[3], filters[3], kernel_size=ks, padding=padding)
        self.conv6_t = T_nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        #   Block 3
        self.conv7_1 = T_nn.Conv2d(filters[3], filters[2], kernel_size=ks, padding=padding)
        self.conv7_2 = T_nn.Conv2d(filters[2], filters[2], kernel_size=ks, padding=padding)
        self.conv7_t = T_nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        #   Block 2
        self.conv8_1 = T_nn.Conv2d(filters[2], filters[1], kernel_size=ks, padding=padding)
        self.conv8_2 = T_nn.Conv2d(filters[1], filters[1], kernel_size=ks, padding=padding)
        self.conv8_t = T_nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        #   Block 1
        self.conv9_1 = T_nn.Conv2d(filters[1], filters[0], kernel_size=ks, padding=padding)
        self.conv9_2 = T_nn.Conv2d(filters[0], filters[0], kernel_size=ks, padding=padding)

        # Output Part of Network.
        self.conv10 = T_nn.Conv2d(filters[0], output_channels, kernel_size=ks, padding=padding)

    def forward(self, x):
        """ Method for forward propagation in the network.
        Parameters:
            x(torch.Tensor): Input for the network of size (1, 512, 512).
        Returns:
            output(torch.Tensor): Output after the forward propagation 
                                    of network on the input.
        """

        # Encoding Part of Network.
        #   Block 1
        conv1 = T_nn_functional.relu(self.conv1_1(x))
        conv1 = T_nn_functional.relu(self.conv1_2(conv1))
        pool1 = self.maxpool1(conv1)
        #   Block 2
        conv2 = T_nn_functional.relu(self.conv2_1(pool1))
        conv2 = T_nn_functional.relu(self.conv2_2(conv2))
        pool2 = self.maxpool2(conv2)
        #   Block 3
        conv3 = T_nn_functional.relu(self.conv3_1(pool2))
        conv3 = T_nn_functional.relu(self.conv3_2(conv3))
        pool3 = self.maxpool3(conv3)
        #   Block 4
        conv4 = T_nn_functional.relu(self.conv4_1(pool3))
        conv4 = T_nn_functional.relu(self.conv4_2(conv4))
        pool4 = self.maxpool4(conv4)

        # Bottleneck Part of Network.
        conv5 = T_nn_functional.relu(self.conv5_1(pool4))
        conv5 = T_nn_functional.relu(self.conv5_2(conv5))

        # Decoding Part of Network.
        #   Block 4
        up6 = T.cat((self.conv5_t(conv5), conv4), dim=1)
        conv6 = T_nn_functional.relu(self.conv6_1(up6))
        conv6 = T_nn_functional.relu(self.conv6_2(conv6))
        #   Block 3
        up7 = T.cat((self.conv6_t(conv6), conv3), dim=1)
        conv7 = T_nn_functional.relu(self.conv7_1(up7))
        conv7 = T_nn_functional.relu(self.conv7_2(conv7))
        #   Block 2
        up8 = T.cat((self.conv7_t(conv7), conv2), dim=1)
        conv8 = T_nn_functional.relu(self.conv8_1(up8))
        conv8 = T_nn_functional.relu(self.conv8_2(conv8))
        #   Block 1
        up9 = T.cat((self.conv8_t(conv8), conv1), dim=1)
        conv9 = T_nn_functional.relu(self.conv9_1(up9))
        conv9 = T_nn_functional.relu(self.conv9_2(conv9))

        # Output Part of Network.
        output = T.sigmoid(self.conv10(conv9))

        return output

    def summary(self, input_size=(1, 512, 512), batch_size=-1, device='cuda'):
        """ Get the summary of the network in a chart like form
        with name of layer size of the inputs and parameters 
        and some extra memory details.
        This method uses the torchsummary package.
        For more information check the link.
        Link :- https://github.com/sksq96/pytorch-summary
        Parameters:
            input_size(tuple): Size of the input for the network in
                                 format (Channel, Width, Height).
                                 Default: (1,512,512)
            batch_size(int): Batch size for the network.
                                Default: -1
            device(str): Device on which the network is loaded.
                            Device can be 'cuda' or 'cpu'.
                            Default: 'cuda'
        Returns:
            A printed output for IPython Notebooks.
            Table with 3 columns for Layer Name, Input Size and Parameters.
            torchsummary.summary() method is used.
        """
        return TS_summary(self, input_size, batch_size, device)
