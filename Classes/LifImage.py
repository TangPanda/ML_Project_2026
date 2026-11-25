import pandas as pd
import numpy as np
import math as m
import Classes.singleSlice as singleSlice

class LifImage:
    def __init__(self, image):
        self.image = image
        self.data = None

    def zframes(self):
        data = []
        for z in range(self.image.dims.z):
            channel0_data = np.array(self.image.get_frame(z=z,t=0,c=0))
            # if z == 50: #this is to visualize zstack 50
            #     print(channel0_data)
            #     print(self.image.get_frame(z=z,t=0,c=0))

            channel1_data = np.array(self.image.get_frame(z=z,t=0,c=1))
            merged_data = np.sqrt(channel0_data**2 + channel1_data**2)
            data.append({
                "zslice": z,
                "channel_0": channel0_data,
                "channel_1": channel1_data,
                "merged_channels": merged_data
            })
        self.data = pd.DataFrame(data)
        return self.data

    def channel0(self):
        if self.data is None:
            raise ValueError("Call zframes() before channel0().")
        return self.data.channel0
    def channel1(self):
        if self.data is None:
            raise ValueError("Call zframes() before channel1().")
        return self.data.channel1
    def merged_channels(self):
        if self.data is None:
            raise ValueError("Call zframes() before merged_channels().")
        return self.data.merged_channels

    def max_intensity(self, channel):
        if self.data is None:
            raise ValueError("Call zframes() before max_intensity().")
        maximum = max(
            np.max(np.asarray(matrix))
            for matrix in self.data[channel]
        )
        return maximum

    def scaled_intensity(self, channel, scale_by):
        if self.data is None:
            raise ValueError('Call zframes() before scaled_intensity().')

        scaled_vals = np.array(self.data[channel])/scale_by
        # scaled_vals = []
        # for i in range(len(self.data[channel])):
        #     slice_obj = singleSlice(self.data[channel].loc[i])
        #
        #     # Scale intensities using the stack-wide maximum
        #     scaled_slice = slice_obj.scale_vals(
        #         index=channel,
        #         scaleBy=scale_by
        #     )
        #
        #     scaled_vals.append(scaled_slice)

        # scaled_df = pd.DataFrame(scaled_vals)
        return scaled_vals



