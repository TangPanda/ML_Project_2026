import pandas as pd
import numpy as np
import math as m


class LifImage:
    def __init__(self, image):
        self.image = image
        self.data = None

    def zframes(self):
        data = []
        for z in range(self.image.dims.z):
            channel0_data = np.array(self.image.get_frame(z=z,t=0,c=0))
            if z == 50:
                print(channel0_data)
                print(self.image.get_frame(z=z,t=0,c=0))

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

    # def to_grayscale(self):
    #     if self.scaled_vals is None:
    #         raise ValueError("call zframes first.")
    #
    #     gray_ch0 = []
    #     gray_ch1 = []
    #     gray_merged = []
    #
    #     for z in range(len(self.scaled_vals)):
    #         channel0_gray = self.scaled_vals.loc[z, "channel_0"].astype(float)
    #         gray_ch0.append(channel0_gray)
    #
    #         channel1_gray = self.scaled_vals.loc[z, "channel_1"].astype(float)
    #         gray_ch1.append(channel1_gray)
    #
    #         merged_gray = (channel0_gray + channel1_gray)/2
    #         gray_merged.append(merged_gray)
    #
    #     self.scaled_vals["gray_ch0"] = gray_ch0
    #     self.scaled_vals["gray_ch1"] = gray_ch1
    #     self.scaled_vals["gray_merged"] = gray_merged
    #
    #     return self.scaled_vals



