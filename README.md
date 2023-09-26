# Convenient 50Salads Dataset Action Labels

I was unable to find coherient and conventiant labels for the [50 Salads dataset](https://cvip.computing.dundee.ac.uk/datasets/foodpreparation/50salads/), so created these frame level segment labels to hopefully help with consistent training across works

## Usage

Download the RGB videos from the official 50 Salads [download page](https://cvip.computing.dundee.ac.uk/datasets/foodpreparation/50salads/data/) and decode the video (to be added).

The frame level labels in the Frame_Lables should match decoded RGB frame with labels in the form of <code>[start_idx,end_idx,class_name,class_idx]</code> 

Classes belonging to the 17 (+ 2 start and end background) classes, which are indexed in the Actions.txt file.

The splits within the Splits folder match those within the [TCN implementation](https://github.com/colincsl/TemporalConvolutionalNetworks/tree/master/splits/50Salads).

For convenience we provide a pytorch dataloader which operates on our frame labels. (to be added)
