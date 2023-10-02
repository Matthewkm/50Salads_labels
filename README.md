# Convenient 50Salads Dataset Action Labels

I was unable to find coherient and conventiant labels for the [50 Salads dataset](https://cvip.computing.dundee.ac.uk/datasets/foodpreparation/50salads/), so created these segment level frame index labels to hopefully help with consistent training across works

## Usage

- Download the RGB videos from the official 50 Salads [download page](https://cvip.computing.dundee.ac.uk/datasets/foodpreparation/50salads/data/) and unzip into a file named "rgb"

- [Install and add ffmpeg to environment path](https://video.stackexchange.com/questions/20495/how-do-i-set-up-and-use-ffmpeg-in-windows) and then decode the video using <code> python decode_video.py --vidDir rgb --outDir rgb_images </code>

- The frame level labels in the Frame_Lables should match decoded RGB frame with labels in the form of: [start_idx,end_idx,class_name,class_idx] 

- Classes belonging to the 17 (+ 2 start and end background) classes, which are indexed in the Actions.txt file.

- The splits within the Splits folder match those within the [TCN implementation](https://github.com/colincsl/TemporalConvolutionalNetworks/tree/master/splits/50Salads).

- For convenience we provide a pytorch dataloader along with some simple image transformations. The data loader can be tested within the dataset.py, or via:
```python
import torchvision
from transforms import *

train_trans = torchvision.transforms.Compose([
			GroupScale(256),
			GroupRandomCrop(224),
			Stack(),
			ToTorchFormatTensor(),
			GroupNormalize(
				mean=[.485, .456, .406],
				std=[.229, .224, .225]
			)]
		)

from dataset import SaladsDataSet
train_data = SaladsDataSet(rootpath='rgb_images/',label_path='Frame_labels/',split=1,t_stride=8,num_frames=8,sampling_type='surround',mode='train',transforms=train_trans)
```

