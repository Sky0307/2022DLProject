# 2022DLProject

2022 Deep Learning Project

# Instructions

Your Github repository for this project should contain your PDF report, your DOCUMENTED code/notebook files. It should also contain directions showing the required libraries and steps needed to re-train the model from scratch. And more importantly, it should also contain clear directions on how to recreate the exact trained model and its performance results you are presenting in the PDF, by loading some save PyTorch weights for your model. This is essential, for reproducibility reasons and something that we, researchers, must do constantly for our papers. If the weights files are too heavy to be submitted on edimension or Github, you may upload them to a Google Drive/Dropbox instead.

- [ ] upload report
- [ ] upload latest code(building GTZAN dataset and custom dataset)

## Dataset

Can be downloaded [here](https://drive.google.com/file/d/1zUmZGB0pKuOuHccWjySoyKTCr0BUeUPs/view?usp=sharing)

## Trained Model

Can be found [here](./model/CnnModel.pt)

### To Train the model

Download the dataset, unzip and upload to your google drive, then save it in a folder named **datasets**

In **datasets** folder, create 2 new folders **model** and **log** to store the result of your training

Then run the `training.ipynb`

### To Load the model


```python
from model import CnnModel

DIRECTORY = "YOUR_DIRECTORY_TO_THE FILE	"
model = CnnModel().cuda()
model.load_state_dict(torch.load(DIRECTORY + "CnnModel.pt", map_location='cpu'))
model.eval()

```

## Test on Full-Length Song:

1. download the sample music from [here](https://drive.google.com/file/d/1hvHtrnRMKQ-OSNM_Jgcbi8VZe98V0cRK/view?usp=sharing)
2. unzip the file
3. upload to the same folder, **datasets**
4. You may upload your own songs to test which genre it belongs to

Sample result:
![sample result](./pictures/individual%20song%20test.jpg)