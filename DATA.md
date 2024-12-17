# Dataset Preparation

We provide our labels in the `labels` directory.

## HMDB51

1. We downloaded the offical version of HMDB51 dataset from the dataset provider [SERRE LAB](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads) and videos are placed in the data directory.

2. Once the dataset has been fully downloaded, generate the CSV files for training and validation using `csv_writer.py`, as `train.csv` and `val.csv`. The required format for the CSV files is as follows:

```
<path_1>,<label_1>
<path_2>,<label_2>
...
<path_n>,<label_n>
```
where `<path_i>` points to a video file, and `<label_i>` is an integer between `0` and `num_classes - 1`.

|       Dataset      | Number of Classes |  Total Number of Videos   | Train | Validation | Average Video Duration | Resoultion |
|:----------------:|:----------:|:----------:|:---:|:-------:|:-------:|:-------:|
| HMDB51 |    51   | 6766 | 5441 |  1335  | 1 ~ 10 sec | 320 × 240 |


## UCF50

1. We downloaded the offical version of UCF50 dataset from the dataset provider [UCF Center for Research in Computer Vision](https://www.crcv.ucf.edu/data/UCF50.php) and videos are placed in the data directory.

2. Once the dataset has been fully downloaded, generate the CSV files for training and validation using `csv_writer.py`, as `train.csv` and `val.csv`. The required format for the CSV files is as follows:

```
<path_1>,<label_1>
<path_2>,<label_2>
...
<path_n>,<label_n>
```
where `<path_i>` points to a video file, and `<label_i>` is an integer between `0` and `num_classes - 1`.

|       Dataset      | Number of Classes |  Total Number of Videos   | Train | Validation | Average Video Duration | Resoultion |
|:----------------:|:----------:|:----------:|:---:|:-------:|:-------:|:-------:|
| UCF50 |    50   | 6681 | 5360 |  1321  | 1 ~ 10 sec | 320 × 240 |

## Diving-48

1. Please download the dataset and annotations from the [dataset provider](http://www.svcl.ucsd.edu/projects/resound/dataset.html). Note that we use the V2 splits.

2. Set up the training and validation CSV files as above for Kinetics.

## ActivityNet-v1.3

1. Please request access for the full [v1.3](http://activity-net.org/download.html) dataset [here](https://docs.google.com/forms/d/e/1FAIpQLSeKaFq9ZfcmZ7W0B0PbEhfbTHY41GeEgwsa7WobJgGUhn4DTQ/viewform). Download the entire dataset including all missing videos.

2. Download the annotations JSON file [here](http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/files/activity_net.v1-3.min.json)

3. Set up the training and validation CSV files as above for Kinetics.
