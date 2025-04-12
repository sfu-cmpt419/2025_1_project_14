# SFU CMPT 419 Project -- Predicting Skin Lesion ABCD Values (Always Buy Canadian Donuts)

This project involves the training of a deep learning model to predict ABCD (Asymmetry, Border, Colour, and Dermoscopic Structure) values directly from an RGB image of a lesion without the use of a segmentation mask.


## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/r/personal/hamarneh_sfu_ca/Documents/TEACHING/CMPT419_SPRING2025/FOR_STUDENTS/ProjectGroup_Timesheets/Group_14_Timesheet.xlsx?d=wb4fc5fad695147f6accf7fc09ad9d10e&csf=1&web=1&e=7a8AYw) | [Slack channel](https://cmpt419spring2025.slack.com/archives/C086FBD62HJ) | [Project report](https://www.overleaf.com/5171835784jfdvtpsmkbjs#8e29b4 | [Dermoscopedia](https://dermoscopedia.org/ABCD_rule) | [ISIC 2018 Challenge](https://challenge.isic-archive.com/data/#2018)
) |
|-----------|---------------|-------------------------|




## Video/demo/GIF
Record a short video (1:40 - 2 minutes maximum) or gif or a simple screen recording or even using PowerPoint with audio or with text, showcasing your work.


## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

4. [Guidance](#guide)


<a name="demo"></a>
## 1. Example demo

A minimal example to showcase your work

```python
from amazing import amazingexample
imgs = amazingexample.demo()
for img in imgs:
    view(img)
```

### What to find where
```bash
repository
├── src                          ## source code of ABCD ground-truth collection implementation, ABCD ground-truth value results, and our model to predict ABCD values
├── .gitignore                   ## If needed, documentation   
├── README.md
├── requirements.yml             ## Python code dependencies
```

<a name="installation"></a>

## 2. Installation

Note: These steps require that conda is installed on your system.

```bash
git clone https://github.com/sfu-cmpt419/2025_1_project_14.git
cd 2025_1_project_14
conda env create -f requirements.yml
conda activate abcd-env
```

<a name="repro"></a>
## 3. Reproduction
Demonstrate how your work can be reproduced, e.g. the results in your report.
```bash
mkdir tmp && cd tmp
wget https://yourstorageisourbusiness.com/dataset.zip
unzip dataset.zip
conda activate amazing
python evaluate.py --epochs=10 --data=/in/put/dir
```

Ground-truth ABCD values can be found in src/truth_data. To generate these ground-truth values manually, the training, validation, and test datasets can be found [here](https://challenge.isic-archive.com/data/#2018). Tasks 1-2 contain the relevant data used in this project. Within source, the asymmetry, border, color, and diff_structs folder contain individual scripts to generate the respective values. The path to the relevant datasets will also need to be provided.

The script to train our model can be found in src/model/train_abcd_model.py. The following arguments are needed to run this:

```python
    csv_path = args.csv_path
    image_dir = args.image_dir
    output_model = args.output_model
    img_height = args.img_height
    img_width = args.img_width
    batch_size = args.batch_size
    epochs = args.epochs
    test_split = args.test_split
```

<a name="guide"></a>
## 4. Guidance

- Use [git](https://git-scm.com/book/en/v2)
    - Do NOT use history re-editing (rebase)
    - Commit messages should be informative:
        - No: 'this should fix it', 'bump' commit messages
        - Yes: 'Resolve invalid API call in updating X'
    - Do NOT include IDE folders (.idea), or hidden files. Update your .gitignore where needed.
    - Do NOT use the repository to upload data
- Use [VSCode](https://code.visualstudio.com/) or a similarly powerful IDE
- Use [Copilot for free](https://dev.to/twizelissa/how-to-enable-github-copilot-for-free-as-student-4kal)
- Sign up for [GitHub Education](https://education.github.com/) 
