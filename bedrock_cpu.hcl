version = "1.0"

train {
    step get_dataset {
        image = "python:3.7"
        install = ["pip3 install -r dataset/requirements.txt"]
        script = [
            {
                sh = ["python3 dataset/download.py"]
            }
        ]

        resources {
            cpu = "0.5"
            memory = "1G"
        }
    }

    step train1 {
        image = "python:3.7"
        install = ["pip3 install -r src/requirements.txt"]
        script = [
            {
                sh = ["python3 dataset/download.py",
                      "python3 src/train_stage1.py"
                     ]
            }
        ]

        resources {
            cpu = "8"
            memory = "8G"
        }
        depends_on = ["get_dataset"]
    }

    step train2 {
        image = "nvidia/cuda:10.0-base"
        install = ["pip3 install -r src/requirements.txt"]
        script = [
            {
                sh = ["python3 dataset/download.py", "python3 src/train_stage2.py"]
            }
        ]

        resources {
            cpu = "8"
            memory = "8G"
        }
        depends_on = ["train1"]
    }


    parameters {
        DATASET = "1hHulJlmOznP4YK9ppC41rCkSVtkW7J-K"
        DATA_FOLDER = "dataset/flower"
        N_CLASS = "5"
        N_EPOCH = "10"
        BATCH_SIZE = "8"
        SAVE_INTERVAL = "2"
        SAVE_PATH = "/artefact/"
    }
}
