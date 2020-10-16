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
            cpu = "1"
            memory = "8G"
            gpu = "1"
        }
        depends_on = ["get_dataset"]
    }

    step train2 {
        image = "python:3.7"
        install = ["pip3 install -r src/requirements.txt"]
        script = [
            {
                sh = ["python3 dataset/download.py", "python3 src/train_stage2.py"]
            }
        ]

        resources {
            cpu = "1"
            memory = "8G"
            gpu= "1"
        }
        depends_on = ["train1"]
    }


    parameters {
        DATASET = "1hHulJlmOznP4YK9ppC41rCkSVtkW7J-K"
        DATA_FOLDER = "dataset/flower"
        N_CLASS = "5"
        N_EPOCH = "10"
        BATCH_SIZE = "4"
        SAVE_INTERVAL = "2"
        LOG_INTERVAL = "100"
        SAVE_PATH = "/artefact/"
        IMAGE_SIZE = "224"
    }
}
