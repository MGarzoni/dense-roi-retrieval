# Abstract

This master thesis proposes a multi-modal retrieval system for open-domain question answering, building upon dense passage retrievers and incorporating multi-modal information to retrieve candidate regions of interest (ROIs) from document images given a user query. Our main research goal was to investigate the efficacy of dense representations of questions and multi-modal contexts in retrieving relevant content, and evaluating the impact of multi-modal information compared to uni-modal baselines. To this end, the study leverages the VisualMRC dataset which offers annotations for visual components, particularly ROIs such as titles or graphs, to facilitate efficient content retrieval. The proposed methodology involves pre-processing the multi-modal ROIs, employing a bi-encoder setup to encode the question and ROIs separately, and use such encodings to calculate similarity in their shared multi-dimensional embedding space. The training objective is achieved through contrastive learning by passing to the model a question, along with one positive and k negative contexts, and minimizing the loss function by reducing the negative log-likelihood associated with the positive ROI. We evaluate our trained models on three different modality scenarios, text-only, vision-only, and multi-modal, and we evaluate their retrieval performance on standard metrics such as Normalized Cumulative Discounted Gain @ k, Mean Re- ciprocal Rank @ k, and Recall @ k. The results reveal the benefits of both vision-only and multi-modal approaches over text-only, while also highlighting challenges related to the number of negative ROIs. Our results support the first hypothesis but raise questions about the second, suggesting that the inclusion of layout information may not always improve retrieval performance. The strengths of our approach include efficient ROI retrieval and dataset adaptability, while limitations involve dataset variability and encoding techniques. In light of this, we suggest several avenues for future work such as exploring new datasets, incorporating hard negatives in contrastive learning, and refining ROI dissimilarity. Additionally, we speculate that integrating keyword matching and retrieval-augmented generation approaches could enhance the retrieval pipeline. Overall, the present thesis hopes to advance research in multimodal retrieval models, emphasizing the importance of visual and textual context for open-domain question answering.

# Setup

Create a development environment with conda or pipenv, with python version 3.10.12, and activate it:
`conda create -n env-name python=3.10.12`
`conda activate env-name`
Then navigate to the source directory of this repo:
`cd path/to/repo`
Install in the new environment the required packages and libraries for this project:
`python -m pip install -r requirements.txt`

# Data

Contact the author for dowloading the original VisualMRC dataset. Once downloaded and unpacked, it should live in the source repo directory, and be named as `vmrc_data` folder, with `images`, `new_format_splits` and `og_splits` as subfolders containing the various files. These are the default directories the system will use, but may be overriden in the `parameters.ini` file.

# Files description

The `main.py` file is used to execute the multi-modal retrieval pipeline. The `parameters.ini` file holds values for all the parameters used in the execution of the system, and these can also be configured in the CLI when running the main file, e.g.: `python main.py --num-neg-rois=16`. The `data_preparation.py` is responsible for loading the VisualMRC data in original format and convert it to a format that can be tokenized in batches and passed to the MultiModalRetriever. The `datamodule.py` file implements two classes: (1) the `MultiModalDataset` is responsible for loading the csv files and applying the roi extraction, transformations, tokenizations and preparation, before data is passed to (2) the `MultiModalRetrieverDataModule` which passes it to the dataloaders. The `models.py` implements a bi-encoder model for multi-modal retrieval using the PyTorchLightning framework, and is responsible for the encoding mechanisms, the similarity and loss calculations, the train, evaluation and testing steps, as well as the calculation of retrieval metrics at both the instance and corpus levels. The `constants.py` file holds all the constants used across the pipeline. The `parser_utils.py` is used to initialize and hold the parsing of the configuration arguments.

# Run training

Open a new terminal and run `python main.py` and experiment with the following parameters: `num-neg-rois`, `batch-size`, `acc-grad-batches` and, most importantly, `modality`. The description of each parameter is provided in its relative file.
