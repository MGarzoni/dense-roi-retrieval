import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from configargparse import Namespace
from typing import Any
from constants import *

tqdm.pandas()

class VisualMRCPrepper:
    """
    Load VisualMRC data in original format and convert it to a format that can be tokenized in batches and passed to the MultiModalRetriever
    """

    def __init__(
            self,
            visual_mrc_train_path: str,
            visual_mrc_testing_path: str,
            visual_mrc_validation_path: str,
            load_multimodal_csv_data: bool,
            train_path: str,
            test_path: str,
            val_path: str,
            num_neg_rois: int,
            num_tot_samples: Any
    ):

        # Save the paths to the original json files (original = in old nested format)
        self.visual_mrc_train_path = visual_mrc_train_path
        self.visual_mrc_testing_path = visual_mrc_testing_path
        self.visual_mrc_validation_path = visual_mrc_validation_path
        self.load_multimodal_csv_data = load_multimodal_csv_data

        # Save the paths to the newly formatted csv files
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        
        self.num_neg_rois = num_neg_rois
        self.num_passages = self.num_neg_rois + 1
        self.num_tot_samples = num_tot_samples

        # Initialize data dict
        self.data = dict()

    @classmethod
    def from_argparse_args(cls, args: Namespace):

        data_prepper = cls(
            visual_mrc_train_path=args.visual_mrc_train_path,
            visual_mrc_testing_path=args.visual_mrc_testing_path,
            visual_mrc_validation_path=args.visual_mrc_validation_path,
            load_multimodal_csv_data=args.load_multimodal_csv_data,
            train_path=args.train_path,
            test_path=args.test_path,
            val_path=args.val_path,
            num_neg_rois=args.num_neg_rois,
            num_tot_samples=args.num_tot_samples
        )

        return data_prepper

    def _load_json(self, data_folder_path: Path, split_name: str) -> list[dict[str, Any]]:
        """
        Given the data_folder_path and a split_name, this function loads in the json file relative to the split_name
        Args:
            data_folder_path: Path to the folder which contains all the data
            split_name: str, specifying which of the splits [train/test/val] needs to be loaded
        Returns:
            dict, the contents of the input json file for the given split
        """
        if split_name == "train":
            path_to_file = Path(data_folder_path, self.visual_mrc_train_path)
        elif split_name == "val":
            path_to_file = Path(data_folder_path, self.visual_mrc_validation_path)
        else:
            path_to_file = Path(data_folder_path, self.visual_mrc_testing_path)
        
        # Load json
        with open(path_to_file, "rt") as f:
            data = json.load(f)

        if self.num_tot_samples and self.num_tot_samples != 'all':
            return data[:int(self.num_tot_samples)]
        else:
            return data
    
    def _translate_word_bbox(self, bbox: list[int], roi_shape: list[int]) -> list[int]:
        """
        Translate the bboxes of each word in roi to locate it within the roi image (and not the original doc image)
        Args:
            bbox: list, containing bbox coordinates in COCO list format
            roi_shape: list, containing coordinates of roi image
        Returns:
            list of bbox coordinates in COCO format, preserving original width and height
        """
        x_min, y_min, width, height = bbox
        new_x_min = x_min - roi_shape[0]
        new_y_min = y_min - roi_shape[1]
        
        return [new_x_min, new_y_min, width, height]
    
    def _process_data_to_df(self, og_data: list[dict[str, Any]]) -> pd.DataFrame:
        """
        This method is used to extract the information from the json file and create a dataframe where every row refers to a QA pair and the dataframe contains the following columns:
            question, answer, image_filename, rois(roi_id, is_relevant, roi_bbox, roi_words, roi_words_boxes)
        Args:
            og_data: list of dicts, containing data in original format
        Returns:
            data_df: dataframe where every row refers to a QA pair
        """
        rows = []
        
        # Iterate through all the samples
        for item in tqdm(og_data, desc="Creating DataFrame"):
            image_filename = item[IMAGE_FILENAME_KEY]
            
            # Iterate through QA data and save question, answer and relevant ids
            # for qa_item in item['qas']:
            for qa_item in item[QA_DATA_KEY]:
                question = qa_item[QUESTION_KEY][TEXT_KEY]
                answer = qa_item[ANSWER_KEY][TEXT_KEY]
                relevant_ids = set(qa_item[ANSWER_KEY][RELEVANT_KEY])
                
                # Iterate throught the rois for this sample and get roi_id, its bbox from the doc image and save a boolean to indicate whether or not it is relevant to answering the question
                rois = []
                # for bbox_item in item[ROIS_COLUMN]:
                for bbox_item in item[BOUNDING_BOXES_KEY]:
                    roi_id = bbox_item[ID_KEY]
                    roi_bbox = list(bbox_item[SHAPE_KEY].values())
                    is_relevant = roi_id in relevant_ids
                    
                    # Iterate through the ocr info for the current roi and get each word and its bboxes (translated)
                    roi_words = []
                    roi_words_boxes = []
                    if OCR_INFO_KEY not in bbox_item.keys():
                        roi_words = []
                        roi_words_boxes = []
                    else:
                        for ocr_item in bbox_item.get(OCR_INFO_KEY):
                            roi_words.append(ocr_item[WORD_KEY])
                            roi_words_boxes.append(
                                self._translate_word_bbox(
                                    bbox=list(ocr_item[BBOX_KEY].values()),
                                    roi_shape=roi_bbox
                                )
                            )
                    
                    # Save roi data
                    roi_data = {
                        ROI_ID: roi_id,
                        ROI_IS_RELEVANT: is_relevant,
                        ROI_BBOX: roi_bbox,
                        ROI_WORDS: roi_words,
                        ROIS_WORDS_BOXES: roi_words_boxes
                    }
                    
                    # Append roi data to the rois for each roi
                    rois.append(roi_data)
                
                # For each QA pair, append the information 
                rows.append({
                    QUESTION_KEY: question,
                    ANSWER_KEY: answer,
                    IMAGE_FILENAME_KEY: image_filename,
                    ROIS_COLUMN: rois
                })
        
        # Cast to pandas dataframe, where every row is a QA pair
        data_df = pd.DataFrame(rows)
        return data_df
    
    def _ensure_num_rois(self, final_data_df: pd.DataFrame) -> pd.DataFrame:
        """
        This method is used to ensure that the rois column avlue for each QA pair (row) contains at least the amount of rois requested for the experiment
        As of now this forces the user to run the data_preparation.py script and generate the new csvs every time the self.num_neg_rois need to be changed
        Args:
            final_data_df: pandas dataframe, output of self._process_data_to_df, where every row is a QA pair with their rois associated information
        Returns:
            processed_data: pandas dataframe, same format as input but now we are sure that every value at in rois column contains the at least the amount of rois requested for the experiment
        """
        
        # Add a column with the amount of rois per qa pair
        final_data_df[NUM_ROIS_COLUMN] = final_data_df.apply(lambda x: len(x[ROIS_COLUMN]), axis=1)

        # Normalize amount of negative roi ids for qa pair, ensuring there are as many as requested
        # to_drop = 0
        new_data = []
        for item_id, item in tqdm(
            iterable=final_data_df.iterrows(),
            desc=f"Ensuring correct number of ROIs per sample ...",
            total=len(final_data_df)
        ):
            relevant_ids = set([i[ROI_ID] for i in item[ROIS_COLUMN] if i[ROI_IS_RELEVANT]])
            negative_ids = set([i[ROI_ID] for i in item[ROIS_COLUMN] if not i[ROI_IS_RELEVANT]])
            
            # if enough negatives, just append the current item, skip otherwise
            if len(negative_ids) >= self.num_neg_rois:
                new_data.append(item)
            else:
                continue
                
            assert len(relevant_ids) > 0, f"not even one relevant roi id for this qa pair"
        
        # Create final dataframe with processed data
        processed_data = pd.DataFrame(data=new_data)
        print(f'Sizes => original: {len(final_data_df)}, processed: {len(processed_data)}')
        
        return processed_data
    
    def parse_visualmrc_data(self, data_folder_path: Path, split_name: str) -> pd.DataFrame:
        """
        This function loads the json file for the given split in the original VisualMRC format (metadata),
        and parses it using the methods defined above to return a formatted version for the MultiModalRetriever
        Args:
            data_folder_path: Path to the folder which contains all the data
            split_name: str, specifying which of the splits [train/test/val] needs to be loaded
        Returns:
            dataframe, containing all the data of given split in a format where each row is represented by a question
        """
        
        # Load original json file
        og_data = self._load_json(data_folder_path=Path(data_folder_path), split_name=split_name)
        
        # Process data from clean json to pd dataframe where each row is a QA pair with its associated ROIs
        final_data_df = self._process_data_to_df(og_data=og_data)
        
        # Ensure num rois
        processed_data = self._ensure_num_rois(final_data_df=final_data_df)
        
        return processed_data
    
    def save_to_csv(self, data: pd.DataFrame, data_folder_path: Path, split_name: str) -> None:
        """
        Function first sets the path to the output file for the processed data and then saves it as csv in the defined location
        Args:
            data: reformatted data in DataFrame format
            data_folder_path: path to the folder with all the data in it
            split: str that specifies if the train/test/val set needs to be loaded
        """
        if split_name == "train":
            output_file_path = Path(data_folder_path, self.train_path)
        elif split_name == "val":
            output_file_path = Path(data_folder_path, self.val_path)
        else:
            output_file_path = Path(data_folder_path, self.test_path)

        # Save data to csv
        data.to_csv(output_file_path)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("VisualMRCPrepper")
        parser.add_argument(
            "--visual-mrc-train-path",
            type=str,
            help="Path to file that contains the VisualMRC training data"
        )
        parser.add_argument(
            "--visual-mrc-testing-path",
            type=str,
            help="Path to the file that contains the VisualMRC testing data"
        )
        parser.add_argument(
            "--visual-mrc-validation-path",
            type=str,
            help="Path to the file that contains the VisualMRC validation data"
        )
        parser.add_argument(
            "--load-multimodal-csv-data",
            action="store_true",
            help="Option to load the the predefined data splits in VisualMRC format",
        )
        parser.add_argument(
            "--train-path",
            type=str,
            help="Path to file that contains the training data ready to be loaded into the MultiModalRetriever"
        )
        parser.add_argument(
            "--test-path",
            type=str,
            help="Path to the file that contains the VisualMRC testing data ready to be loaded into the MultiModalRetriever"
        )
        parser.add_argument(
            "--val-path",
            type=str,
            help="Path to the file that contains the VisualMRC validation data ready to be loaded into the MultiModalRetriever"
        )
        
        return parent_parser
