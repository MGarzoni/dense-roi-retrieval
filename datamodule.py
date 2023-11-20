import random
from PIL import Image
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
import torch
from torch import Tensor
from pathlib import Path
from typing import Any
from configargparse import Namespace
import pytorch_lightning as pl
from torch.utils.data import (
    DataLoader,
)
from transformers import AutoTokenizer, LayoutLMv3ImageProcessor, AutoConfig
from datasets import Dataset, DatasetDict
from dataclasses import dataclass
from torch.nn.functional import pad
from constants import *
from torchvision.transforms import Resize

tqdm.pandas()

@dataclass()
class TextOnlyBiEncoderSample:
    """
    Dataclass defining attributes for the text-only bi-encoder sample
    """
    question_input_ids: Tensor
    question_attention_mask: Tensor
    labels: list[int]
    rois_input_ids: Tensor
    rois_attention_mask: Tensor
    
@dataclass()
class VisionOnlyBiEncoderSample:
    """
    Dataclass defining attributes for the vision-only bi-encoder sample
    """
    question_input_ids: Tensor
    question_attention_mask: Tensor
    labels: list[int]
    rois_pixel_values: Tensor
    
@dataclass()
class MultiModalBiEncoderSample:
    """
    Dataclass defining attributes for the multimodal bi-encoder sample
    """
    question_input_ids: Tensor
    question_attention_mask: Tensor
    labels: list[int]
    rois_pixel_values: Tensor
    rois_input_ids: Tensor
    rois_attention_mask: Tensor
    rois_bbox: Tensor

class MultiModalDataset:
    """
    Module is responsible for loading the csv files and apply the roi extraction, transformations, tokenizations and preparation
    before data is passed to MultiModalRetrieverDataModule which passes it to the dataloaders
    """
    def __init__(
            self,
            roi_encoder_ckpt: str,
            path: Path,
            num_neg_rois: int,
            split_name: str,
            width_of_resized_image: int,
            height_of_resized_image: int,
            data_folder_path: Path,
            modality: str,
            num_tot_samples: int
    ):
        # Save datamodule parameters
        self.num_neg_rois = num_neg_rois
        self.split_name = split_name
        self.num_passages = self.num_neg_rois + 1
        self.width_of_resized_image = width_of_resized_image
        self.height_of_resized_image = height_of_resized_image
        self.data_folder_path = data_folder_path
        self.num_tot_samples = num_tot_samples
        self.processed_data = self._load_from_path(path=path)
        self.modality = modality
        
        # Initialize LayoutLMv3 tokenizer for words in ROI
        self.roi_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=roi_encoder_ckpt,
        )

        # Initialize the LayoutLMv3 feature extractor for the roi image and bboxes
        self.roi_features_extractor = LayoutLMv3ImageProcessor.from_pretrained(
            pretrained_model_name_or_path=roi_encoder_ckpt,
            do_resize=False, # already taken care of
            do_normalize=False,
            apply_ocr=False, # already have this info
        )

        # Initialize the configuration of the LMv3 roi encoder
        self.roi_config = AutoConfig.from_pretrained(roi_encoder_ckpt)
        
    def _load_from_path(self, path: Path) -> DatasetDict:
        """
        Function takes the path to the metadata of the MultiModalRetriever data and loads in the DataSet object
        Args:
            path: path, to load the csv data from
        Returns:
            datasetdict, containing the data in a HuggingFace Dataset format backed by an Arrow table (a two-dimensional dataset with chunked arrays for columns, together with a schema providing field names)
        """
        metadata = pd.read_csv(path)

        # Load all data if all is requested
        if str(self.num_tot_samples) == "all":
            dict_dataset = Dataset.from_dict(metadata)
        else:
            # else load its first self.num_tot_samples
            cap = int(self.num_tot_samples)
            dict_dataset = Dataset.from_dict(metadata[:cap])
            
        processed_data = DatasetDict({f"{self.split_name}": dict_dataset})

        return processed_data
    
    def _extract_roi_image(self, doc_image: Image, roi_bbox: list[int]) -> Image.Image:
        """
        Extracts a region of interest (ROI) from an image based on the given coordinates.
        Args:
            doc_image: the PIL image file of the doc image
            roi_bbox: list of coordinates in COCO format of roi bbox
        Returns:
            PIL.Image.Image: extracted region of interest as a PIL image
        """
                
        # extract coordinates
        x_min, y_min, width, height = roi_bbox
        
        # calculate the region box
        roi_region = (x_min, y_min, x_min + width, y_min + height)
        
        # crop the image to the region box
        roi_image = doc_image.crop(roi_region)
        
        return roi_image
    
    def _normalize_bbox_llmv3(self, bbox: list[int], width: int, height: int) -> list[int]:
        """
        LayoutLMv3 tokenizer expects the boxes to be in the range 0 to 1000, thus the purpose of this method is to:
            - first divide the coordinate value by the size of the img
            - multiply by the BBOX_SCALING_FACTOR = 1000
            - cast the coordinate to integer
        Implementation is based on https://github.com/huggingface/transformers/blob/172f42c512e1bf32554ef910fe82f07916b4d4af/src/transformers/pipelines/document_question_answering.py#L53
        Args:
            bbox: list, containing the bbox coordinates in pascal_voc format
        Returns:
            normalized_bbox: list, containing normalized bbox coordinates to range expected by lmv3
        """
        normalized_bbox = [
            int(BBOX_SCALING_FACTOR * (bbox[0] / width)),
            int(BBOX_SCALING_FACTOR * (bbox[1] / height)),
            int(BBOX_SCALING_FACTOR * (bbox[2] / width)),
            int(BBOX_SCALING_FACTOR * (bbox[3] / height)),
        ]
        return normalized_bbox
    
    def _coco_to_pascal_voc(self, rois_words_boxes: list[list[int]]) -> list[list[int]]:
        """
        Method to convert input boxes in coco format ([x_min, y_min, width, height]) to pascal_voc format ([x_min, y_min, x_max, y_max])
        Args:
            bbox: list of list of ints, containing bbox coordinates of words for every roi in coco format
        Returns:
            rois_actual_words_bboxes: list of list of ints, containing bbox coordinates of words for every roi in pascal_voc format
        """
        rois_actual_words_bboxes = []
        for roi in rois_words_boxes:
            roi_bbox = []
            for bbox in roi:
                x_min, y_min, width, height = bbox
                roi_bbox.append([x_min, y_min, x_min + width, y_min + height])
            rois_actual_words_bboxes.append(roi_bbox)
        return rois_actual_words_bboxes
    
    def _get_rois_data(self, item: dict[str, Any]) -> pd.DataFrame:
        """
        This method receives an item (row from data df) and extracts one positive roi and self.num_neg_rois negative rois
        Args:
            item: dict, the current row to extract rois data from
        Returns:
            rois_data: pandas dataframe, where the first element is a positive roi and the self.num_neg_rois other elements are negative rois
        """
        
        # Get id of a positive roi and ids of negative rois
        pos_roi_id = random.choice(seq=[roi[ROI_ID] for roi in item[ROIS_COLUMN] if roi[ROI_IS_RELEVANT]])
        neg_rois_ids = random.sample(
            population=[roi[ROI_ID] for roi in item[ROIS_COLUMN] if not roi[ROI_IS_RELEVANT]],
            k=self.num_neg_rois
        )
        
        # Extract pos roi and negative rois
        pos_roi = []
        neg_rois = []
        for roi in item[ROIS_COLUMN]:
            if roi[ROI_ID] == pos_roi_id:
                pos_roi.append(roi)
            elif roi[ROI_ID] in neg_rois_ids:
                neg_rois.append(roi)
                
        # Make a df where first row is the positive roi and other rows are negative rois
        rois_data = pd.DataFrame(data=pos_roi+neg_rois)
        
        return rois_data
    
    # TRANSFORMATIONS: RESIZE IMAGES AND TRANSFORM BOXES
    def _resize_roi_image(self, roi_image: Image.Image) -> Image.Image:
        """
        This function is used to resize the images to (224,224) expected by the LayoutLMv3 image processor
        Args:
            roi_image: PIL.Image, source image of the extracted roi
        Returns:
            resized_image: PIL.Image, source image resized to (224,224) size
        """

        resized_image = Resize(size=(self.width_of_resized_image, self.height_of_resized_image))(roi_image)
        return resized_image

    def _transform_words_boxes(self, width: int, height: int, words_boxes: list[list[int]]) -> list[list[int]]:
        """
        This function is used to transform the bounding boxes coordinates of the words, so they are relative to the resized images
        Args:
            width: int, width of source roi image
            height: int, height of source ROI
            words_boxes: list of lists, bounding box coordinates of each word in the ROI
        Returns:
            transformed_boxes: list, list of lists containing the updated bounding box coordinates of the words in the ROI
        """

        # scale the start x and y coordinates to the target width and height
        scale_top_left_x = self.width_of_resized_image / width
        scale_top_left_y = self.height_of_resized_image / height

        # initialize list to add transformed boxes to
        transformed_words_boxes = []
        for word_box in words_boxes:
            old_top_left_x, old_top_left_y, old_width, old_height = word_box

            # new coordinates result from multiplying the old coordinates by the scaling factors
            new_top_left_x = int(old_top_left_x * scale_top_left_x)
            new_top_left_y = int(old_top_left_y * scale_top_left_y)
            new_width = int(old_width * scale_top_left_x)
            new_height = int(old_height * scale_top_left_y)

            # add new boxes for each word
            transformed_words_boxes.append([new_top_left_x, new_top_left_y, new_width, new_height])

        return transformed_words_boxes
    
    def _process_item(self, item: pd.Series): # not sure how to set the return type to be one of TextOnlyBiEncoderSample, VisionOnlyBiEncoderSample, MultiModalBiEncoderSample
        """
        This method is responsible for processing the input item (row from data df), extracting the rois data, transforming images and words boxes and tokenizing the inputs
        Args:
            item: dict, the current row to extract data from
        Returns:
            Any of [TextOnlyBiEncoderSample, VisionOnlyBiEncoderSample, MultiModalBiEncoderSample], based on specified modality
        """
        
        # Convert str to list of dicts
        item[ROIS_COLUMN] = literal_eval(item[ROIS_COLUMN])
        
        # Tokenize and extract all the question inputs
        tokenized_question = self.roi_tokenizer(
            item[QUESTION_KEY].split(),
            boxes=[[0,0,0,0]] * len(item[QUESTION_KEY].split()), # pass 0-coordinate bboxes just because the lmv3 tokenizer expects the bboxes
            truncation=True,
            padding=True, # set the padding to True so that within one sample the tensors have the same length
            return_tensors="pt"
        ).data
        
        # Save label ids which are same for every modality
        label_ids = [1] * 1 + [0] * self.num_neg_rois
        
        # Get pos_roi and neg_rois in dataframe format and ensure amount of rois
        rois_data = self._get_rois_data(item=item)
        if rois_data.shape[0] != self.num_passages:
            rois_data = rois_data[:self.num_passages]

        # For text
        if self.modality == "text":
            
            # Save words and words boxes
            rois_words = list(rois_data.roi_words)
            rois_words_boxes = list(rois_data.roi_words_boxes)

        # For vision and multimodal
        elif self.modality in ["vision", "multi"]:
            
            # Open image
            # print("HERE OPENING IMAGE\n")
            doc_image = Image.open(Path(self.data_folder_path, item[IMAGE_FILENAME_KEY]))
            doc_image_width, doc_image_height = doc_image.size
            
            # Save words and words boxes
            # print("HERE GETTING WORDS AND BOXES\n")
            rois_words = list(rois_data.roi_words)
            rois_words_boxes = list(rois_data.roi_words_boxes)
            
            # Convert rois_words_boxes from coco to pascal_voc format calling self._coco_to_pascal_voc
            # print("HERE CONVERTING TO PASCAL\n")
            converted_rois_words_boxes = self._coco_to_pascal_voc(rois_words_boxes=rois_words_boxes)
            
            # Create new column and save the arrays versions of the rois images
            # print("HERE GETTING IMAGES AS ARRAYS\n")
            rois_data[ROI_IMAGE_COLUMN] = rois_data.apply(
                lambda x: self._extract_roi_image(
                    doc_image=doc_image,
                    roi_bbox=x[ROI_BBOX]
                    ),
                axis=1
            )
            rois_images = list(rois_data.roi_image)
            # print(f"HERE ALL IMAGES: {rois_images}\n")
            
            # Apply transformations: image resizing and bboxes rescaling
            resized_rois_images = []
            transformed_rois_words_boxes = []
            for roi_image, roi_words_boxes in zip(rois_images, converted_rois_words_boxes):
                
                # Resize image and add to list
                # print(f"RESIZING HERE === {self._resize_roi_image(roi_image=roi_image)}\n\n")
                resized_rois_images.append(self._resize_roi_image(roi_image=roi_image))
                
                # Transform words boxes and add to list
                transformed_rois_words_boxes.append(
                    self._transform_words_boxes(
                        width=doc_image_width,
                        height=doc_image_height,
                        words_boxes=roi_words_boxes
                    )
                )
            # print(f"HERE ALL RESIZED IMAGES before normalizing boxes: {resized_rois_images}\n")
            
            # Normalize bboxes by the LMv3 scaling factor
            norm_rois_words_boxes = []
            for roi in transformed_rois_words_boxes:
                roi_words_boxes = []
                for word_boxes in roi:
                    
                    # Normalize to create segment-level boxes for LMv3
                    roi_words_boxes.append(
                        self._normalize_bbox_llmv3(
                            bbox=word_boxes,
                            width=self.width_of_resized_image,
                            height=self.height_of_resized_image
                        )
                    )
                norm_rois_words_boxes.append(roi_words_boxes)
            
            # Process pixel_values
            # print(f"HERE FINALLY ALL RESIZED IMAGES: {resized_rois_images}\n")
            rois_pixel_values = self.roi_features_extractor(
                images=resized_rois_images,
                return_tensors="pt"
            )[PIXEL_VALUES_KEY]

        # Get roi tokenized inputs
        processed_rois = self.roi_tokenizer(
            text=rois_words,
            boxes=norm_rois_words_boxes if self.modality == 'multi' else rois_words_boxes, # still passing boxes even in text-only just because they are required by the call, but then aren't passed to the encoder
            truncation=True,
            padding=True, # set the padding to True so that within one sample the tensors have the same length
            return_tensors="pt"
        ).data

        # text-only
        if self.modality == "text":
            return TextOnlyBiEncoderSample(
            tokenized_question[INPUT_IDS_KEY],
            tokenized_question[ATTENTION_MASK_KEY],
            label_ids,
            processed_rois[INPUT_IDS_KEY],
            processed_rois[ATTENTION_MASK_KEY]
        )
        
        # vision-only
        elif self.modality == "vision":
            return VisionOnlyBiEncoderSample(
            tokenized_question[INPUT_IDS_KEY],
            tokenized_question[ATTENTION_MASK_KEY],
            label_ids,
            rois_pixel_values
        )
        
        # multimodal
        else:
            return MultiModalBiEncoderSample(
                tokenized_question[INPUT_IDS_KEY],
                tokenized_question[ATTENTION_MASK_KEY],
                label_ids,
                rois_pixel_values,
                processed_rois[INPUT_IDS_KEY],
                processed_rois[ATTENTION_MASK_KEY],
                processed_rois[BBOX_KEY]
            )

    def _pad_sequence(self, processed_sequence: list[Tensor], max_length: int) -> list[Tensor]:
        """
        Pad the text-related sequences
        Args:
            processed_sequence: list, containing tensors such as attention_mask or input_ids
            max_length: int, max length value to pad the sequences to
        Returns:
            list of tensors, containing padded sequences
        """
        return [
            pad(
                input=seq,
                pad=(0, max_length - seq.size(1))
            )
            for seq in processed_sequence
        ]

    def _pad_sequence_bboxes(self, roi_bboxes: list[Tensor], max_length: int) -> list[Tensor]:
        """
        Pad the bbox sequences
        Args:
            roi_bboxes: list, containing tensors of the bbox sequences
            max_length: int, max length value to pad the sequences to
        Returns:
            list of tensors, containing padded sequences
        """
        return [
            pad(
                input=sequence,
                pad=(0, 0, 0, max_length - sequence.size(1), 0, 0)
            ) 
            for sequence in roi_bboxes
        ]

    def _collate_fn(self, batch: dict[Tensor]) -> dict[list[Tensor]]:
        """
        Implementation of the custom collate function
        Args:
            batch: dict, containing tensors holding the tokenized inputs
        Returns:
            tokenized_elements: dict of lists of tensors, containing tokenized inputs
        """

        # Initialize dictionary with lists where to add the tokenized texts inputs
        tokenized_elements = {
            QUESTION_INPUT_IDS: [],
            QUESTION_ATTENTION_MASK: [],
            ROIS_INPUT_IDS: [],
            ROIS_ATTENTION_MASK: []
        }

        # Initialize pixel values, bboxes and labels lists where to save the processed inputs
        if self.modality != "text": # for vision and multimodal
            rois_pixel_values = []
            if self.modality == "multi": # only for multimodal
                rois_bbox = []
        labels = []

        # Iterate through batch
        for item in batch:
            
            # Apply processing on item
            sample = self._process_item(item=item)
            
            # Add tokenized question inputs
            tokenized_elements[QUESTION_INPUT_IDS].append(sample.question_input_ids)
            tokenized_elements[QUESTION_ATTENTION_MASK].append(sample.question_attention_mask)

            # Add tokenized roi inputs
            if self.modality != "vision": # for text and multimodal
                tokenized_elements[ROIS_INPUT_IDS].append(sample.rois_input_ids)
                tokenized_elements[ROIS_ATTENTION_MASK].append(sample.rois_attention_mask)
            if self.modality != "text": # for vision and multimodal
                rois_pixel_values.append(sample.rois_pixel_values)
                if self.modality == "multi": # only for multimodal
                    rois_bbox.append(sample.rois_bbox)
            
            # Add labels
            labels.append(sample.labels)

        # Find the max seq_length between question and roi texts and set it as max_seq_length which will be used for padding
        max_question_seq_length = max([t.shape[1] for t in tokenized_elements[QUESTION_INPUT_IDS]])
        if self.modality != "vision": # only for text and multimodal
            max_roi_seq_length = max([t.shape[1] for t in tokenized_elements[ROIS_INPUT_IDS]])
            max_seq_length = max(max_question_seq_length, max_roi_seq_length)

        # Pad the text elements first
        for key, seq in tokenized_elements.items():
            if self.modality != "vision": # only for text and multimodal
                stacked_sequence = torch.stack(self._pad_sequence(
                    processed_sequence=seq,
                    max_length=max_seq_length
                ))
                tokenized_elements[key] = stacked_sequence
            else:
                # now vision, so only stack the question related keys
                if "question" in key:
                    stacked_sequence = torch.stack(self._pad_sequence(
                        processed_sequence=seq,
                        max_length=max_question_seq_length # just use the max questions len since no roi text is passed
                    ))
                    tokenized_elements[key] = stacked_sequence
                
        # Pad the bboxes
        if self.modality == "multi": # only for multimodal
            rois_bbox = torch.stack(self._pad_sequence_bboxes(
                roi_bboxes=rois_bbox,
                max_length=max_seq_length
            ))

        # Stack pixel_values and label
        if self.modality != "text": # for vision and multimodal
            rois_pixel_values = torch.stack([p.squeeze() for p in rois_pixel_values])
        labels = torch.stack([torch.LongTensor(label) for label in labels])
        tokenized_elements.update({'labels': labels})

        # Update the dictionary with the rest of the elements and return it as a batch
        if self.modality == "vision": # only for text and multimodal
            tokenized_elements.update({
                ROIS_PIXEL_VALUES: rois_pixel_values,
                LABELS: labels
            })
            del tokenized_elements[ROIS_INPUT_IDS]
            del tokenized_elements[ROIS_ATTENTION_MASK]
        if self.modality == "multi": # only for multimodal
            tokenized_elements.update({
            ROIS_PIXEL_VALUES: rois_pixel_values,
            ROIS_BBOX: rois_bbox,
            LABELS: labels
        })

        return tokenized_elements
        
    def to_dataloader(self, batch_size: int, shuffle: bool, drop_last: bool) -> DataLoader:
        """
        Returns data in the Data Loader format, using the custom _collate_fn
        Args:
            batch_size: int, specifying the size for each batch
            shuffle: bool, whether to shuffle the data or not
            drop_last: bool, whether to drop the last batch or not
        Returns:
            dataloader, containing data in batched of defined size
        """        
        return DataLoader(
            dataset=self.processed_data[self.split_name],
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=self._collate_fn,
            num_workers=10
        )
        
class MultiModalRetrieverDataModule(pl.LightningDataModule):
    """
    Initialise data by calling Dataset class and return dataloader objects
    """
    def __init__(
            self,
            roi_encoder_ckpt: str,
            question_encoder_ckpt: str,
            num_neg_rois: int,
            height_of_resized_image: int,
            width_of_resized_image: int,
            train_set_path: Path,
            test_set_path: Path,
            val_set_path: Path,
            batch_size: int,
            data_folder_path: Path,
            modality: str,
            num_tot_samples: int
    ):
        super().__init__()

        # Encoder checkpoints
        self.roi_encoder_ckpt = roi_encoder_ckpt
        self.question_encoder_ckpt = question_encoder_ckpt

        # Datamodule specific args
        self.num_neg_rois = num_neg_rois
        self.width_of_resized_image = width_of_resized_image
        self.height_of_resized_image = height_of_resized_image
        self.batch_size = batch_size
        self.modality = modality
        self.num_tot_samples = num_tot_samples

        # Splits
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Splits paths
        self.data_folder_path = data_folder_path
        self.train_set_path = train_set_path
        self.test_set_path = test_set_path
        self.val_set_path = val_set_path

    @classmethod
    def from_argparse_args(cls, args: Namespace, data_folder_path: str):

        multimodal_datamodule = cls(
            roi_encoder_ckpt=args.roi_encoder_ckpt,
            question_encoder_ckpt = args.question_encoder_ckpt,
            num_neg_rois=args.num_neg_rois,
            width_of_resized_image=args.width_of_resized_image,
            height_of_resized_image=args.height_of_resized_image,
            data_folder_path=Path(data_folder_path),
            train_set_path=Path(data_folder_path, args.train_path),
            test_set_path=Path(data_folder_path, args.test_path),
            val_set_path=Path(data_folder_path, args.test_path),
            batch_size=args.batch_size,
            modality=args.modality,
            num_tot_samples=args.num_tot_samples
        )

        return multimodal_datamodule

    def setup(self, stage: str) -> MultiModalDataset:
        """
        Setup the dataloader based on the stage
        Args:
            stage: str, specify for which step [fit, test] to set the dataloader for
        Returns:
            MultiModalDataset, holding data for given stage
        """
        
        # For train and val steps, set train and val dataloaders
        if stage == "fit":
            
            # Train
            self.train_dataset = MultiModalDataset(
                roi_encoder_ckpt=self.roi_encoder_ckpt,
                path=self.train_set_path,
                split_name="train",
                num_neg_rois=self.num_neg_rois,
                width_of_resized_image=self.width_of_resized_image,
                height_of_resized_image=self.height_of_resized_image,
                data_folder_path=self.data_folder_path,
                modality=self.modality,
                num_tot_samples=self.num_tot_samples
            )

            # Val
            self.val_dataset = MultiModalDataset(
                roi_encoder_ckpt=self.roi_encoder_ckpt,
                path=self.val_set_path,
                split_name="val",
                num_neg_rois=self.num_neg_rois,
                width_of_resized_image=self.width_of_resized_image,
                height_of_resized_image=self.height_of_resized_image,
                data_folder_path=self.data_folder_path,
                modality=self.modality,
                num_tot_samples=self.num_tot_samples
            )

        # Test step
        if stage == "test":
            self.test_dataset = MultiModalDataset(
                roi_encoder_ckpt=self.roi_encoder_ckpt,
                path=self.test_set_path,
                split_name="test",
                num_neg_rois=self.num_neg_rois,
                width_of_resized_image=self.width_of_resized_image,
                height_of_resized_image=self.height_of_resized_image,
                data_folder_path=self.data_folder_path,
                modality=self.modality,
                num_tot_samples=self.num_tot_samples
            )

    def train_dataloader(self) -> DataLoader:
        """
        Calls to_dataloader function that wraps data into dataloader object for training set.
        """
        return self.train_dataset.to_dataloader(batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        """
        Calls to_dataloader function that wraps data into dataloader object for validation set.
        """
        return self.val_dataset.to_dataloader(batch_size=self.batch_size, shuffle=False, drop_last=True)

    def test_dataloader(self) -> DataLoader:
        """
        Calls to_dataloader function that wraps data into dataloader object for testing set.
        """
        return self.test_dataset.to_dataloader(batch_size=self.batch_size, shuffle=False, drop_last=True)

    @staticmethod
    def add_dataloader_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MultiModalDPRDataModule")
        parser.add_argument(
            "--num-neg-rois",
            type=int,
            help="Number of the negative samples needed for the split of ROIS"
        )
        parser.add_argument(
            "--width-of-resized-image",
            type=int,
            help="Width of the image that is accepted by the LayoutLMv3 model"
        )
        parser.add_argument(
            "--height-of-resized-image",
            type=int,
            help="Height of the image that is accepted by the LayoutLMv3 model"
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            help="Batch size",
        )
        parser.add_argument(
            "--modality",
            type=str,
            help="Controls which data is used for the experiment; may be 'text', 'vision' or 'multimodal'",
        )
        parser.add_argument(
            "--num-tot-samples",
            help="Controls how many total samples are loaded, can be an int or a string saying 'all'",
        )
        return parent_parser
