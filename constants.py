# VisualMRC data structure
IMAGE_FILENAME_KEY = 'image_filename'
BOUNDING_BOXES_KEY = 'bounding_boxes'
QA_DATA_KEY = 'qa_data'
OCR_INFO_KEY = 'ocr_info'
SHAPE_KEY = 'shape'
HEIGHT_KEY = 'height'
WIDTH_KEY = 'width'
WORD_KEY = 'word'
BBOX_KEY = 'bbox'
QUESTION_KEY = 'question'
ANSWER_KEY= 'answer'
TEXT_KEY = 'text'
RELEVANT_KEY = 'relevant'
ID_KEY = 'id'

# Columns in the MultiModalRetriever dataset
ROIS_COLUMN = 'rois'
NUM_ROIS_COLUMN = 'num_rois'
ROI_IMAGE_COLUMN = 'roi_image'
ROI_ID = 'roi_id'
ROI_IS_RELEVANT = 'is_relevant'
ROI_BBOX = 'roi_bbox'
ROI_WORDS = 'roi_words'
ROIS_WORDS_BOXES = 'roi_words_boxes'

# MultiModalRetriever module
INPUT_IDS_KEY = 'input_ids'
TOKEN_TYPE_IDS_KEY = 'token_type_ids'
ATTENTION_MASK_KEY = 'attention_mask'
BBOX_KEY = 'bbox'
PIXEL_VALUES_KEY = 'pixel_values'

QUESTION_INPUT_IDS = 'question_input_ids'
QUESTION_TOKEN_TYPE_IDS = 'question_token_type_ids'
QUESTION_ATTENTION_MASK = 'question_attention_masks'
ROIS_INPUT_IDS = 'rois_input_ids'
ROIS_ATTENTION_MASK = 'rois_attention_mask'
ROIS_BBOX = 'rois_bbox'
ROIS_PIXEL_VALUES = 'pixel_values'
LABELS = 'labels'

BBOX_SCALING_FACTOR = 1000