import logging
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from configargparse import Namespace
from torch import Tensor
from transformers import AutoModel
from recsys_metrics import precision, recall, hit_rate, mean_reciprocal_rank, normalized_dcg
from constants import *
from utils.ranger21 import Ranger21
from torch.optim.lr_scheduler import LambdaLR

# init logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s:\t%(message)s"
)

class MultiModalRetriever(pl.LightningModule):
    """
    Class implementing a bi-encoder model for multi-modal retrieval of regions of interest
    """

    def __init__(
            self,
            roi_encoder_ckpt:str,
            lr: float,
            min_lr: float,
            max_lr: float,
            warmup_ratio: float,
            optimizer: str,
            max_epochs: int,
            similarity_function: str,
            modality: str,
            batch_size: int,
            num_batches_per_epoch: int,
            num_neg_rois: int,
            precision: str,
            grad_clip_val: float,
            acc_grad_batches: int,
            metric_at_k: int
        ):
        super().__init__()

        # Training params
        self.lr = lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.similarity_function = similarity_function
        self.modality = modality
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.total_iters = self.num_batches_per_epoch * self.max_epochs
        self.warmup_ratio = warmup_ratio
        self.warmup_steps = self.total_iters * self.warmup_ratio
        self.num_neg_rois = num_neg_rois
        self.num_passages = self.num_neg_rois + 1
        self.precision = precision
        self.grad_clip_val = grad_clip_val
        self.acc_grad_batches = acc_grad_batches
        self.metric_at_k = metric_at_k
        self.corpus_level_metrics = None

        # Initialize lists where to save train and val steps outputs
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Initialize encoder
        self.roi_encoder = AutoModel.from_pretrained(roi_encoder_ckpt)
        
        # Set similarity function
        if self.similarity_function == "dot":
            self.similarity_function = dot_product_scores
            
        elif self.similarity_function == "cosine":
            self.similarity_function = cosine_scores
            
        else:
            raise AttributeError(
                f"The similarity function can only be 'dot' or 'cosine', not '{self.similarity_function}'"
            )

    @classmethod
    def from_argparse_args(cls, args: Namespace, num_batches_per_epoch: int):

        multimodal_retriever = cls(
            lr=args.lr,
            min_lr=args.min_lr,
            max_lr=args.max_lr,
            warmup_ratio=args.warmup_ratio,
            optimizer=args.optimizer,
            roi_encoder_ckpt=args.roi_encoder_ckpt,
            max_epochs=args.max_epochs,
            similarity_function=args.similarity_function,
            modality=args.modality,
            batch_size=args.batch_size,
            num_batches_per_epoch=num_batches_per_epoch,
            num_neg_rois=args.num_neg_rois,
            precision=args.precision,
            grad_clip_val=args.grad_clip_val,
            acc_grad_batches=args.acc_grad_batches,
            metric_at_k=args.metric_at_k
        )

        return multimodal_retriever
    
    def get_schedule_linear(
        self,
        optimizer,
        warmup_steps,
        total_training_steps,
        steps_shift=0,
        last_epoch=-1,
    ):
        """Taken from: https://github.com/facebookresearch/DPR/blob/a31212dc0a54dfa85d8bfa01e1669f149ac832b7/dpr/utils/model_utils.py#L106
        It creates a schedule with a learning rate that decreases linearly after linearly increasing during a warmup period.
        """

        def lr_lambda(current_step):
            current_step += steps_shift
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                self.min_lr,
                float(total_training_steps - current_step) / float(max(1, total_training_steps - warmup_steps)),
            )

        lambda_lr = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_lambda,
            last_epoch=last_epoch,
            verbose=True
        )
        
        return lambda_lr

    def configure_optimizers(self):
        """
        Configure optimizer and Cosine Annealing Learning Rate scheduler
        """
        # Optimizer
        if self.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # when running on the GPU cluster
        elif self.optimizer.lower() == "ranger":
            optimizer = Ranger21(
                params=self.parameters(),
                lr=self.lr,
                num_epochs=self.max_epochs,
                num_batches_per_epoch=self.num_batches_per_epoch,
                use_warmup=True,
                warmdown_active=True,
                warmdown_min_lr=1e-5,
                weight_decay=1e-4
            )
        else:
            raise AssertionError(
                "Currently only 'ranger' and 'adam' are supported as optimizers"
            )

        # Define learning rate scheduler
        scheduler = self.get_schedule_linear(
            optimizer=optimizer,
            warmup_steps=self.warmup_steps,
            total_training_steps=self.total_iters
        )
        optimizer.step()
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, metric):
        """
        Activate the LR scheduler by performing step
        """
        scheduler.step()

    def _reshape_batch(self, batch: dict[Tensor]) -> dict[Tensor]:
        """
        Data comes in as [batch_size, num_pos+num_neg, seq_len, dim], so then reshape it to [batch_size*num_passages, seq_len, dim]
        Args:
            batch: dict, containing tokenized inputs in shape [batch_size, num_pos+num_neg, seq_len, dim]
        Returns:
            reshaped_batch: dict, containing tokenized inputs in shape [batch_size*num_passages, seq_len, dim] -> [seq_len, B*num_passages]
        """
        reshaped_batch = {}
        for key, value in batch.items():
            if key in [QUESTION_INPUT_IDS, QUESTION_TOKEN_TYPE_IDS, QUESTION_ATTENTION_MASK, ROIS_INPUT_IDS, ROIS_ATTENTION_MASK]:
                batch_size, dim, seq_length = value.shape
                reshaped_batch[key] = value.view(-1, seq_length)
            elif key == ROIS_PIXEL_VALUES:
                batch_size, num_samples, num_channels, width, height = value.shape
                reshaped_batch[key] = value.view(batch_size * num_samples, num_channels, width, height)
            elif key == ROIS_BBOX:
                batch_size, num_samples, seq_length, dimension = value.shape
                reshaped_batch[key] = value.view(batch_size * num_samples, seq_length, dimension)
            elif key == LABELS:
                reshaped_batch[key] = value

        return reshaped_batch

    # def encode_question(self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor) -> Tensor:
    def encode_question(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Apply question encoder on tokenized question inputs to obtain its embedding representation
        Note: pooler_output is not indexed here, but later in the forward method where this method is called
        Args:
            input_ids: tensor, containing the input_ids from the question tokenizer
            attention_mask: tensor, containing the attention_mask from the question tokenizer
            token_type_ids: tensor, containing the token_type_ids from the question tokenizer
        Returns:
            tensor, containing the question's embedding representation in shape [batch_size*num_passages, dim]
        """
        question_encoding = self.roi_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        return question_encoding
    
    def encode_text_only_roi(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Encode the roi for text-only by passing only input_ids and attention_masks
        Args:
            input_ids: tensor, containing the input_ids from the roi tokenizer
            attention_mask: tensor, containing the attention_mask from the roi tokenizer
        Returns:
            tensor, containing the roi's embedding representation in shape [batch_size*num_passages, dim]
        """
        roi_encoding = self.roi_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=None,
            pixel_values=None
        )
        return roi_encoding
    
    def encode_vision_only_roi(self, pixel_values: Tensor) -> Tensor:
        """
        Encode the roi for vision-only by passing only pixel_values
        Args:
            pixel_values: tensor, containing the pixel_values from the roi tokenizer
        Returns:
            tensor, containing the roi's embedding representation in shape [batch_size*num_passages, dim]
        """
        roi_encoding = self.roi_encoder(
            input_ids=None,
            attention_mask=None,
            bbox=None,
            pixel_values=pixel_values
        )
        return roi_encoding
    
    def encode_multimodal_roi(self, input_ids: Tensor, attention_mask: Tensor, bbox: Tensor, pixel_values: Tensor) -> Tensor:
        """
        Encode the roi for multimodality by passing all the tokenized inputs
        Args:
            input_ids: tensor, containing the input_ids from the roi tokenizer
            attention_mask: tensor, containing the attention_mask from the roi tokenizer
            bbox: tensor, containing the processed bbox from the roi tokenizer
            pixel_values: tensor, containing the pixel_values from the roi tokenizer
        Returns:
            tensor, containing the roi's embedding representation in shape [batch_size*num_passages, dim
        """
        roi_encoding = self.roi_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values
        )
        return roi_encoding
    
    def get_sim_scores(self, q_encoding: Tensor, roi_encodings: Tensor) -> Tensor:
        """
        This method is used to calculate the logits (unnormalized similarity scores between question and ROIs embeddings) from the embeddings,
        by appying the similarity function on the inputs embeddings (dot product or cosine similarity)
        Args:
            q_encoding: tensor, containing the embedding representation of the questions
            roi_encodings: tensor, containing the embedding representation of the rois
        Returns:
            tensor, containing similarity scores (distance) between the embeddings
        """
        
        # Calculate similarity scores
        sim_scores = self.similarity_function(
            q_encoding=q_encoding,
            roi_encodings=roi_encodings
        )
        
        return sim_scores

    def forward(self, batch: dict[Tensor]) -> tuple[Tensor, Tensor]:
        """
        Forward pass the data in batch, reshape the batch, and pass data through the encoders, encode and output embedding representations (pooler_output and last_hidden_state)
        Args:
            batch, dict of tensors containing the required tokenized inputs to be encoded
        Returns:
            tuple, containing the question encodings (pooler output) and roi encodings (last hidden state, representation at CLS token)
        """
        
        # before reshaping: [batch_size, num_pos+num_neg, seq_len, dim]
        reshaped_batch = self._reshape_batch(batch=batch)
        # after reshaping: [batch_size*num_passages, seq_len, dim]

        # Encode the question
        question_encoding = self.encode_question(
            input_ids=reshaped_batch[QUESTION_INPUT_IDS],
            attention_mask=reshaped_batch[QUESTION_ATTENTION_MASK]
        )

        # Encode rois
        # text-only
        if "text" in self.modality:
            rois_encoding = self.encode_text_only_roi(
                input_ids=reshaped_batch[ROIS_INPUT_IDS],
                attention_mask=reshaped_batch[ROIS_ATTENTION_MASK]
            )
        # vision-only
        elif "vision" in self.modality:
            rois_encoding = self.encode_vision_only_roi(pixel_values=reshaped_batch[ROIS_PIXEL_VALUES])
        # multimodal
        else:
            rois_encoding = self.encode_multimodal_roi(
                input_ids=reshaped_batch[ROIS_INPUT_IDS],
                attention_mask=reshaped_batch[ROIS_ATTENTION_MASK],
                bbox=reshaped_batch[ROIS_BBOX],
                pixel_values=reshaped_batch[ROIS_PIXEL_VALUES]
            )

        # Index ROI and question representations at CLS token
        cls_question_encoding = question_encoding.last_hidden_state[:, 0, :]
        cls_rois_encoding = rois_encoding.last_hidden_state[:, 0, :]
        
        # Calculate similarity scores, resulting in [num_passages, batch_sizes] similarity scores
        sim_scores = self.get_sim_scores(
            q_encoding=cls_question_encoding,
            roi_encodings=cls_rois_encoding
        )

        # Reshape sim scores to [batch_size, num_passages]
        reshaped_sim_scores = torch.transpose(
            input=sim_scores,
            dim0=0,
            dim1=1
        )

        # Transpose to match the shape of the labels
        transposed_sim_scores = torch.transpose(
            input=reshaped_sim_scores[:self.num_passages],
            dim0=0,
            dim1=1
        )

        return transposed_sim_scores
    
    def calc_nll_loss(self, sim_scores: Tensor, labels: Tensor) -> Tensor:
        """
        Method to calculate the loss; applies the following:
            - flatten inputs
            - apply logsoftmax
            - calculates negative log likelihood loss between sim scores and target labels
            - based on https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss
        Args:
            sim_scores: tensor, containing the similarity scores between the embeddings
            labels: tensor, contains ground truth labels
        Returns:
            tensor, containing loss value between the computed similarity scores and the target labels
        """
        
        # Initialize softmax activation and nll loss functions
        log_softmax_activation = torch.nn.LogSoftmax(dim=1)
        nll_loss = torch.nn.NLLLoss()
        
        # Apply torch negative log-likelihood loss
        loss = nll_loss(
            input=log_softmax_activation(sim_scores).flatten(),
            target=labels.flatten()
        )

        return loss

    def training_step(self, batch: dict[Tensor], batch_idx: int) -> Tensor:
        """
        Data comes in as: [batch_size, num_pos+num_neg, seq_len, dim]
        Forward data through model (reshape, get emebddings, calculate similarity scores)
        Compute loss per sample in batch and add to training_step_outputs
        Args:
            batch: dict of tensors, containing tokenized inputs to encode
        Returns,
            train_loss: tensor, containing train step loss value
        """
        
        # Forward pass to get embeddings and similarity scores
        sim_scores = self(batch)

        # Compute cross-entropy per sample in B as per DPR paper
        train_loss = self.calc_nll_loss(
            sim_scores=sim_scores,
            labels=batch[LABELS]
        )
                
        # Add loss to the validation_step_outputs
        self.training_step_outputs.append(train_loss)
        
        # Logging
        self.log(
            name="train_loss",
            value=train_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            batch_size=self.batch_size
        )
        
        return train_loss
    
    def on_train_epoch_end(self):
        """
        Used to calculate the average loss at the end of a training epoch and log it
        """
        train_losses = torch.stack(self.training_step_outputs)

        # Get average over losses
        avg_epoch_train_losses = train_losses.mean()
        
        # Log it
        self.log(
            name="train_loss_epoch",
            value=avg_epoch_train_losses,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            batch_size=self.batch_size
        )

        # Clear list of outputs
        self.training_step_outputs.clear()

    def validation_step(self, batch: dict[Tensor], batch_idx: int) -> Tensor:
        """
        Data comes in as: [batch_size, num_pos+num_neg, seq_len, dim]
        Forward data through model (reshape, get emebddings, calculate similarity scores)
        Compute loss per sample in batch and add to validation_step_outputs
        Args:
            batch: dict of tensors, containing tokenized inputs to encode
        Returns,
            val_loss: tensor, containing val step loss value
        """
        
        # Forward pass to get embeddings and similarity scores
        sim_scores = self(batch)
        
        # Compute cross-entropy per sample in B as per DPR paper
        val_loss = self.calc_nll_loss(
            sim_scores=sim_scores,
            labels=batch[LABELS]
        )
        
        # Add loss to the validation_step_outputs
        self.validation_step_outputs.append(val_loss)
        
        # Logging
        self.log(
            name="val_loss",
            value=val_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            batch_size=self.batch_size
        )
        
        return val_loss
    
    def on_validation_epoch_end(self):
        """
        Used to calculate the average loss at the end of a training epoch and log it
        """
        val_losses = torch.stack(self.validation_step_outputs)

        # Get average over losses
        avg_epoch_val_losses = val_losses.mean()
        
        # Log it
        self.log(
            name="val_loss_epoch",
            value=avg_epoch_val_losses,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            batch_size=self.batch_size
        )

        # Clear list of outputs
        self.validation_step_outputs.clear()
    
    def test_step(self, batch: dict[Tensor], batch_idx: int) -> Tensor:
        """
        Data comes in as: [batch_size, num_pos+num_neg, seq_len, dim]
        Forward data through model (reshape, get emebddings, calculate similarity scores)
        Calculate retrieval metrics and log them
        Args:
            batch: dict of tensors, containing tokenized inputs to encode
        """
        
        # Forward pass to get embeddings and get similarity scores
        sim_scores = self(batch)
        
        # Prepare preds and targets
        sim_scores = sim_scores.flatten()
        labels = batch[LABELS].flatten()
        
        # Calculate retrieval metrics       
        prec = precision(
            preds=sim_scores,
            target=labels,
            k=self.metric_at_k
        )
        rec = recall(
            preds=sim_scores,
            target=labels,
            k=self.metric_at_k
        )
        hr = hit_rate(
            preds=sim_scores,
            target=labels,
            k=self.metric_at_k
        )
        mrr = mean_reciprocal_rank(
            preds=sim_scores,
            target=labels,
            k=self.metric_at_k
        )
        ndcg = normalized_dcg(
            preds=sim_scores,
            target=labels,
            k=self.metric_at_k
        )
        
        batch_level_metrics = {
            f"precision@{self.metric_at_k}": prec,
            f"recall@{self.metric_at_k}": rec,
            f"hit_rate@{self.metric_at_k}": hr,
            f"mean_reciprocal_rank@{self.metric_at_k}": mrr,
            f"norm_dcg@{self.metric_at_k}": ndcg
        }
        
        # Log metrics
        self.log_dict(
            dictionary=batch_level_metrics,
            prog_bar=True,
            batch_size=self.batch_size
        )
        
        # Append to list of test_step outputs
        self.test_step_outputs.append(batch_level_metrics)
        
        return batch_level_metrics        
        
    def on_test_epoch_end(self):
        """
        Used to calculate the evaluation metrics values at the corpus level (aggregated from each batch)
        """
        
        # Initialize dictionaries to store the summed tensor metrics
        sum_metrics = {}

        # Iterate through each dictionary in the list
        for batch_level_metrics in self.test_step_outputs:
            for metric, value in batch_level_metrics.items():
                if metric in sum_metrics:
                    sum_metrics[metric] += value
                else:
                    sum_metrics[metric] = value

        # Calculate the mean values by dividing the summed tensors by the number of data points
        num_data_points = len(self.test_step_outputs)
        corpus_level_metrics = {
            metric: sum_val / num_data_points
            for metric, sum_val in sum_metrics.items()
        }

        print(f"\nCorpus-level metrics:\n\t{corpus_level_metrics}\n")
        self.corpus_level_metrics = corpus_level_metrics
        
        # Log metrics
        self.log_dict(
            dictionary=corpus_level_metrics,
            prog_bar=True
        )
        
        # Clear list of outputs
        self.test_step_outputs.clear()

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("MultiModalRetriever")
        parser.add_argument(
            "--lr",
            help="Learning rate",
            type=float
        )
        parser.add_argument(
            "--min-lr",
            help="Minumum learning rate for Linear Warmup scheduler",
            type=float
        )
        parser.add_argument(
            "--max-lr",
            help="Maximum learning rate for Linear Warmup scheduler",
            type=float
        )
        parser.add_argument(
            "--warmup-ratio",
            help="Warmup ratio to use for the Linear Warmup scheduler",
            type=float
        )
        parser.add_argument(
            "--optimizer",
            help="Optimizer",
            type=str
        )
        parser.add_argument(
            "--question-encoder-ckpt",
            help="Checkpoint path for question encoder",
            type=str
        )
        parser.add_argument(
            "--roi-encoder-ckpt",
            help="Checkpoint path for roi encoder",
            type=str
        )
        parser.add_argument(
            "--max-epochs",
            help="Max epochs to train for",
            type=int
        )
        parser.add_argument(
            "--similarity-function",
            help="The similarity function used to compare embeddings distance",
            type=str
        )
        parser.add_argument(
            "--precision",
            help="Precision for computational speedup",
            type=str
        )
        parser.add_argument(
            "--grad-clip-val",
            help="Amount to limit magnitude of gradients to, improving stability of optimization",
            type=float
        )
        parser.add_argument(
            "--acc-grad-batches",
            help="Amount of batches used to accumulate the gradients on",
            type=int
        )
        parser.add_argument(
            "--metric-at-k",
            help="The amount of ROIs to consider for calculating metrics, such as recall @ k",
            type=int
        )

        return parent_parser

    
# SIMILARITY FUNCTIONS AND CROSS ENTROPY LOSS (negative-log likelihood) FUNCTION
def dot_product_scores(q_encoding: Tensor, roi_encodings: Tensor) -> Tensor:
    """
    This function calculates the dot product similarity score (which indicates how close or far two vectors are in an embedding space)
    between the questions embeddings and the rois embeddings
    Args:
        q_encoding: tensor, containing the embedding representation of the questions
        roi_encodings: tensor, containing the embedding representation of the rois
    Returns:
        tensor, containing dot-product similarity scores (distance) between the embeddings
    """
    
    dot_sim_scores = torch.matmul(
        input=q_encoding,
        other=torch.transpose(
            input=roi_encodings,
            dim0=0,
            dim1=1
        )
    )
    
    return dot_sim_scores

def cosine_scores(q_encoding: Tensor, roi_encodings: Tensor) -> Tensor:
    """
    This function calculates the cosine distance similarity score (which also indicates how close or far two vectors are in an embedding space),
    between the questions embeddings and the rois embeddings
    Args:
        q_encoding: tensor, containing the embedding representation of the questions
        roi_encodings: tensor, containing the embedding representation of the rois
    Returns:
        tensor, containing cosine similarity scores (distance) between the embeddings
    """
    cosine_similarity_scores = F.cosine_similarity(
        x1=q_encoding,
        x2=roi_encodings,
        dim=1
    )
    
    return cosine_similarity_scores
