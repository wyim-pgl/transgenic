import sys
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoConfig, PreTrainedModel, AutoModel, AutoModelForImageSegmentation
from transformers import LEDForConditionalGeneration, EsmForMaskedLM, T5ForConditionalGeneration
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from configuration_transgenic import TransgenicHyenaConfig


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
	"""
	Shift input ids one token to the right.
	"""
	shifted_input_ids = input_ids.new_zeros(input_ids.shape)
	shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
	shifted_input_ids[:, 0] = decoder_start_token_id

	if pad_token_id is None:
		raise ValueError("config.pad_token_id has to be defined.")
	# replace possible -100 values in labels by `pad_token_id`
	shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

	return shifted_input_ids

#Copied from transformers.models.led.modeling_led.py
@dataclass
class LEDSeq2SeqLMOutput(ModelOutput):
	"""
	Base class for sequence-to-sequence language models outputs.

	Args:
		loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
			Language modeling loss.
		logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
			Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
		past_key_values (`List[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
			List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size,
			num_heads, sequence_length, embed_size_per_head)`).

			Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
			used (see `past_key_values` input) to speed up sequential decoding.
		decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
			Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
			shape `(batch_size, sequence_length, hidden_size)`.

			Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
		decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
			Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
			sequence_length)`.

			Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
			self-attention heads.
		cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
			Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
			sequence_length)`.

			Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
			weighted average in the cross-attention heads.
		encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
			Sequence of hidden-states at the output of the last layer of the encoder of the model.
		encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
			Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
			shape `(batch_size, sequence_length, hidden_size)`.

			Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
		encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
			Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
			sequence_length)`.

			Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
			self-attention heads.
		encoder_global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
			Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
			where `x` is the number of tokens with global attention mask.

			Global attentions weights after the attention softmax, used to compute the weighted average in the
			self-attention heads. Those are the attention weights from every token with global attention to every token
			in the sequence.
	"""

	loss: Optional[torch.FloatTensor] = None
	logits: torch.FloatTensor = None
	past_key_values: Optional[List[torch.FloatTensor]] = None
	decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
	decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	encoder_last_hidden_state: Optional[torch.FloatTensor] = None
	encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
	encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	encoder_global_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	segmentation_logits: Optional[Tuple[torch.FloatTensor, ...]] = None

#Copied from transformers.models.led.modeling_led.py
@dataclass
class LEDSeq2SeqModelOutput(ModelOutput):
	"""
	Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
	decoding.

	Args:
		last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
			Sequence of hidden-states at the output of the last layer of the decoder of the model.

			If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
			hidden_size)` is output.
		past_key_values (`List[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
			List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size,
			num_heads, sequence_length, embed_size_per_head)`).

			Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
			used (see `past_key_values` input) to speed up sequential decoding.
		decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
			Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
			shape `(batch_size, sequence_length, hidden_size)`.

			Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
		decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
			Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
			sequence_length)`.

			Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
			self-attention heads.
		cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
			Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
			sequence_length)`.

			Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
			weighted average in the cross-attention heads.
		encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
			Sequence of hidden-states at the output of the last layer of the encoder of the model.
		encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
			Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
			shape `(batch_size, sequence_length, hidden_size)`.

			Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
		encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
			Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
			sequence_length)`.

			Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
			self-attention heads.
		encoder_global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
			Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
			where `x` is the number of tokens with global attention mask.

			Global attentions weights after the attention softmax, used to compute the weighted average in the
			self-attention heads. Those are the attention weights from every token with global attention to every token
			in the sequence.
	"""

	last_hidden_state: torch.FloatTensor = None
	past_key_values: Optional[List[torch.FloatTensor]] = None
	decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
	decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	encoder_last_hidden_state: Optional[torch.FloatTensor] = None
	encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
	encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	encoder_global_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	segmentation_logits: Optional[Tuple[torch.FloatTensor, ...]] = None

@dataclass
class HyenaModelOutput(ModelOutput):
	last_hidden_state: torch.FloatTensor = None
	attention_mask: torch.FloatTensor = None
	encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
	segmentation_logits: Optional[Tuple[torch.FloatTensor, ...]] = None
	segmentation_loss: Optional[Tuple[torch.FloatTensor, ...]] = None

class TransgenicPreTrainedModel(PreTrainedModel):
	config_class = TransgenicHyenaConfig
	base_model_prefix = "led"
	supports_gradient_checkpointing = True

	def _init_weights(self, module):
		std = self.config.init_std
		if isinstance(module, nn.Linear):
			module.weight.data.normal_(mean=0.0, std=std)
			if module.bias is not None:
				module.bias.data.zero_()
		elif isinstance(module, nn.Embedding):
			module.weight.data.normal_(mean=0.0, std=std)
			if module.padding_idx is not None:
				module.weight.data[module.padding_idx].zero_()

	@property
	def dummy_inputs(self):
		pad_token = self.config.pad_token_id
		input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
		dummy_inputs = {
			"attention_mask": input_ids.ne(pad_token),
			"input_ids": input_ids,
		}
		return dummy_inputs

class DynamicProportionalPooling(nn.Module):
	def __init__(self, padding_value=0.0):
		"""
		Dynamic proportional pooling layer with fixed reduction by a factor of 3.

		Args:
			padding_value (float): Value used for padding at the beginning of the sequence.
		"""
		super(DynamicProportionalPooling, self).__init__()
		self.padding_value = padding_value

	def forward(self, embedding, attention_mask):
		"""
		Forward pass for dynamic proportional pooling with a fixed reduction factor.

		Args:
			embedding (torch.Tensor): Input tensor of shape (batch_size, seq_length, hidden_dim).
			attention_mask (torch.Tensor): Input tensor of shape (batch_size, seq_length).
		
		Returns:
			Tuple: Output tuple of pooled tensors (pooled_embedding, pooled_mask).
		"""
		if type(embedding) == HyenaModelOutput:
			embedding = embedding.last_hidden_state
		batch_size, seq_length, hidden_dim = embedding.size()

		# Calculate required padding to ensure divisibility by 3
		pad_length = (3 - (seq_length % 3)) % 3

		# Add padding at the beginning if needed
		if pad_length > 0:
			embedding = nn.functional.pad(embedding, (0,0, pad_length, 0), value=self.padding_value)
			attention_mask = nn.functional.pad(attention_mask, (pad_length,0, 0, 0), value=self.padding_value)

		# Apply AvgPool1d
		pool_embed = nn.AvgPool1d(kernel_size=3, stride=3)
		pool_mask = nn.MaxPool1d(kernel_size=3, stride=3)
		
		pooled_embed = pool_embed(embedding.permute(0, 2, 1)).permute(0, 2, 1)
		pooled_mask = pool_mask(attention_mask.float())

		return (pooled_embed, pooled_mask)

class HyenaDownsample(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=6, stride=6):
		super(HyenaDownsample, self).__init__()
		self.conv1 = nn.Conv1d(in_channels, out_channels // 2, kernel_size=kernel_size//2, padding=1)
		self.conv2 = nn.Conv1d(out_channels // 2, out_channels, kernel_size=kernel_size//2, padding=1)
		self.downsample = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = self.downsample(x)
		return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet1D(nn.Module):
	def __init__(self, input_dim=256, hidden_dim=256, num_classes=9):
		super(UNet1D, self).__init__()
		
		# Encoder
		self.enc1 = nn.Sequential(
			nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
			nn.ReLU()
		)
		self.pool1 = nn.MaxPool1d(2)
		
		self.enc2 = nn.Sequential(
			nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv1d(hidden_dim*2, hidden_dim*2, kernel_size=3, padding=1),
			nn.ReLU()
		)
		self.pool2 = nn.MaxPool1d(2)
		
		# Bottleneck
		self.bottleneck = nn.Sequential(
			nn.Conv1d(hidden_dim*2, hidden_dim*4, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv1d(hidden_dim*4, hidden_dim*4, kernel_size=3, padding=1),
			nn.ReLU()
		)
		
		# Decoder
		self.up2 = nn.ConvTranspose1d(hidden_dim*4, hidden_dim*2, kernel_size=2, stride=2)
		self.dec2 = nn.Sequential(
			nn.Conv1d(hidden_dim*4, hidden_dim*2, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv1d(hidden_dim*2, hidden_dim*2, kernel_size=3, padding=1),
			nn.ReLU()
		)
		
		self.up1 = nn.ConvTranspose1d(hidden_dim*2, hidden_dim, kernel_size=2, stride=2)
		self.dec1 = nn.Sequential(
			nn.Conv1d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
			nn.ReLU()
		)
		
		# Final classification layer
		self.final = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)
	
	def forward(self, x):
		# Encoder
		enc1 = self.enc1(x)  # [batch, hidden_dim, seq_len]
		pooled1 = self.pool1(enc1)  # [batch, hidden_dim, 128]
		
		enc2 = self.enc2(pooled1)  # [batch, hidden_dim*2, 128]
		pooled2 = self.pool2(enc2)  # [batch, hidden_dim*2, 64]
		
		# Bottleneck
		bottleneck = self.bottleneck(pooled2)  # [batch, hidden_dim*4, 64]
		
		# Decoder
		up2 = self.up2(bottleneck)  # [batch, hidden_dim*2, 128]
		dec2 = self.dec2(torch.cat([up2, enc2], dim=1))  # [batch, hidden_dim*2, 128]
		
		up1 = self.up1(dec2)  # [batch, hidden_dim, seq_len]
		dec1 = self.dec1(torch.cat([up1, enc1], dim=1))  # [batch, hidden_dim, seq_len]
		
		# Final classification layer
		out = self.final(dec1)  # [batch, num_classes, seq_len]
		
		# Transpose to [batch, seq_len, num_classes]
		return out.permute(0, 2, 1)

import torch
import torch.nn as nn

class DownSample1D(nn.Module):
	def __init__(self, in_channels, out_channels, dropout_rate=0.2):
		super().__init__()
		self.conv_layers = nn.Sequential(
			nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
			nn.SiLU(),
			nn.Dropout1d(p=dropout_rate),  # Dropout after activation
			nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
			nn.SiLU(),
			nn.Dropout1d(p=dropout_rate)  # Another dropout for deeper regularization
		)
		self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)

	def forward(self, x):
		x = self.conv_layers(x)
		x_pooled = self.avg_pool(x)
		return x_pooled, x  # Returning pooled output + skip connection

class UpSample1D(nn.Module):
	def __init__(self, in_channels, skip_channels, out_channels, dropout_rate=0.2):
		super().__init__()
		self.conv_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
		self.conv_layers = nn.Sequential(
			nn.Conv1d(out_channels + skip_channels, out_channels, kernel_size=3, stride=1, padding=1),
			nn.SiLU(),
			nn.Dropout1d(p=dropout_rate),  # Dropout after first activation
			nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
			nn.SiLU(),
			nn.Dropout1d(p=dropout_rate)  # Dropout after second activation
		)

	def forward(self, x, skip):
		x = self.conv_transpose(x)  # Upsample
		x = torch.cat([x, skip], dim=1)  # Concatenate with skip connection
		x = self.conv_layers(x)  # Convolutions
		return x

class FinalConv1D(nn.Module):
	def __init__(self, in_channels, out_channels, dropout_rate=0.2):
		super().__init__()
		self.conv_layers = nn.Sequential(
			nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
			nn.SiLU(),
			nn.Dropout1d(p=dropout_rate),  # Dropout before final layer
			nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
		)

	def forward(self, x):
		return self.conv_layers(x)

class UNet1DSegmentationHead(nn.Module):
	def __init__(self, num_classes=9, dropout_rate=0.2):
		super().__init__()

		self._downsample_blocks = nn.ModuleList([
			DownSample1D(256, 512, dropout_rate),
			DownSample1D(512, 1024, dropout_rate),
			DownSample1D(1024, 2048, dropout_rate)
		])
		
		self.conv_layers = nn.Sequential(
			nn.Conv1d(2048, 4096, kernel_size=3, stride=1, padding=1),
			nn.SiLU(),
			nn.Dropout1d(p=dropout_rate),  # Dropout after activation
			nn.Conv1d(4096, 4096, kernel_size=3, stride=1, padding=1),
			nn.SiLU(),
			nn.Dropout1d(p=dropout_rate)  # Another dropout for deeper regularization
		)

		self._upsample_blocks = nn.ModuleList([
			UpSample1D(4096, 2048, 2048, dropout_rate),
			UpSample1D(2048, 1024, 1024, dropout_rate),
			UpSample1D(1024, 512, 512, dropout_rate)
		])

		self.final_block = FinalConv1D(512, num_classes, dropout_rate)
		self.dropout = nn.Dropout1d(p=dropout_rate)  # Global dropout for extra regularization

	def forward(self, x):
		skips = []
		for down in self._downsample_blocks:
			x, skip = down(x)
			x = self.dropout(x)  # Dropout in encoder
			skips.append(skip)

		skips = skips[::-1]  # Reverse for upsampling

		x = self.conv_layers(x)

		for i, up in enumerate(self._upsample_blocks):
			x = up(x, skips[i])
			x = self.dropout(x)  # Dropout in decoder

		x = self.final_block(x)
		return x


def init_weights(m):
	if isinstance(m, nn.Conv1d):
		# Kaiming Initialization for Conv1d layers
		nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)
	
	elif isinstance(m, nn.ConvTranspose1d):
		# Bilinear Initialization for ConvTranspose1d
		nn.init.xavier_normal_(m.weight)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)
	
	elif isinstance(m, nn.BatchNorm1d):
		# Constant Initialization for BatchNorm layers
		nn.init.constant_(m.weight, 1)
		nn.init.constant_(m.bias, 0)
	
	elif isinstance(m, nn.Linear):
		# Xavier Initialization for Linear layers
		nn.init.xavier_normal_(m.weight)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)

class FocalLoss(nn.Module):
	def __init__(self, alpha=0.25, gamma=2):
		super(FocalLoss, self).__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.bce = nn.BCEWithLogitsLoss(reduction='none')

	def forward(self, logits, targets):
		bce_loss = self.bce(logits, targets)
		probas = torch.sigmoid(logits)
		pt = probas * targets + (1 - probas) * (1 - targets)  # Prob of true class
		loss = self.alpha * (1 - pt) ** self.gamma * bce_loss  # Focal loss formula
		return loss.mean()

class WeightedFocalLoss(nn.Module):
	"Non weighted version of Focal Loss"
	def __init__(self, alpha=None, gamma=None, pos_weight=None):
		super(WeightedFocalLoss, self).__init__()
		if alpha == None: # Set positive class alpha values, negative values are 1-alpha
			self.alpha = torch.tensor([0.7500, 0.9950, 0.8750, 0.8750, 0.9900, 0.9900, 0.9688, 0.9688, 0.9950])
		else:
			self.alpha = alpha
		if gamma == None:
			self.gamma = torch.tensor([2, 2, 2, 2, 2, 2, 2, 2, 2])
		else:
			self.gamma = gamma
		self.pw = pos_weight

	def forward(self, inputs, targets):
		#self.alpha = self.alpha.to(inputs.device)
		self.pw = self.pw.to(inputs.device)
		self.gamma = self.gamma.to(inputs.device)
		BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.pw)
		pt = torch.sigmoid(inputs) * targets + (1 - torch.sigmoid(inputs)) * (1 - targets) # probability of ground truth class
		pt = torch.clamp(pt, min=1e-8, max=1 - 1e-8)
		#at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
		F_loss = (1-pt)**self.gamma * BCE_loss
		return F_loss.mean()

class DiceLoss(nn.Module):
	def __init__(self, smooth=1e-6):
		super(DiceLoss, self).__init__()
		self.smooth = smooth  # Prevent division by zero

	def forward(self, y_pred, y_true):
		# Ensure predictions are between 0 and 1
		y_pred = torch.sigmoid(y_pred)
		y_true = y_true.float()

		dice_loss = 0.0
		num_classes = y_true.size(2)  # Number of classes (channels)

		for c in range(num_classes):
			intersection = torch.sum(y_pred[:, :,c] * y_true[:,:, c])
			denominator = torch.sum(y_true[:,:, c]) + torch.sum(y_pred[:,:, c])

			dice_score = (2. * intersection + self.smooth) / (denominator + self.smooth)
			dice_loss += (1 - dice_score)  # Minimize Dice Loss for each class

		return dice_loss / num_classes  # Average over classes

class IoULoss(nn.Module):
	def __init__(self, smooth=1e-6):
		super(IoULoss, self).__init__()
		self.smooth = smooth  # Prevent division by zero

	def forward(self, y_pred, y_true):
		y_pred = torch.sigmoid(y_pred)  # Ensure predictions are between 0 and 1
		y_true = y_true.float()  # Ensure ground truth is float type

		num_classes = y_true.size(2)  # Number of classes (channels)
		iou_loss = 0.0

		for c in range(num_classes):
			intersection = torch.sum(y_pred[:,:, c] * y_true[:,:, c])
			union = torch.sum(y_pred[:,:, c]) + torch.sum(y_true[:,:, c]) - intersection

			iou_score = (intersection + self.smooth) / (union + self.smooth)
			iou_loss += (1 - iou_score)  # Minimize IoU Loss for each class

		return iou_loss / num_classes  # Average over all classes

class HyenaEncoder(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.do_segment = config.do_segment
		self.segmentation_model = config.s_model
		self.numSegClasses = config.numSegClasses

		HyenaConfig = AutoConfig.from_pretrained(config.encoder_model, trust_remote_code=True)
		HyenaConfig.max_seq_len = config.max_encoder_seqlen
		HyenaConfig.d_model = config.d_model
		HyenaConfig.n_layer = config.encoder_n_layer
		self.hyena = AutoModel.from_config(HyenaConfig, trust_remote_code=True)
		
		#HyenaConfig.n_layer = 3
		#self.segmentation_model = AutoModel.from_config(HyenaConfig, trust_remote_code=True).backbone.layers
		#self.segmentation_head = nn.Linear(config.d_model, config.numSegClasses)

		#self.segmentation_head  = UNet1D(input_dim=256, hidden_dim=256, num_classes=9)
		self.segmentation_head = UNet1DSegmentationHead(num_classes=9, dropout_rate=0.0)
		self.segmentation_head.apply(init_weights)
	
	def forward(self, input_ids, segLabels = None, *args, **kwargs):
		
		output = self.hyena(input_ids)
		
		if self.do_segment:
			seg_logits = self.segmentation_head(output.last_hidden_state.permute(0,2,1)).permute(0,2,1)

			if segLabels != None:

				# Define weight of positive splice junction classes to counteract class imbalance
				#pos_weight = torch.ones(self.numSegClasses)
				#pos_weight[[1,8]] = 10.0
				#pos_weight[[4,5]] = 5.0
				#pos_weight[[2,3,6,7]] = 2.0
				pos_weight = torch.tensor([2.4, 4012, 6, 6, 1586, 1578, 46, 30, 4044])
				pos_weight = pos_weight.to(segLabels.device)

				# Define per-nucleotide weights to focus on basic genic features
				#weight = torch.Tensor((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
				#weight = weight.to(segLabels.device)

				# Define the loss function with pos_weight
				#segLossFn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
				#segLossFn = FocalLoss(alpha=0.5, gamma=2)
				segLossFn = WeightedFocalLoss( 
					pos_weight = torch.tensor([2.4, 4012, 6, 6, 1586, 1578, 46, 30, 4044]),
					gamma=torch.tensor([2, 2, 2, 2, 2, 2, 2, 2, 2]))
				#segLossFn=IoULoss()
				#boundaryLossFn = BoundaryLoss()
				seg_loss = segLossFn(seg_logits, segLabels)
				#boundary_loss = boundaryLossFn(seg_logits[:,0:segLabels.shape[1]], segLabels)
				#seg_loss = None
				boundary_loss = None
			else:
				seg_loss = None
		else:
			seg_logits = None
			seg_loss = None

		return HyenaModelOutput(
			last_hidden_state=output.last_hidden_state,
			encoder_hidden_states=output.hidden_states,
			segmentation_logits=seg_logits,
			segmentation_loss=seg_loss
		)

class transgenicModel(TransgenicPreTrainedModel):
	_tied_weights_keys = ["decoder_embed_tokens.weight", "decoder.embed_tokens.weight"]

	def __init__(self, config):
		super().__init__(config)

		padding_idx, vocab_size = config.pad_token_id, config.vocab_size
		#self.decoder_embed_tokens = NumberMaskEmbedTokens(vocab_size, config.d_model, padding_idx)
		self.decoder_embed_tokens = nn.Embedding(vocab_size, config.d_model*2, padding_idx)
		
		# Encoder model
		self.encoder = HyenaEncoder(config)

		# Compression
		#self.pool = DynamicProportionalPooling()
		self.downsample = HyenaDownsample(config.d_model, config.d_model*2, kernel_size=6, stride=6)
		
		# Decoder Model
		config.d_model = config.d_model * 2
		self.decoder = LEDForConditionalGeneration(config).led.decoder
		self.decoder.embed_tokens = self.decoder_embed_tokens

		# Initialize weights and apply final processing
		self.post_init()

	def get_input_embeddings(self):
		return self.decoder_embed_tokens

	def set_input_embeddings(self, value):
		self.decoder_embed_tokens = value
		self.decoder.embed_tokens = self.decoder_embed_tokens

	def get_encoder(self):
		return self.encoder

	def get_decoder(self):
		return self.decoder

	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		decoder_input_ids: Optional[torch.LongTensor] = None,
		decoder_attention_mask: Optional[torch.LongTensor] = None,
		head_mask: Optional[torch.Tensor] = None,
		decoder_head_mask: Optional[torch.Tensor] = None,
		cross_attn_head_mask: Optional[torch.Tensor] = None,
		encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
		global_attention_mask: Optional[torch.FloatTensor] = None,
		past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple[torch.Tensor], LEDSeq2SeqModelOutput]:
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		use_cache = use_cache if use_cache is not None else self.config.use_cache
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		# Using this like Bart, as LED is derived from it. So far
		# No checkpoint on the hub exists that uses that in practice.
		# https://github.com/huggingface/transformers/blob/ac3cb660cad283163f7c73cad511124e845ca388/src/transformers/models/bart/modeling_bart.py#L1153
		if decoder_input_ids is None and decoder_inputs_embeds is None:
			decoder_input_ids = shift_tokens_right(
				input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
			)
		# Compute the embeddings with encoder
		if encoder_outputs is None:
			encoder_outputs = self.encoder(input_ids)
			encoder_outputs.attention_mask = attention_mask
		else:
			encoder_outputs = HyenaModelOutput(
				last_hidden_state=encoder_outputs, 
				attention_mask=attention_mask
				)

		# Compress to a size that the decoder can handle
		#pooled = self.pool(encoder_outputs.last_hidden_state, encoder_outputs.attention_mask)
		downsampled = self.downsample(encoder_outputs.last_hidden_state.permute(0,2,1)).permute(0,2,1)
		attention_mask = torch.ones(downsampled.shape[0:2]).to(downsampled.device)

		# decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
		decoder_outputs = self.decoder(
			input_ids=decoder_input_ids,
			attention_mask=decoder_attention_mask,
			encoder_hidden_states=downsampled, #pooled[0],
			encoder_attention_mask=attention_mask, #pooled[1].long(),
			global_attention_mask=global_attention_mask,
			head_mask=decoder_head_mask,
			cross_attn_head_mask=cross_attn_head_mask,
			past_key_values=past_key_values,
			inputs_embeds=decoder_inputs_embeds,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		if not return_dict:
			return decoder_outputs + encoder_outputs

		return LEDSeq2SeqModelOutput(
			last_hidden_state=decoder_outputs.last_hidden_state,
			past_key_values=decoder_outputs.past_key_values,
			decoder_hidden_states=decoder_outputs.hidden_states,
			decoder_attentions=decoder_outputs.attentions,
			cross_attentions=decoder_outputs.cross_attentions,
			encoder_last_hidden_state=encoder_outputs.last_hidden_state,
			encoder_hidden_states=encoder_outputs.encoder_hidden_states,
			encoder_attentions=encoder_outputs.attention_mask,
			encoder_global_attentions=None,
			segmentation_logits = encoder_outputs.segmentation_logits,
		)

class transgenicForConditionalGeneration(TransgenicPreTrainedModel):
	base_model_prefix = "transgenic"
	_keys_to_ignore_on_load_missing = ["final_logits_bias"]
	_tied_weights_keys = ["transgenic.decoder_embed_tokens.weight", "lm_head.weight"]

	def __init__(self, config, unlink=False):
		if not unlink:
			_tied_weights_keys = []
		super().__init__(config)
		self.transgenic = transgenicModel(config)
		self.register_buffer("final_logits_bias", torch.zeros((1, self.transgenic.decoder_embed_tokens.num_embeddings)))
		self.lm_head = nn.Linear(config.d_model, self.transgenic.decoder_embed_tokens.num_embeddings, bias=False)

		# Initialize weights and apply final processing
		self.post_init()
		self.initialize_weights()

	def get_encoder(self):
		return self.transgenic.get_encoder()

	def get_decoder(self):
		return self.transgenic.get_decoder()

	def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
		new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
		self._resize_final_logits_bias(new_embeddings.weight.shape[0])
		return new_embeddings

	def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
		old_num_tokens = self.final_logits_bias.shape[-1]
		if new_num_tokens <= old_num_tokens:
			new_bias = self.final_logits_bias[:, :new_num_tokens]
		else:
			extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
			new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
		self.register_buffer("final_logits_bias", new_bias)

	def get_output_embeddings(self):
		return self.lm_head

	def set_output_embeddings(self, new_embeddings):
		self.lm_head = new_embeddings
	
	def initialize_weights(self):
		for m in self.transgenic.decoder.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
		#nn.init.xavier_uniform_(self.transgenic.encoder.hidden_mapping.weight)
		#nn.init.constant_(self.transgenic.encoder.hidden_mapping.bias, 0)

	#@add_start_docstrings_to_model_forward(LED_INPUTS_DOCSTRING)
	#@replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
	#@add_end_docstrings(LED_GENERATION_EXAMPLE)
	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		decoder_input_ids: Optional[torch.LongTensor] = None,
		decoder_attention_mask: Optional[torch.LongTensor] = None,
		head_mask: Optional[torch.Tensor] = None,
		decoder_head_mask: Optional[torch.Tensor] = None,
		cross_attn_head_mask: Optional[torch.Tensor] = None,
		encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
		global_attention_mask: Optional[torch.FloatTensor] = None,
		past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple[torch.Tensor], LEDSeq2SeqLMOutput]:
		r"""
		labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
			Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
			config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
			(masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

		Returns:

		Conditional generation example:

		```python
		>>> from transformers import AutoTokenizer, LEDForConditionalGeneration

		>>> tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
		>>> TXT = "My friends are <mask> but they eat too many carbs."

		>>> model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")
		>>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]

		>>> prediction = model.generate(input_ids)[0]
		>>> print(tokenizer.decode(prediction, skip_special_tokens=True))
		```"""
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		if labels is not None:
			if use_cache:
				print("The `use_cache` argument is changed to `False` since `labels` is provided.", file=sys.stderr)
			use_cache = False
			if decoder_input_ids is None and decoder_inputs_embeds is None:
				decoder_input_ids = shift_tokens_right(
					labels, self.config.pad_token_id, self.config.decoder_start_token_id
				)
		
		outputs = self.transgenic(
			input_ids,
			attention_mask=attention_mask,
			decoder_input_ids=decoder_input_ids,
			decoder_attention_mask=decoder_attention_mask,
			encoder_outputs=encoder_outputs,
			global_attention_mask=global_attention_mask,
			head_mask=head_mask,
			decoder_head_mask=decoder_head_mask,
			cross_attn_head_mask=cross_attn_head_mask,
			past_key_values=past_key_values,
			inputs_embeds=inputs_embeds,
			decoder_inputs_embeds=decoder_inputs_embeds,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

		if not return_dict:
			output = (lm_logits,) + outputs[1:]
			return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

		masked_lm_loss = None
		if labels is not None:
			#weights = torch.ones(50262).to(lm_logits.device)
			#weights[262:] = 2
			loss_fct = nn.CrossEntropyLoss()
			masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
			#masked_lm_loss = HybridLoss(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
			#loss_fct = HybridSequenceLoss()
			#masked_lm_loss = loss_fct(lm_logits, labels)

		return LEDSeq2SeqLMOutput(
			loss=masked_lm_loss,
			logits=lm_logits,
			past_key_values=outputs.past_key_values,
			decoder_hidden_states=outputs.decoder_hidden_states,
			decoder_attentions=outputs.decoder_attentions,
			cross_attentions=outputs.cross_attentions,
			encoder_last_hidden_state=outputs.encoder_last_hidden_state,
			encoder_hidden_states=outputs.encoder_hidden_states,
			encoder_attentions=outputs.encoder_attentions,
			encoder_global_attentions=outputs.encoder_global_attentions,
			segmentation_logits=outputs.segmentation_logits,
		)

	def prepare_inputs_for_generation(
		self,
		decoder_input_ids,
		past_key_values=None,
		attention_mask=None,
		global_attention_mask=None,
		head_mask=None,
		decoder_head_mask=None,
		cross_attn_head_mask=None,
		use_cache=None,
		encoder_outputs=None,
		**kwargs,
	):
		# cut decoder_input_ids if past is used
		if past_key_values is not None:
			decoder_input_ids = decoder_input_ids[:, -1:]

		return {
			"input_ids": None,  # encoder_outputs is defined. input_ids not needed
			"encoder_outputs": encoder_outputs.last_hidden_state,
			"past_key_values": past_key_values,
			"decoder_input_ids": decoder_input_ids,
			"attention_mask": attention_mask,
			"global_attention_mask": global_attention_mask,
			"head_mask": head_mask,
			"decoder_head_mask": decoder_head_mask,
			"cross_attn_head_mask": cross_attn_head_mask,
			"use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
		}

	def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
		return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

	@staticmethod
	def _reorder_cache(past_key_values, beam_idx):
		reordered_past = ()
		for layer_past in past_key_values:
			# cached cross_attention states don't have to be reordered -> they are always the same
			reordered_past += (
				tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
				+ layer_past[2:],
			)
		return reordered_past

class BiLSTMRegressionHead(nn.Module):
	def __init__(self, input_dim=256, hidden_dim=128):
		super(BiLSTMRegressionHead, self).__init__()
		self.bilstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
		self.fc = nn.Linear(hidden_dim * 2, 1)  # BiLSTM is bidirectional → output is 2 * hidden_dim
		self.sigmoid = nn.Sigmoid()  # To scale output between 0 and 1 (optional)

	def forward(self, x):
		# x: (batch, seq_len=200, channels=256)
		# BiLSTM Forward Pass
		lstm_out, _ = self.bilstm(x)  # (batch, seq_len, hidden_dim * 2)

		# Select the last timestep's output
		last_hidden_state = lstm_out[:, -1, :]  # (batch, hidden_dim * 2)

		# Fully connected layer for regression
		output = self.fc(last_hidden_state)  # (batch, 1)

		# Optionally scale the output to [0, 200] if using sigmoid
		output = self.sigmoid(output) * 200  # (batch, 1), ensures position is between 0-200

		return output.squeeze(1)  # Output shape: (batch,)


class transgenicModelT5(TransgenicPreTrainedModel):

	def __init__(self, config):
		super().__init__(config)
		# Encoder model
		self.encoder = HyenaEncoder(config)

		# Compression
		self.pool = DynamicProportionalPooling()

		# Decoder
		self.decoder = T5ForConditionalGeneration(config).decoder

		# Initialize weights and apply final processing
		self.post_init()

	def get_input_embeddings(self):
		return self.decoder.embed_tokens

	def set_input_embeddings(self, value):
		self.decoder.embed_tokens = value

	def get_encoder(self):
		return self.encoder

	def get_decoder(self):
		return self.decoder

	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		decoder_input_ids: Optional[torch.LongTensor] = None,
		decoder_attention_mask: Optional[torch.LongTensor] = None,
		head_mask: Optional[torch.Tensor] = None,
		decoder_head_mask: Optional[torch.Tensor] = None,
		cross_attn_head_mask: Optional[torch.Tensor] = None,
		encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
		global_attention_mask: Optional[torch.FloatTensor] = None,
		past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple[torch.Tensor], LEDSeq2SeqModelOutput]:
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		use_cache = use_cache if use_cache is not None else self.config.use_cache
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		# Using this like Bart, as LED is derived from it. So far
		# No checkpoint on the hub exists that uses that in practice.
		# https://github.com/huggingface/transformers/blob/ac3cb660cad283163f7c73cad511124e845ca388/src/transformers/models/bart/modeling_bart.py#L1153
		if decoder_input_ids is None and decoder_inputs_embeds is None:
			decoder_input_ids = shift_tokens_right(
				input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
			)
		# Compute the embeddings with nucleotide transformer encoder
		if encoder_outputs is None:
			encoder_outputs = self.encoder(input_ids)
			encoder_outputs.attention_mask = attention_mask
		else:
			encoder_outputs = HyenaModelOutput(
				last_hidden_state=encoder_outputs, 
				attention_mask=attention_mask
				)
		
		# Compress to a size that the decoder can handle
		pooled = self.pool(encoder_outputs.last_hidden_state, encoder_outputs.attention_mask)

		decoder_outputs = self.decoder(
			input_ids=decoder_input_ids,
			attention_mask=decoder_attention_mask,
			encoder_hidden_states=pooled[0],
			encoder_attention_mask=pooled[1].long(),
			head_mask=decoder_head_mask,
			cross_attn_head_mask=cross_attn_head_mask,
			past_key_values=past_key_values,
			inputs_embeds=decoder_inputs_embeds,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		if not return_dict:
			return decoder_outputs + encoder_outputs

		return LEDSeq2SeqModelOutput(
			last_hidden_state=decoder_outputs.last_hidden_state,
			past_key_values=decoder_outputs.past_key_values,
			decoder_hidden_states=decoder_outputs.hidden_states,
			decoder_attentions=decoder_outputs.attentions,
			cross_attentions=decoder_outputs.cross_attentions,
			encoder_last_hidden_state=encoder_outputs.last_hidden_state,
			encoder_hidden_states=encoder_outputs.encoder_hidden_states,
			encoder_attentions=encoder_outputs.attention_mask,
			encoder_global_attentions=None,
			segmentation_logits = encoder_outputs.segmentation_logits,
		)

class transgenicForConditionalGenerationT5(TransgenicPreTrainedModel):
	base_model_prefix = "transgenic"
	_keys_to_ignore_on_load_missing = ["final_logits_bias"]
	_tied_weights_keys = ["transgenic.decoder.embed_tokens.weight", "gff_head.weight"]

	def __init__(self, config, unlink=False):
		if not unlink:
			_tied_weights_keys = []
		super().__init__(config)
		self.transgenic = transgenicModelT5(config)
		self.gff_head = nn.Linear(config.d_model, config.vocab_size, bias=True)

		# Initialize weights and apply final processing
		self.post_init()
		self.initialize_weights()

	def get_encoder(self):
		return self.transgenic.get_encoder()

	def get_decoder(self):
		return self.transgenic.get_decoder()

	def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
		new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
		self._resize_final_logits_bias(new_embeddings.weight.shape[0])
		return new_embeddings

	def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
		old_num_tokens = self.final_logits_bias.shape[-1]
		if new_num_tokens <= old_num_tokens:
			new_bias = self.final_logits_bias[:, :new_num_tokens]
		else:
			extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
			new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
		self.register_buffer("final_logits_bias", new_bias)

	def get_output_embeddings(self):
		return self.gff_head

	def set_output_embeddings(self, new_embeddings):
		self.gff_head = new_embeddings
	
	def initialize_weights(self):
		for m in self.transgenic.decoder.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		decoder_input_ids: Optional[torch.LongTensor] = None,
		decoder_attention_mask: Optional[torch.LongTensor] = None,
		head_mask: Optional[torch.Tensor] = None,
		decoder_head_mask: Optional[torch.Tensor] = None,
		cross_attn_head_mask: Optional[torch.Tensor] = None,
		encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
		global_attention_mask: Optional[torch.FloatTensor] = None,
		past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple[torch.Tensor], LEDSeq2SeqLMOutput]:
		r"""
		labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
			Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
			config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
			(masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

		Returns:

		Conditional generation example:

		```python
		>>> from transformers import AutoTokenizer, LEDForConditionalGeneration

		>>> tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
		>>> TXT = "My friends are <mask> but they eat too many carbs."

		>>> model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")
		>>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]

		>>> prediction = model.generate(input_ids)[0]
		>>> print(tokenizer.decode(prediction, skip_special_tokens=True))
		```"""
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		if labels is not None:
			if use_cache:
				print("The `use_cache` argument is changed to `False` since `labels` is provided.", file=sys.stderr)
			use_cache = False
			if decoder_input_ids is None and decoder_inputs_embeds is None:
				decoder_input_ids = shift_tokens_right(
					labels, self.config.pad_token_id, self.config.decoder_start_token_id
				)
		
		outputs = self.transgenic(
			input_ids,
			attention_mask=attention_mask,
			decoder_input_ids=decoder_input_ids,
			decoder_attention_mask=decoder_attention_mask,
			encoder_outputs=encoder_outputs,
			global_attention_mask=global_attention_mask,
			head_mask=head_mask,
			decoder_head_mask=decoder_head_mask,
			cross_attn_head_mask=cross_attn_head_mask,
			past_key_values=past_key_values,
			inputs_embeds=inputs_embeds,
			decoder_inputs_embeds=decoder_inputs_embeds,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		lm_logits = self.gff_head(outputs[0])

		if not return_dict:
			output = (lm_logits,) + outputs[1:]
			return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

		masked_lm_loss = None
		if labels is not None:
			weights = torch.ones(272).to(lm_logits.device)
			weights[4:14] = 5
			loss_fct = nn.CrossEntropyLoss(weights=weights)
			masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
			#masked_lm_loss = HybridLoss(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
			#loss_fct = HybridSequenceLoss()
			#masked_lm_loss = loss_fct(lm_logits, labels)

		return LEDSeq2SeqLMOutput(
			loss=masked_lm_loss,
			logits=lm_logits,
			past_key_values=outputs.past_key_values,
			decoder_hidden_states=outputs.decoder_hidden_states,
			decoder_attentions=outputs.decoder_attentions,
			cross_attentions=outputs.cross_attentions,
			encoder_last_hidden_state=outputs.encoder_last_hidden_state,
			encoder_hidden_states=outputs.encoder_hidden_states,
			encoder_attentions=outputs.encoder_attentions,
			encoder_global_attentions=outputs.encoder_global_attentions,
			segmentation_logits=outputs.segmentation_logits,
		)

	def prepare_inputs_for_generation(
		self,
		decoder_input_ids,
		past_key_values=None,
		attention_mask=None,
		global_attention_mask=None,
		head_mask=None,
		decoder_head_mask=None,
		cross_attn_head_mask=None,
		use_cache=None,
		encoder_outputs=None,
		**kwargs,
	):
		# cut decoder_input_ids if past is used
		if past_key_values is not None:
			decoder_input_ids = decoder_input_ids[:, -1:]

		return {
			"input_ids": None,  # encoder_outputs is defined. input_ids not needed
			"encoder_outputs": encoder_outputs,
			"past_key_values": past_key_values,
			"decoder_input_ids": decoder_input_ids,
			"attention_mask": attention_mask,
			"global_attention_mask": global_attention_mask,
			"head_mask": head_mask,
			"decoder_head_mask": decoder_head_mask,
			"cross_attn_head_mask": cross_attn_head_mask,
			"use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
		}

	def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
		return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

	@staticmethod
	def _reorder_cache(past_key_values, beam_idx):
		reordered_past = ()
		for layer_past in past_key_values:
			# cached cross_attention states don't have to be reordered -> they are always the same
			reordered_past += (
				tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
				+ layer_past[2:],
			)
		return reordered_past
