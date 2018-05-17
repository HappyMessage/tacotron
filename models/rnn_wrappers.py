import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.framework import ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import check_ops
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.framework import tensor_shape

_zero_state_tensors = rnn_cell_impl._zero_state_tensors


def _compute_attention(attention_mechanism, cell_output, attention_state, attention_layer):
  """Computes the attention and alignments for a given attention_mechanism."""
  alignments, next_attention_state = attention_mechanism(cell_output, state=attention_state)

  # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
  expanded_alignments = array_ops.expand_dims(alignments, 1)
  # Context is the inner product of alignments and values along the
  # memory time dimension.
  # alignments shape is
  #   [batch_size, 1, memory_time]
  # attention_mechanism.values shape is
  #   [batch_size, memory_time, memory_size]
  # the batched matmul is over memory_time, so the output shape is
  #   [batch_size, 1, memory_size].
  # we then squeeze out the singleton dim.
  context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
  context = array_ops.squeeze(context, [1])

  if attention_layer is not None:
    attention = attention_layer(array_ops.concat([cell_output, context], 1))
  else:
    attention = context

  return attention, alignments, next_attention_state


class PrenetCell:
  """Two fully connected layers used as an information bottleneck for the attention.
  """
  def __init__(self, is_training, layer_sizes=[256, 256], activation=tf.nn.relu, scope=None):
    """
    Args:
      is_training: Boolean, determines if the model is in training or inference to control dropout
      layer_sizes: list of integers, the length of the list represents the number of pre-net
        layers and the list values represent the layers number of units
      activation: callable, activation functions of the prenet layers.
      scope: Prenet scope.
    """
    super(PrenetCell, self).__init__()
    self.drop_rate = 0.5

    self.layer_sizes = layer_sizes
    self.is_training = is_training
    self.activation = activation

    self.scope = 'prenet' if scope is None else scope

  def __call__(self, inputs):
    x = inputs

    with tf.variable_scope(self.scope):
      for i, size in enumerate(self.layer_sizes):
        dense = tf.layers.dense(x, units=size, activation=self.activation,
          name='dense_{}'.format(i + 1))
        #The paper discussed introducing diversity in generation at inference time
        #by using a dropout of 0.5 only in prenet layers (in both training and inference).
        x = tf.layers.dropout(dense, rate=self.drop_rate, training=True,
          name='dropout_{}'.format(i + 1) + self.scope)
    return x


class FrameProjectionCell:
  """Projection layer to r * num_mels dimensions or num_mels dimensions
  """
  def __init__(self, shape=80, activation=None, scope=None):
    """
    Args:
      shape: integer, dimensionality of output space (r*n_mels for decoder or n_mels for postnet)
      activation: callable, activation function
      scope: FrameProjection scope.
    """
    super(FrameProjectionCell, self).__init__()

    self.shape = shape
    self.activation = activation

    self.scope = 'Linear_projection' if scope is None else scope

  def __call__(self, inputs):
    with tf.variable_scope(self.scope):
      #If activation==None, this returns a simple Linear projection
      #else the projection will be passed through an activation function
      output = tf.layers.dense(inputs, units=self.shape, activation=self.activation,
        name='projection_{}'.format(self.scope))

      return output


class TacotronDecoderCellState(
  collections.namedtuple("TacotronDecoderCellState",
   ("cell_state", "attention", "time", "alignments",
    "attention_state", "alignment_history"))):
  """`namedtuple` storing the state of a `TacotronDecoderCell`.
  Contains:
    - `cell_state`: The state of the wrapped `RNNCell` at the previous time
    step.
    - `attention`: The attention emitted at the previous time step.
    - `time`: int32 scalar containing the current time step.
    - `alignments`: A single or tuple of `Tensor`(s) containing the alignments
     emitted at the previous time step for each attention mechanism.
    - `attention_state`: contains attention mechanism state.
    - `alignment_history`: a single or tuple of `TensorArray`(s)
     containing alignment matrices from all time steps for each attention
     mechanism. Call `stack()` on each to convert to a `Tensor`.
  """


class TacotronDecoderCell(RNNCell):
  """Tactron 2 Decoder Cell
  Decodes encoder output and previous mel frames into next r frames

  Decoder Step i:
    1) Prenet to compress last output information
    2) Concat compressed inputs with previous context vector (input feeding) *
    3) Decoder RNN (actual decoding) to predict current state s_{i} *
    4) Compute new context vector c_{i} based on s_{i} and a cumulative sum of previous alignments *
    5) Predict new output y_{i} using s_{i} and c_{i} (concatenated)

  * : This is typically taking a vanilla LSTM, wrapping it using tensorflow's attention wrapper,
  and wrap that with the prenet before doing an input feeding, and with the prediction layer
  that uses RNN states to project on output space. Actions marked with (*) can be replaced with
  tensorflow's attention wrapper call if it was using cumulative alignments instead of previous alignments only.
  """

  def __init__(self, prenet, attention_mechanism, rnn_cell, frame_projection):
    """Initialize decoder parameters

    Args:
        prenet: A tensorflow fully connected layer acting as the decoder pre-net
        attention_mechanism: A _BaseAttentionMechanism instance, usefull to
          learn encoder-decoder alignments
        rnn_cell: Instance of RNNCell, main body of the decoder
        frame_projection: tensorflow fully connected layer with r * num_mels output units
    """
    super(TacotronDecoderCell, self).__init__()
    #Initialize decoder layers
    self._prenet = prenet
    self._attention_mechanism = attention_mechanism
    self._cell = rnn_cell
    self._frame_projection = frame_projection

    self._attention_layer_size = self._attention_mechanism.values.get_shape()[-1].value

  def _batch_size_checks(self, batch_size, error_message):
    return [check_ops.assert_equal(batch_size,
      self._attention_mechanism.batch_size,
      message=error_message)]

  @property
  def output_size(self):
    return self._frame_projection.shape

  @property
  def state_size(self):
    """The `state_size` property of `TacotronDecoderCell`.

    Returns:
      An `TacotronDecoderCell` tuple containing shapes used by this object.
    """
    return TacotronDecoderCellState(
      cell_state=self._cell.state_size,
      time=tensor_shape.TensorShape([]),
      attention=self._attention_layer_size,
      attention_state=self._attention_mechanism.alignments_size,
      alignments=self._attention_mechanism.alignments_size,
      alignment_history=())

  def zero_state(self, batch_size, dtype):
    """Return an initial (zero) state tuple for this `AttentionWrapper`.

    Args:
      batch_size: `0D` integer tensor: the batch size.
      dtype: The internal state data type.
    Returns:
      An `TacotronDecoderCellState` tuple containing zeroed out tensors and,
      possibly, empty `TensorArray` objects.
    Raises:
      ValueError: (or, possibly at runtime, InvalidArgument), if
      `batch_size` does not match the output size of the encoder passed
      to the wrapper object at initialization time.
    """
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      cell_state = self._cell.zero_state(batch_size, dtype)
      error_message = (
        "When calling zero_state of TacotronDecoderCell %s: " % self._base_name +
        "Non-matching batch sizes between the memory "
        "(encoder output) and the requested batch size.")
      with ops.control_dependencies(
        self._batch_size_checks(batch_size, error_message)):
        cell_state = nest.map_structure(
          lambda s: array_ops.identity(s, name="checked_cell_state"),
          cell_state)
      return TacotronDecoderCellState(
        cell_state=cell_state,
        time=array_ops.zeros([], dtype=tf.int32),
        attention=_zero_state_tensors(self._attention_layer_size, batch_size, dtype),
        attention_state=self._attention_mechanism.initial_alignments(batch_size, dtype),
        alignments=self._attention_mechanism.initial_alignments(batch_size, dtype),
        alignment_history=tensor_array_ops.TensorArray(dtype=dtype, size=0,
        dynamic_size=True))

  def __call__(self, inputs, state):
    #Information bottleneck (essential for learning attention)
    prenet_output = self._prenet(inputs)                                     # [N, T_in, 256]

    #Concat context vector and prenet output to form LSTM cells input (input feeding)
    LSTM_input = tf.concat([prenet_output, state.attention], axis=-1)        # [N, T_in, 256+512]

    #Unidirectional LSTM layers
    LSTM_output, next_cell_state = self._cell(LSTM_input, state.cell_state)  # [N, T_in, 256]

    #Compute the attention (context) vector and alignments using
    #the new decoder cell hidden state as query vector
    #and cumulative alignments to extract location features
    #The choice of the new cell hidden state (s_{i}) of the last
    #decoder RNN Cell is based on Luong et Al. (2015):
    #https://arxiv.org/pdf/1508.04025.pdf
    previous_alignment_history = state.alignment_history
    context_vector, alignments, cumulated_alignments = _compute_attention(self._attention_mechanism,
      LSTM_output,
      state.attention_state,
      attention_layer=None)

    #Concat LSTM outputs and context vector to form projections inputs
    projections_input = tf.concat([LSTM_output, context_vector], axis=-1)    # [N, T_in, 256+512]

    #Compute predicted frames
    cell_outputs = self._frame_projection(projections_input)

    #Save alignment history
    alignment_history = previous_alignment_history.write(state.time, alignments)

    #Prepare next decoder state
    next_state = TacotronDecoderCellState(
      time=state.time + 1,
      cell_state=next_cell_state,
      attention=context_vector,
      attention_state=cumulated_alignments,
      alignments=alignments,
      alignment_history=alignment_history)

    return cell_outputs, next_state
