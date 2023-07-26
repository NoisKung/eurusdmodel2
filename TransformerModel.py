import tensorflow as tf
from TransformerBlock import TransformerBlock  # Import the TransformerBlock class correctly

# Define the Transformer model architecture
class TransformerModel(tf.keras.Model):
    def __init__(self, input_dim, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.transformer_blocks = [TransformerBlock(input_dim, num_heads, ff_dim, dropout) for _ in range(num_transformer_blocks)]
        self.mlp = tf.keras.Sequential([tf.keras.layers.Dense(units, activation='relu') for units in mlp_units])
        self.final_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = inputs
        print("Input shape in TransformerBlock:", x.shape)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')(x)
        x = self.mlp(x)
        return self.final_layer(x)