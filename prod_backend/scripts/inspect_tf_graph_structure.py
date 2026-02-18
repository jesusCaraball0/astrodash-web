import sys
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def inspect_tf_graph_structure():
    """Inspect the actual TensorFlow graph structure to understand the architecture."""
    tf_model_path = "/Users/jesuscaraball0/code/personal_code/astrodash-web/data/pre_trained_models/original_dash_models/zeroZ/tensorflow_model.ckpt"

    # Load the meta graph
    saver = tf.train.import_meta_graph(tf_model_path + '.meta')

    with tf.Session() as sess:
        saver.restore(sess, tf_model_path)
        graph = tf.get_default_graph()

        print("=== TensorFlow Graph Structure ===")

        # Get all operations in the graph
        all_ops = graph.get_operations()

        # Filter for key operations
        conv_ops = [op for op in all_ops if 'conv2d' in op.name.lower()]
        pool_ops = [op for op in all_ops if 'max_pool' in op.name.lower()]
        relu_ops = [op for op in all_ops if 'relu' in op.name.lower()]
        matmul_ops = [op for op in all_ops if 'matmul' in op.name.lower()]
        dropout_ops = [op for op in all_ops if 'dropout' in op.name.lower()]
        softmax_ops = [op for op in all_ops if 'softmax' in op.name.lower()]

        print(f"Convolutional operations: {[op.name for op in conv_ops]}")
        print(f"Pooling operations: {[op.name for op in pool_ops]}")
        print(f"ReLU operations: {[op.name for op in relu_ops]}")
        print(f"MatMul operations: {[op.name for op in matmul_ops]}")
        print(f"Dropout operations: {[op.name for op in dropout_ops]}")
        print(f"Softmax operations: {[op.name for op in softmax_ops]}")

        print("\n=== Layer Connections ===")

        # Try to trace the forward pass
        try:
            # Get input and output tensors
            x = graph.get_tensor_by_name("Placeholder:0")
            y_conv = graph.get_tensor_by_name("Softmax:0")

            print(f"Input tensor: {x.name}, shape: {x.shape}")
            print(f"Output tensor: {y_conv.name}, shape: {y_conv.shape}")

            # Get the operation that produces the output
            output_op = y_conv.op
            print(f"Output operation: {output_op.name}")

            # Trace backwards to understand the architecture
            print("\n=== Backward Trace from Output ===")
            current_op = output_op
            depth = 0
            max_depth = 20

            while current_op and depth < max_depth:
                print(f"Depth {depth}: {current_op.name} ({current_op.type})")

                # Get inputs to this operation
                inputs = current_op.inputs
                if inputs:
                    print(f"  Inputs: {[inp.name for inp in inputs]}")
                    current_op = inputs[0].op
                else:
                    break
                depth += 1

        except Exception as e:
            print(f"Error tracing graph: {e}")

        # Also check for any layer-specific operations
        print("\n=== Layer-Specific Operations ===")
        layer1_ops = [op for op in all_ops if 'layer1' in op.name]
        layer2_ops = [op for op in all_ops if 'layer2' in op.name]
        readout_ops = [op for op in all_ops if 'readout' in op.name]

        print(f"Layer1 operations: {[op.name for op in layer1_ops]}")
        print(f"Layer2 operations: {[op.name for op in layer2_ops]}")
        print(f"Readout operations: {[op.name for op in readout_ops]}")

if __name__ == "__main__":
    inspect_tf_graph_structure()
