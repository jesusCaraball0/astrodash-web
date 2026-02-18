import sys
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Absolute paths
PROJECT_ROOT = "/home/jesusca/code_personal/astrodash-web"
TF_MODEL = f"/Users/jesuscaraball0/code/personal_code/astrodash-web/data/pre_trained_models/original_dash_models/zeroZ/tensorflow_model.ckpt"

def inspect_tf_graph():
    """Inspect the TensorFlow model graph to find tensor names"""
    print("=== Inspecting TensorFlow Model Graph ===")
    
    # Load TensorFlow model
    saver = tf.train.import_meta_graph(TF_MODEL + '.meta')
    
    with tf.Session() as sess:
        saver.restore(sess, TF_MODEL)
        
        # Get the graph
        graph = tf.get_default_graph()
        
        # List all operations in the graph
        print("All operations in the graph:")
        for op in graph.get_operations():
            print(f"  {op.name}: {op.type}")
        
        # Look for specific tensor types
        print("\nLooking for input placeholders:")
        for op in graph.get_operations():
            if op.type == "Placeholder":
                print(f"  Placeholder: {op.name}")
        
        print("\nLooking for softmax operations:")
        for op in graph.get_operations():
            if "softmax" in op.name.lower() or "Softmax" in op.name:
                print(f"  Softmax: {op.name}")
        
        print("\nLooking for output operations (last few layers):")
        # Get operations that might be outputs
        for op in graph.get_operations():
            if op.type in ["MatMul", "Add", "Softmax", "Identity"]:
                print(f"  {op.type}: {op.name}")

if __name__ == "__main__":
    inspect_tf_graph() 