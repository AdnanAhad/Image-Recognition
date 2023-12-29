import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Path to the frozen inference graph and labels
PATH_TO_FROZEN_GRAPH = "frozen_inference_graph.pb"
PATH_TO_LABELS = "mscoco_label_map.pbtxt"

# Load the frozen TensorFlow model
detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, "rb") as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.compat.v1.import_graph_def(od_graph_def, name="")

# Load label map
category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True
)

# Load image
image_path = "mix.jpeg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detection
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
        detection_boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
        detection_scores = detection_graph.get_tensor_by_name("detection_scores:0")
        detection_classes = detection_graph.get_tensor_by_name("detection_classes:0")
        num_detections = detection_graph.get_tensor_by_name("num_detections:0")

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_expanded = np.expand_dims(image_rgb, axis=0)

        # Actual detection
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded},
        )

        # Filter out detections with low confidence
min_confidence = 0.5  # You can adjust this threshold
boxes = boxes[scores > min_confidence]
classes = classes[scores > min_confidence].astype(np.int32)
scores = scores[scores > min_confidence]

# Get labels for detected classes
detected_labels = [category_index[c]["name"] for c in classes]

# Build a sentence describing the contents
if detected_labels:
    sentence = f"The image contains: {', '.join(detected_labels)} with confidence scores {', '.join(map(str, scores))}."
else:
    sentence = "No objects detected in the image."

# Print or use the sentence as needed
print(sentence)

# Visualization of the results
vis_util.visualize_boxes_and_labels_on_image_array(
    image_rgb,
    np.squeeze(boxes),
    classes,
    scores,
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
)

# Visualization of the results
# vis_util.visualize_boxes_and_labels_on_image_array(
#     image_rgb,
#     np.squeeze(boxes),
#     np.squeeze(classes).astype(np.int32),
#     np.squeeze(scores),
#     category_index,
#     use_normalized_coordinates=True,
#     line_thickness=8,
# )

# Display the results
plt.imshow(image_rgb)

# Save the figure instead of displaying it interactively
plt.imsave("result_image.png", image_rgb)

print("Hello WOrld at last")
