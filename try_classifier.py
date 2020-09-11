import pandas as pd
from simpletransformers.classification import ClassificationModel

import torch

cuda_available = torch.cuda.is_available()
print("Cuda available: ", cuda_available)

# Train and Evaluation data needs to be in a Pandas Dataframe of two columns. The first column is the text with type str, and the second column is the label with type int.
train_data = [["Example sentence belonging to class 1", 1], ["Example sentence belonging to class 0", 0]]
train_df = pd.DataFrame(train_data)

eval_data = [["Example eval sentence belonging to class 1", 1], ["Example eval sentence belonging to class 0", 0]]
eval_df = pd.DataFrame(eval_data)

model_args = {
    "output_dir": "models/albert/trial/",
    "overwrite_output_dir" : True
}
# Create a ClassificationModel
model = ClassificationModel("albert", "albert-base-v2", args=model_args, use_cuda=cuda_available)


# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

print(result)
