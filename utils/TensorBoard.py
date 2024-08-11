from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch

def save_summary(writer, loss_dict, global_step, tag, lr=None, momentum=None):
    """
    Save training and evaluation metrics to TensorBoard.

    Args:
        writer (SummaryWriter): The TensorBoard SummaryWriter instance.
        loss_dict (dict): A dictionary of loss values, where the keys are the metric names.
        global_step (int): The current global step (i.e., iteration or epoch).
        tag (str): The tag to use for the metrics (e.g., 'Training', 'Validation').
        lr (float, optional): The current learning rate.
        momentum (float, optional): The current momentum value.
    """
    # Log training/evaluation metrics
    for metric, value in loss_dict.items():
        writer.add_scalar(f'{tag}/{metric}', value, global_step)

    # Log learning rate and momentum (if provided)
    if lr is not None:
        writer.add_scalar(f'{tag}/Learning Rate', lr, global_step)
    if momentum is not None:
        writer.add_scalar(f'{tag}/Momentum', momentum, global_step)

def log_training_metrics(writer, model, batch, step):
    """
    Log training metrics to TensorBoard.

    Args:
        writer (SummaryWriter): The TensorBoard SummaryWriter instance.
        model (nn.Module): The PyTorch model.
        batch (dict): A batch of data from the training dataset.
        step (int): The current training step.
    """
    # Forward pass and loss calculation
    outputs = model(batch)
    loss = F.mse_loss(outputs, batch['labels'])
    loss.backward()

    # Log the training loss
    writer.add_scalar('Training/Loss', loss.item(), global_step=step)

    # Log the model graph
    writer.add_graph(model, batch['features'])

def log_evaluation_metrics(writer, model, batch, epoch):
    """
    Log evaluation metrics to TensorBoard.

    Args:
        writer (SummaryWriter): The TensorBoard SummaryWriter instance.
        model (nn.Module): The PyTorch model.
        batch (dict): A batch of data from the evaluation dataset.
        epoch (int): The current evaluation epoch.
    """
    # Forward pass and metric calculation
    outputs = model(batch)
    precision, recall, f1 = evaluate_model(outputs, batch['labels'])

    # Log the evaluation metrics
    writer.add_scalar('Validation/Precision', precision, global_step=epoch)
    writer.add_scalar('Validation/Recall', recall, global_step=epoch)
    writer.add_scalar('Validation/F1-score', f1, global_step=epoch)

def evaluate_model(outputs, labels):
    """
    Evaluate the model's performance based on the given outputs and labels.

    Args:
        outputs (torch.Tensor): The model's output predictions.
        labels (torch.Tensor): The ground truth labels.

    Returns:
        tuple: Precision, recall, and F1-score.
    """
    # Implement your evaluation logic here
    # This is a placeholder, you should replace it with your actual evaluation code
    precision = 0.8
    recall = 0.7
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

def log_model_checkpoint(writer, model, batch, checkpoint_path):
    """
    Log a model checkpoint to TensorBoard.

    Args:
        writer (SummaryWriter): The TensorBoard SummaryWriter instance.
        model (nn.Module): The PyTorch model.
        batch (dict): A batch of data from the dataset.
        checkpoint_path (str): The path to save the model checkpoint.
    """
    # Save the model checkpoint
    torch.save(model.state_dict(), checkpoint_path)
    writer.add_graph(model, batch['features'])

def visualize_results(writer, batch, model_outputs, step):
    """
    Visualize the input point cloud, ground truth bounding boxes, and model predictions in TensorBoard.

    Args:
        writer (SummaryWriter): The TensorBoard SummaryWriter instance.
        batch (dict): A batch of data from the dataset.
        model_outputs (torch.Tensor): The model's output predictions.
        step (int): The current step (iteration or epoch).
    """
    # Visualize the input point cloud
    writer.add_mesh('Input Point Cloud', vertices=batch['features'], global_step=step)

    # Visualize the ground truth bounding boxes
    writer.add_boxes('Ground Truth Boxes', batch['labels'], global_step=step)

    # Visualize the model predictions
    writer.add_boxes('Predicted Boxes', model_outputs, global_step=step)
