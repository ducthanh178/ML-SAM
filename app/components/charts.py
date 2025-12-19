import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots


def plot_accuracy_comparison(metrics_sgd: dict, metrics_sam: dict):
    """Plot train vs test accuracy comparison for SGD and SAM."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Train Accuracy", "Test Accuracy"),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Train accuracy
    fig.add_trace(
        go.Bar(
            name="SGD",
            x=["SGD"],
            y=[metrics_sgd.get("train_accuracy", [0])[-1] if metrics_sgd.get("train_accuracy") else 0],
            marker_color="#FF6B6B",
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            name="SAM",
            x=["SAM"],
            y=[metrics_sam.get("train_accuracy", [0])[-1] if metrics_sam.get("train_accuracy") else 0],
            marker_color="#4ECDC4",
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Test accuracy
    fig.add_trace(
        go.Bar(
            name="SGD",
            x=["SGD"],
            y=[metrics_sgd.get("test_accuracy", 0)],
            marker_color="#FF6B6B",
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            name="SAM",
            x=["SAM"],
            y=[metrics_sam.get("test_accuracy", 0)],
            marker_color="#4ECDC4",
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=2)
    fig.update_layout(
        height=400,
        title_text="Final Accuracy Comparison: SGD vs SAM",
        title_x=0.5,
        hovermode="x unified"
    )
    
    return fig


def plot_training_curves(metrics_sgd: dict, metrics_sam: dict):
    """Plot training curves for loss and accuracy."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Loss", "Accuracy"),
        vertical_spacing=0.15
    )
    
    epochs_sgd = list(range(1, len(metrics_sgd.get("train_loss", [])) + 1))
    epochs_sam = list(range(1, len(metrics_sam.get("train_loss", [])) + 1))
    
    # Loss curves
    if epochs_sgd:
        fig.add_trace(
            go.Scatter(
                x=epochs_sgd,
                y=metrics_sgd.get("train_loss", []),
                name="SGD Train Loss",
                line=dict(color="#FF6B6B", width=2),
                mode="lines"
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=epochs_sgd,
                y=metrics_sgd.get("val_loss", []),
                name="SGD Val Loss",
                line=dict(color="#FF6B6B", width=2, dash="dash"),
                mode="lines"
            ),
            row=1, col=1
        )
    
    if epochs_sam:
        fig.add_trace(
            go.Scatter(
                x=epochs_sam,
                y=metrics_sam.get("train_loss", []),
                name="SAM Train Loss",
                line=dict(color="#4ECDC4", width=2),
                mode="lines"
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=epochs_sam,
                y=metrics_sam.get("val_loss", []),
                name="SAM Val Loss",
                line=dict(color="#4ECDC4", width=2, dash="dash"),
                mode="lines"
            ),
            row=1, col=1
        )
    
    # Accuracy curves
    if epochs_sgd:
        fig.add_trace(
            go.Scatter(
                x=epochs_sgd,
                y=metrics_sgd.get("train_accuracy", []),
                name="SGD Train Acc",
                line=dict(color="#FF6B6B", width=2),
                mode="lines",
                showlegend=False
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=epochs_sgd,
                y=metrics_sgd.get("val_accuracy", []),
                name="SGD Val Acc",
                line=dict(color="#FF6B6B", width=2, dash="dash"),
                mode="lines",
                showlegend=False
            ),
            row=2, col=1
        )
    
    if epochs_sam:
        fig.add_trace(
            go.Scatter(
                x=epochs_sam,
                y=metrics_sam.get("train_accuracy", []),
                name="SAM Train Acc",
                line=dict(color="#4ECDC4", width=2),
                mode="lines",
                showlegend=False
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=epochs_sam,
                y=metrics_sam.get("val_accuracy", []),
                name="SAM Val Acc",
                line=dict(color="#4ECDC4", width=2, dash="dash"),
                mode="lines",
                showlegend=False
            ),
            row=2, col=1
        )
    
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=2, col=1)
    fig.update_layout(
        height=600,
        title_text="Training Curves: SGD vs SAM",
        title_x=0.5
    )
    
    return fig


def plot_generalization_gap(metrics_sgd: dict, metrics_sam: dict):
    """Plot generalization gap (train vs test accuracy)."""
    train_acc_sgd = metrics_sgd.get("train_accuracy", [0])[-1] if metrics_sgd.get("train_accuracy") else 0
    test_acc_sgd = metrics_sgd.get("test_accuracy", 0)
    train_acc_sam = metrics_sam.get("train_accuracy", [0])[-1] if metrics_sam.get("train_accuracy") else 0
    test_acc_sam = metrics_sam.get("test_accuracy", 0)
    
    gap_sgd = train_acc_sgd - test_acc_sgd
    gap_sam = train_acc_sam - test_acc_sam
    
    fig = go.Figure()
    
    # SGD bars
    fig.add_trace(go.Bar(
        name="SGD Train",
        x=["SGD"],
        y=[train_acc_sgd],
        marker_color="#FF6B6B",
        text=[f"{train_acc_sgd:.3f}"],
        textposition="outside"
    ))
    
    fig.add_trace(go.Bar(
        name="SGD Test",
        x=["SGD"],
        y=[test_acc_sgd],
        marker_color="#FF9999",
        text=[f"{test_acc_sgd:.3f}"],
        textposition="outside"
    ))
    
    # SAM bars
    fig.add_trace(go.Bar(
        name="SAM Train",
        x=["SAM"],
        y=[train_acc_sam],
        marker_color="#4ECDC4",
        text=[f"{train_acc_sam:.3f}"],
        textposition="outside"
    ))
    
    fig.add_trace(go.Bar(
        name="SAM Test",
        x=["SAM"],
        y=[test_acc_sam],
        marker_color="#95E5DF",
        text=[f"{test_acc_sam:.3f}"],
        textposition="outside"
    ))
    
    # Add gap annotations
    fig.add_annotation(
        x="SGD",
        y=(train_acc_sgd + test_acc_sgd) / 2,
        text=f"Gap: {gap_sgd:.3f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="red",
        font=dict(color="red", size=12)
    )
    
    fig.add_annotation(
        x="SAM",
        y=(train_acc_sam + test_acc_sam) / 2,
        text=f"Gap: {gap_sam:.3f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="teal",
        font=dict(color="teal", size=12)
    )
    
    fig.update_layout(
        barmode="group",
        title="Generalization Gap: Train vs Test Accuracy",
        title_x=0.5,
        yaxis_title="Accuracy",
        yaxis_range=[0, 1],
        height=500,
        hovermode="x unified"
    )
    
    return fig


def plot_confidence_comparison(predictions_sgd: dict, predictions_sam: dict, sample_idx: int = 0):
    """Plot prediction confidence comparison for a sample."""
    confidences_sgd = predictions_sgd.get("confidences", [])
    confidences_sam = predictions_sam.get("confidences", [])
    
    if not confidences_sgd or not confidences_sam:
        return None
    
    if sample_idx >= len(confidences_sgd) or sample_idx >= len(confidences_sam):
        sample_idx = 0
    
    conf_sgd = confidences_sgd[sample_idx]
    conf_sam = confidences_sam[sample_idx]
    
    # Get top classes
    num_classes = len(conf_sgd) if isinstance(conf_sgd, list) else 10
    top_k = min(5, num_classes)
    
    if isinstance(conf_sgd, list):
        top_indices_sgd = np.argsort(conf_sgd)[-top_k:][::-1]
        top_conf_sgd = [conf_sgd[i] for i in top_indices_sgd]
        top_indices_sam = np.argsort(conf_sam)[-top_k:][::-1]
        top_conf_sam = [conf_sam[i] for i in top_indices_sam]
    else:
        top_indices_sgd = list(range(top_k))
        top_conf_sgd = conf_sgd[:top_k] if len(conf_sgd) >= top_k else conf_sgd
        top_indices_sam = list(range(top_k))
        top_conf_sam = conf_sam[:top_k] if len(conf_sam) >= top_k else conf_sam
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("SGD Confidence", "SAM Confidence"),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    fig.add_trace(
        go.Bar(
            x=[f"Class {i}" for i in top_indices_sgd],
            y=top_conf_sgd,
            marker_color="#FF6B6B",
            name="SGD",
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=[f"Class {i}" for i in top_indices_sam],
            y=top_conf_sam,
            marker_color="#4ECDC4",
            name="SAM",
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_yaxes(title_text="Confidence", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Confidence", range=[0, 1], row=1, col=2)
    fig.update_layout(
        height=400,
        title_text=f"Prediction Confidence Comparison (Sample {sample_idx})",
        title_x=0.5
    )
    
    return fig


def plot_confidence_stability(predictions_sgd: dict, predictions_sam: dict, num_samples: int = 20):
    """Plot confidence stability across multiple samples."""
    confidences_sgd = predictions_sgd.get("confidences", [])
    confidences_sam = predictions_sam.get("confidences", [])
    
    if not confidences_sgd or not confidences_sam:
        return None
    
    num_samples = min(num_samples, len(confidences_sgd), len(confidences_sam))
    
    # Calculate max confidence for each sample
    max_conf_sgd = [max(conf) if isinstance(conf, list) else conf[0] if len(conf) > 0 else 0 
                    for conf in confidences_sgd[:num_samples]]
    max_conf_sam = [max(conf) if isinstance(conf, list) else conf[0] if len(conf) > 0 else 0 
                    for conf in confidences_sam[:num_samples]]
    
    # Calculate std for stability
    std_sgd = np.std(max_conf_sgd)
    std_sam = np.std(max_conf_sam)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(num_samples)),
        y=max_conf_sgd,
        mode="lines+markers",
        name=f"SGD (std={std_sgd:.3f})",
        line=dict(color="#FF6B6B", width=2),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(num_samples)),
        y=max_conf_sam,
        mode="lines+markers",
        name=f"SAM (std={std_sam:.3f})",
        line=dict(color="#4ECDC4", width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Confidence Stability Across Samples",
        title_x=0.5,
        xaxis_title="Sample Index",
        yaxis_title="Max Confidence",
        yaxis_range=[0, 1],
        height=400,
        hovermode="x unified"
    )
    
    return fig


def plot_loss_landscape(loss_surface: np.ndarray, title: str = "Loss Landscape"):
    """Plot 3D loss landscape."""
    if loss_surface.size == 0:
        return None
    
    if loss_surface.ndim == 2:
        # 2D surface
        x = np.arange(loss_surface.shape[0])
        y = np.arange(loss_surface.shape[1])
        X, Y = np.meshgrid(x, y)
        Z = loss_surface
        
        fig = go.Figure(data=[go.Surface(
            z=Z,
            x=X,
            y=Y,
            colorscale="Viridis",
            showscale=True
        )])
        
        fig.update_layout(
            title=title,
            title_x=0.5,
            scene=dict(
                xaxis_title="Direction 1",
                yaxis_title="Direction 2",
                zaxis_title="Loss",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600
        )
    else:
        # 1D or other shape - plot as heatmap or line
        fig = go.Figure(data=[go.Heatmap(
            z=loss_surface.flatten().reshape(-1, 1).T if loss_surface.ndim == 1 else loss_surface,
            colorscale="Viridis",
            showscale=True
        )])
        
        fig.update_layout(
            title=title,
            title_x=0.5,
            height=400
        )
    
    return fig





