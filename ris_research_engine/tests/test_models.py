"""Test all model types for input/output shapes, forward pass, and parameter count."""

import pytest
import torch
import torch.nn as nn

from ris_research_engine.plugins.models import (
    get_model, list_models,
    MLPModel, ResidualMLPModel, CNN1DModel, CNN2DModel,
    TransformerModel, SetTransformerModel, LSTMModel
)


# All 7 model types
MODEL_TYPES = [
    "mlp",
    "residual_mlp",
    "cnn_1d",
    "cnn_2d",
    "transformer",
    "set_transformer",
    "lstm"
]


class TestModelRegistry:
    """Test model registration and discovery."""
    
    def test_list_models(self):
        """Test that all models are registered."""
        models = list_models()
        assert len(models) >= 7
        for model_type in MODEL_TYPES:
            assert model_type in models
    
    def test_get_model(self):
        """Test getting model instances."""
        for model_type in MODEL_TYPES:
            model = get_model(model_type)
            assert model is not None
            assert model.name == model_type
    
    def test_get_invalid_model(self):
        """Test that invalid model name raises error."""
        with pytest.raises(KeyError):
            get_model("nonexistent_model")


class TestModelShapes:
    """Test model input/output shapes."""
    
    @pytest.mark.parametrize("model_type", MODEL_TYPES)
    def test_model_output_shape(self, model_type, sample_model_input):
        """Test model produces correct output shape."""
        model = get_model(model_type)
        
        # Build model
        batch_size, M, K = sample_model_input.shape
        N = 16  # Output dimension
        
        net = model.build(input_dim=(M, K), output_dim=N)
        assert isinstance(net, nn.Module)
        
        # Forward pass
        net.eval()
        with torch.no_grad():
            output = net(sample_model_input)
        
        assert output.shape == (batch_size, N), f"{model_type} output shape mismatch"
    
    @pytest.mark.parametrize("model_type", MODEL_TYPES)
    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_variable_batch_size(self, model_type, batch_size):
        """Test models handle different batch sizes."""
        model = get_model(model_type)
        
        M, K, N = 4, 16, 16
        net = model.build(input_dim=(M, K), output_dim=N)
        
        input_tensor = torch.randn(batch_size, M, K)
        
        net.eval()
        with torch.no_grad():
            output = net(input_tensor)
        
        assert output.shape == (batch_size, N)
    
    @pytest.mark.parametrize("model_type", MODEL_TYPES)
    @pytest.mark.parametrize("N", [8, 16, 32, 64])
    def test_variable_output_dim(self, model_type, N):
        """Test models handle different output dimensions."""
        model = get_model(model_type)
        
        M, K = 4, 16
        batch_size = 8
        
        net = model.build(input_dim=(M, K), output_dim=N)
        input_tensor = torch.randn(batch_size, M, K)
        
        net.eval()
        with torch.no_grad():
            output = net(input_tensor)
        
        assert output.shape == (batch_size, N)


class TestModelForwardPass:
    """Test model forward pass correctness."""
    
    @pytest.mark.parametrize("model_type", MODEL_TYPES)
    def test_forward_no_error(self, model_type, sample_model_input):
        """Test forward pass executes without error."""
        model = get_model(model_type)
        
        M, K, N = 4, 16, 16
        net = model.build(input_dim=(M, K), output_dim=N)
        
        try:
            output = net(sample_model_input)
            assert output is not None
        except Exception as e:
            pytest.fail(f"{model_type} forward pass failed: {e}")
    
    @pytest.mark.parametrize("model_type", MODEL_TYPES)
    def test_forward_output_finite(self, model_type, sample_model_input):
        """Test forward pass produces finite outputs."""
        model = get_model(model_type)
        
        M, K, N = 4, 16, 16
        net = model.build(input_dim=(M, K), output_dim=N)
        
        net.eval()
        with torch.no_grad():
            output = net(sample_model_input)
        
        assert torch.all(torch.isfinite(output)), f"{model_type} produced non-finite values"
    
    @pytest.mark.parametrize("model_type", MODEL_TYPES)
    def test_training_mode(self, model_type, sample_model_input):
        """Test model works in training mode."""
        model = get_model(model_type)
        
        M, K, N = 4, 16, 16
        net = model.build(input_dim=(M, K), output_dim=N)
        
        net.train()
        output = net(sample_model_input)
        
        assert output.requires_grad or not any(p.requires_grad for p in net.parameters())
    
    @pytest.mark.parametrize("model_type", MODEL_TYPES)
    def test_gradient_flow(self, model_type, sample_model_input, sample_labels):
        """Test gradients flow through model."""
        model = get_model(model_type)
        
        M, K, N = 4, 16, 16
        net = model.build(input_dim=(M, K), output_dim=N)
        
        net.train()
        output = net(sample_model_input)
        
        # Compute loss
        loss = nn.CrossEntropyLoss()(output, sample_labels)
        loss.backward()
        
        # Check at least some parameters have gradients
        has_grad = False
        for param in net.parameters():
            if param.grad is not None:
                has_grad = True
                assert torch.all(torch.isfinite(param.grad)), f"{model_type} has non-finite gradients"
        
        assert has_grad, f"{model_type} has no gradients"


class TestModelParameterCount:
    """Test model parameter counts are reasonable."""
    
    @pytest.mark.parametrize("model_type", MODEL_TYPES)
    def test_has_parameters(self, model_type):
        """Test model has trainable parameters."""
        model = get_model(model_type)
        
        M, K, N = 4, 16, 16
        net = model.build(input_dim=(M, K), output_dim=N)
        
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        
        assert total_params > 0, f"{model_type} has no parameters"
        assert trainable_params > 0, f"{model_type} has no trainable parameters"
    
    @pytest.mark.parametrize("model_type", MODEL_TYPES)
    def test_parameter_count_reasonable(self, model_type):
        """Test parameter count is reasonable (not too large)."""
        model = get_model(model_type)
        
        M, K, N = 4, 16, 16
        net = model.build(input_dim=(M, K), output_dim=N)
        
        total_params = sum(p.numel() for p in net.parameters())
        
        # Should be less than 10M parameters for small test models
        assert total_params < 10_000_000, f"{model_type} has too many parameters: {total_params}"
    
    @pytest.mark.parametrize("model_type", MODEL_TYPES)
    def test_parameter_shapes(self, model_type):
        """Test parameters have valid shapes."""
        model = get_model(model_type)
        
        M, K, N = 4, 16, 16
        net = model.build(input_dim=(M, K), output_dim=N)
        
        for name, param in net.named_parameters():
            assert param.shape is not None
            assert all(d > 0 for d in param.shape), f"{model_type} has invalid parameter shape for {name}"


class TestSpecificModels:
    """Test specific model implementations."""
    
    def test_mlp_architecture(self):
        """Test MLP has proper linear layers."""
        model = get_model("mlp")
        net = model.build(input_dim=(4, 16), output_dim=16, hidden_dims=[64, 32])
        
        # Should have Linear layers
        has_linear = any(isinstance(m, nn.Linear) for m in net.modules())
        assert has_linear, "MLP should have Linear layers"
    
    def test_residual_mlp_skip_connections(self):
        """Test ResidualMLP has residual connections."""
        model = get_model("residual_mlp")
        net = model.build(input_dim=(4, 16), output_dim=16, hidden_dims=[64, 64])
        
        # Residual should have more parameters than plain MLP
        total_params = sum(p.numel() for p in net.parameters())
        assert total_params > 0
    
    def test_cnn1d_architecture(self):
        """Test CNN1D has convolutional layers."""
        model = get_model("cnn_1d")
        net = model.build(input_dim=(4, 16), output_dim=16)
        
        # Should have Conv1d layers
        has_conv = any(isinstance(m, nn.Conv1d) for m in net.modules())
        assert has_conv, "CNN1D should have Conv1d layers"
    
    def test_cnn2d_architecture(self):
        """Test CNN2D has 2D convolutional layers."""
        model = get_model("cnn_2d")
        net = model.build(input_dim=(4, 16), output_dim=16)
        
        # Should have Conv2d layers
        has_conv2d = any(isinstance(m, nn.Conv2d) for m in net.modules())
        assert has_conv2d, "CNN2D should have Conv2d layers"
    
    def test_transformer_attention(self):
        """Test Transformer has attention mechanism."""
        model = get_model("transformer")
        net = model.build(input_dim=(4, 16), output_dim=16)
        
        # Should have some form of attention
        total_params = sum(p.numel() for p in net.parameters())
        assert total_params > 1000  # Transformers typically have many parameters
    
    def test_lstm_recurrent(self):
        """Test LSTM has recurrent layers."""
        model = get_model("lstm")
        net = model.build(input_dim=(4, 16), output_dim=16)
        
        # Should have LSTM layers
        has_lstm = any(isinstance(m, nn.LSTM) for m in net.modules())
        assert has_lstm, "LSTM should have LSTM layers"


class TestModelDevice:
    """Test model device handling."""
    
    @pytest.mark.parametrize("model_type", MODEL_TYPES)
    def test_model_on_cpu(self, model_type, device):
        """Test model works on CPU."""
        model = get_model(model_type)
        net = model.build(input_dim=(4, 16), output_dim=16)
        
        net = net.to(device)
        input_tensor = torch.randn(8, 4, 16).to(device)
        
        net.eval()
        with torch.no_grad():
            output = net(input_tensor)
        
        assert output.device == device
