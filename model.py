import tensorflow as tf
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
import time

class TPUDeviceWorker:
    """Worker that handles inference on a single TPU device."""
    
    def __init__(self, device_id: int, model_path: str):
        self.device_id = device_id
        self.device_name = f'/device:TPU:{device_id}'
        self.model = None
        self.lock = threading.Lock()
        
        print(f"[TPUWorker-{device_id}] Loading model on {self.device_name}")
        with tf.device(self.device_name):
            self.model = tf.keras.models.load_model(model_path, compile=False)
            
            # Create optimized inference function with JIT
            @tf.function(jit_compile=True)
            def _inference_fn(batch):
                return self.model(batch, training=False)
            
            self._inference_fn = _inference_fn
            
            # Warm up
            dummy_input = tf.zeros((256, 8, 8, 108), dtype=tf.float32)
            for _ in range(3):
                _ = self._inference_fn(dummy_input)
        
        print(f"[TPUWorker-{device_id}] Ready")
    
    def predict(self, batch: np.ndarray) -> dict:
        """Run inference on this TPU device (batch size = 256)."""
        with tf.device(self.device_name):
            batch_tensor = tf.constant(batch, dtype=tf.float32)
            output = self._inference_fn(batch_tensor)
            
            if isinstance(output, dict):
                policy = output['policy'].numpy()
                value = output['value'].numpy().flatten()
            elif isinstance(output, (list, tuple)):
                policy = output[0].numpy()
                value = output[1].numpy().flatten()
            else:
                policy = output.numpy()
                value = np.zeros(batch.shape[0])
            
            return {
                'policies': policy,
                'values': value
            }


class ChessEvaluator:
    """Multi-TPU chess position evaluator with 8 independent device workers."""
    
    def __init__(self, model_path: str = '/mnt/disk/yeager/model_analysis/models/alphazero_model.h5', num_tpu_devices: int = 8):
        self.num_devices = num_tpu_devices
        self.workers = []
        
        print(f"[ChessEvaluator] Initializing {num_tpu_devices} TPU workers...")
        
        # Create a worker for each TPU device
        for device_id in range(num_tpu_devices):
            worker = TPUDeviceWorker(device_id, model_path)
            self.workers.append(worker)
        
        print(f"[ChessEvaluator] All {num_tpu_devices} workers ready")
    
    def get_device_worker(self, device_id: int):
        """Get a specific device worker for C++ to call directly."""
        if 0 <= device_id < self.num_devices:
            return self.workers[device_id] 
        else:
            raise ValueError(f"Invalid device_id: {device_id}")


if __name__ == "__main__":
    print("Testing ChessEvaluator with 8 TPU devices...")
    
    evaluator = ChessEvaluator(model_path='/mnt/disk/yeager/model_analysis/models/alphazero_model.h5', num_tpu_devices=8)
    
    # Test each device independently
    for device_id in range(8):
        worker = evaluator.get_device_worker(device_id)
        test_batch = np.random.randn(256, 8, 8, 108).astype(np.float32)
        
        print(f"\nTesting TPU device {device_id}...")
        start = time.time()
        result = worker.predict(test_batch)
        end = time.time()
        
        print(f"Device {device_id}: {(end-start)*1000:.2f}ms")
        print(f"  Policies shape: {result['policies'].shape}")
        print(f"  Values shape: {result['values'].shape}")