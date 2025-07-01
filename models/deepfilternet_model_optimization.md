## Deepfilternet model optimization

### Goal
The deepfilternet model is designed to be light and fast, so it is also suitable for running on mobile devices.  
Some tests are conducted, to see how it performes in realtime on relatively new, and older, android devices.  
Google Pixel 3A is chosen as an older device, the Samsung Flip 3 is chosen as a newer device.  
The Deepfilternet3 model processes audio chunks smoothly on the Samsung Flip 3, while it failed to process chunks correctly on the Google Pixel 3A, it is too slow and the resulting audio is corrupted.  
The Goal was therefore to optimise the Deepfilternet3 model to be able to process audio samples smoothly even on older devices such as the Google Pixel 3A.  

### Fusing Layers
The first technique tried is Fusing Layers.  
The Deepfilternet network is divided into encoder, erb decoder and df decoder.  
Although with different structures, all the three components are of the "CNN + RNN" type, where the RNN part is a squeezed GRU, an efficient type of recurrent neural network.  
In the convolutional part, each Conv layer is followed by a Batch normalization layer (Bn)  and a Relu layer, the Conv + bn + Relu block can be fused, resulting in a single, more efficient layer.  
This not reduce the size of the model, just made the inference more efficient especially on CPU.  
With the fusing layers operation, the model also became performant on the Google Pixel 3A.  

### Quantization
Quantization of the model is also tried. Weights and activation functions are in float32 precision, converting them into int8 precision, it should give an advantage on mobile devices as int8 calculations are performed faster.  
Considering the structure of the network, the most appropriate choice is to apply static quantization to the convolutional layers, as CNN have a highly regular structure and perform repetitive and predictable operations on the feature maps.
This allows efficient inference optimization while maintaining good accuracy.  
For GRU layers, it is preferable to use dynamic quantization, as they process sequential data with hidden states and gates that vary at each time step, making the internal dynamics of the model less suitable for static quantization.  
Dynamic quantization reduces approximation errors, as it avoids directly to quantize activations and is better suited to the variable behaviour of recurrent models.
#### Pytorch
Applying these quantizations to the pytorch model, results in a model that is about half the size and runs on a PC almost twice as fast.  
The problem is that it must be converted to onnx, as the inference on mobile is then done in rust with the tract-onnx library.  
Pytorch Quantized layers, however, cannot be exported to onnx, so this way is not possible.
#### Onnx
it is also possible to do the same quantization directly in Onnx, with Onnxruntime, in this case the model is reduced in size as in Pytorch, the gain in inference time measured on PC is less, as Onnx model is already more optimised than the models in Pytorch format.  
However, trying then to make inference with quantized Onnx model using tract-onnx, it does not work, as there are unsupported quantized layers.  
So it was not possible to use the quantized model on mobile to see the gain in inference time.

### Further Actions
- Verify it is possible to make inference with Onnxruntime in rust, since with Tract-Onnx it doesn't work, and quantization in python is done with Onnxruntime.
- Train a model with fewer parameters or a lower sample rate (16khz, 24khz), perhaps using the distillation process.


