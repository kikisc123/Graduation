use ::neural_network as nn;
use nn::{
    layers::{convolution::Padding, Layer},
    NeuralNetwork,
};
use rand::{CryptoRng, RngCore};

use super::*;

pub fn construct_mnist<R: RngCore + CryptoRng>(
    vs: Option<&tch::nn::Path>,
    batch_size: usize,
    rng: &mut R,
) -> NeuralNetwork<TenBitAS, TenBitExpFP> {
    let mut network = match &vs {
        Some(vs) => NeuralNetwork {
            layers: vec![],
            eval_method: ::neural_network::EvalMethod::TorchDevice(vs.device()),
        },
        None => NeuralNetwork {
            layers: vec![],
            ..Default::default()
        },
    };
    // Dimensions of input image.
    let input_dims = (batch_size, 1, 28, 28);

    let kernel_dims = (16, 1, 5, 5);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    network.layers.push(Layer::LL(conv));//第一层：卷积
    add_activation_layer(&mut network);//第二层：Relu

    let input_dims = network.layers.last().unwrap().output_dimensions();
    let pool = sample_avg_pool_layer(input_dims, (2, 2), 2);
    network.layers.push(Layer::LL(pool));//第三层池化

    let input_dims = network.layers.last().unwrap().output_dimensions();
    let kernel_dims = (16, 16, 5, 5);
    let conv = sample_conv_layer(vs, input_dims, kernel_dims, 1, Padding::Valid, rng).0;
    network.layers.push(Layer::LL(conv));//第四层卷积
    add_activation_layer(&mut network);//第五层Relu

    let input_dims = network.layers.last().unwrap().output_dimensions();
    let pool = sample_avg_pool_layer(input_dims, (2, 2), 2);
    network.layers.push(Layer::LL(pool));//第六层池化

    let fc_input_dims = network.layers.last().unwrap().output_dimensions();
    let (fc, _) = sample_fc_layer(vs, fc_input_dims, 100, rng);
    network.layers.push(Layer::LL(fc));//第七层全连接
    add_activation_layer(&mut network);//第八层relu

    let fc_input_dims = network.layers.last().unwrap().output_dimensions();
    let (fc, _) = sample_fc_layer(vs, fc_input_dims, 10, rng);
    network.layers.push(Layer::LL(fc));//第九层全连接层

    for layer in &network.layers {
        println!("Layer dim: {:?}", layer.input_dimensions());
    }

    assert!(network.validate());

    network
}
