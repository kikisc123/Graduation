use crate::*;
use algebra::{
    fields::near_mersenne_64::F, fixed_point::FixedPointParameters, FpParameters, PrimeField,
    UniformRandom,
};
use crypto_primitives::{
    gc::fancy_garbling::{Encoder, GarbledCircuit},
    AuthAdditiveShare,
};
use futures::stream::StreamExt;
use io_utils::{counting::CountingIO, imux::IMuxAsync};
//use io_utils::imux::IMuxSync;
use neural_network::{
    tensors::{Input, Output},
    NeuralNetwork,
};
use num_traits::identities::Zero;
use protocols::{
    acg::ACGProtocol,
    gc::ServerGcMsgSend,
    linear_layer::LinearProtocol,
    mpc::{ServerMPC, MPC},
    mpc_offline::{OfflineMPC, ServerOfflineMPC},
    neural_network::NNProtocol,
    server_keygen,
};
use protocols_sys::{server_acg, SealServerACG, ServerACG};
use rand::{CryptoRng, RngCore};
use rayon::prelude::*;
use scuttlebutt::Block;
use std::collections::BTreeMap;

use async_std::{
    io::{BufReader, BufWriter},
    net::{TcpListener, TcpStream},
    task,
};

pub use fancy_garbling;
/* 
use fancy_garbling::{
    circuit::CircuitBuilder,
    error::{CircuitBuilderError, FancyError},
    util, BinaryBundle, BinaryGadgets, Bundle, BundleGadgets, Fancy,
};
*/

pub fn server_connect(
    addr: &str,
) -> (
    IMuxAsync<CountingIO<BufReader<TcpStream>>>,
    IMuxAsync<CountingIO<BufWriter<TcpStream>>>,
) {
    task::block_on(async {
        // TODO: Maybe change to rayon_num_threads
        let listener = TcpListener::bind(addr).await.unwrap();
        let mut incoming = listener.incoming();
        let mut readers = Vec::with_capacity(16);
        let mut writers = Vec::with_capacity(16);
        for _ in 0..16 {
            let stream = incoming.next().await.unwrap().unwrap();
            readers.push(CountingIO::new(BufReader::new(stream.clone())));
            writers.push(CountingIO::new(BufWriter::new(stream)));
        }
        (IMuxAsync::new(readers), IMuxAsync::new(writers))
    })
}

pub fn nn_server<R: RngCore + CryptoRng>(
    server_addr: &str,
    nn: NeuralNetwork<TenBitAS, TenBitExpFP>,
    rng: &mut R,
) {
    let (server_offline_state, offline_read, offline_write) = {
        let (mut reader, mut writer) = server_connect(server_addr);
        (
            NNProtocol::offline_server_protocol(&mut reader, &mut writer, &nn, rng).unwrap(),
            reader.count(),
            writer.count(),
        )
    };

    let (_, online_read, online_write) = {
        let (mut reader, mut writer) = server_connect(server_addr);
        (
            NNProtocol::online_server_protocol(
                &mut reader,
                &mut writer,
                &nn,
                &server_offline_state,
            )
            .unwrap(),
            reader.count(),
            writer.count(),
        )
    };
    add_to_trace!(|| "Offline Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        offline_read, offline_write
    ));
    add_to_trace!(|| "Online Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        online_read, online_write
    ));
}

// TODO: Pull out this functionality in `neural_network.rs` so this is clean
pub fn acg<R: RngCore + CryptoRng>(
    server_addr: &str,
    nn: NeuralNetwork<TenBitAS, TenBitExpFP>,
    rng: &mut R,
) {
    //连接server和client
    let (mut reader, mut writer) = server_connect(server_addr);
  
    // Keygen
    let sfhe = server_keygen(&mut reader).unwrap();//接收FHE所需key
    reader.reset();

    let mut linear_shares: BTreeMap<
        usize,
        (
            Input<AuthAdditiveShare<F>>,
            Output<F>,
            Output<AuthAdditiveShare<F>>,
        ),
    > = BTreeMap::new();//第一个，要输出的
    let mut mac_keys: BTreeMap<usize, (F, F)> = BTreeMap::new();//第二个

    let linear_time = timer_start!(|| "Linear layers offline phase");
    for (i, layer) in nn.layers.iter().enumerate() {
        match layer {
            Layer::NLL(NonLinearLayer::ReLU { .. }) => {}
            Layer::LL(layer) => {
                let (shares, keys) = match &layer {//该层的acg输出，核心，下面所有都是为了产生
                    LinearLayer::Conv2d { .. } | LinearLayer::FullyConnected { .. } => {
                        let mut acg_handler = match &layer {
                            LinearLayer::Conv2d { .. } => SealServerACG::Conv2D(
                                server_acg::Conv2D::new(&sfhe, &layer, &layer.kernel_to_repr()),
                            ),
                            LinearLayer::FullyConnected { .. } => {
                                SealServerACG::FullyConnected(server_acg::FullyConnected::new(
                                    &sfhe,
                                    &layer,
                                    &layer.kernel_to_repr(),
                                ))
                            }
                            _ => unreachable!(),
                        };
                        //返回那两个值
                        //返回值shares and keys
                        reader.reset();
                        writer.reset();
                        let((r_auth, linear_share, linear_auth), (mac_key_r, mac_key_y))=
                        LinearProtocol::<TenBitExpParams>::offline_server_acg_protocol(
                            &mut reader,
                            &mut writer,
                            layer.input_dimensions(),
                            layer.output_dimensions(),
                            &mut acg_handler,
                            rng,
                        )
                        .unwrap();
                        add_to_trace!(|| "Communication test", || format!(
                            "Read {} bytes\nWrote {} bytes",
                            reader.count(),
                            writer.count()
                        ));
                        ((r_auth, linear_share, linear_auth), (mac_key_r, mac_key_y))
                    }
                    //如果是池化层
                    LinearLayer::AvgPool { dims, .. } | LinearLayer::Identity { dims } => {
                        let in_zero = Output::zeros(dims.input_dimensions());
                        if linear_shares.keys().any(|k| k == &(i - 1)) {
                            // If the layer comes after a linear layer, apply the function to
                            // the last layer's output share MAC
                            let prev_mac_keys = mac_keys.get(&(i - 1)).unwrap();
                            let prev_output_share = &linear_shares.get(&(i - 1)).unwrap().2;
                            let mut output_share = Output::zeros(dims.output_dimensions());
                            layer.evaluate_naive_auth(prev_output_share, &mut output_share);
                            (//返回值
                                //shares3个
                                (
                                    Input::auth_share_from_parts(in_zero.clone(), in_zero.clone()),
                                    Output::zeros(dims.output_dimensions()),
                                    output_share,
                                ),
                                //keys2个
                                prev_mac_keys.clone(),
                            )
                        } else {
                            // If the layer comes after a non-linear layer, receive the
                            // randomizer from the client, authenticate it, and then apply the
                            // function to the MAC share
                            let (key, input_share) ={
                                let (mut reader1, mut writer1) = server_connect(server_addr);
                                LinearProtocol::<TenBitExpParams>::offline_server_auth_share(
                                &mut reader1,
                                &mut writer1,
                                dims.input_dimensions(),
                                &sfhe,
                                rng,
                            )
                            .unwrap()};
                                
                            let mut output_share = Output::zeros(dims.output_dimensions());
                            layer.evaluate_naive_auth(&input_share, &mut output_share);
                            (//返回值
                                //shares3个
                                (
                                    -input_share,
                                    Output::zeros(dims.output_dimensions()),
                                    output_share,
                                ),
                                //keys2个
                                (key, key),
                            )
                        }
                    }
                };
                linear_shares.insert(i, shares);//shares:
                mac_keys.insert(i, keys);//keys:αi,βi
            }
        }
    }
    timer_end!(linear_time);

    add_to_trace!(|| "Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        reader.count(),
        writer.count()
    ));
}

pub fn garbling<R: RngCore + CryptoRng>(server_addr: &str, layers: &[usize], rng: &mut R) {
    let (reader, mut writer) = server_connect(server_addr);

    let activations: usize = layers.iter().map(|e| *e).sum();
    let _garble_time = timer_start!(|| "Garbling Time");
    //混淆电路的数量
    let mut gc_s = Vec::with_capacity(activations);
    //编码器的数量，这两个肯定和激活函数的数量是一致的呀！
    let mut encoders = Vec::with_capacity(activations);

    let p = (<<F as PrimeField>::Params>::MODULUS.0).into();

    assert_eq!(activations, layers.iter().fold(0, |sum, &x| sum + x));
    let garble_time = timer_start!(|| "Garbling");

    // For each layer, garbled a circuit with the correct number of truncations
    for num in layers.iter() {
        //每一层创造一个circuit
        let c = protocols::gc::make_truncated_relu::<TenBitExpParams>(
            TenBitExpParams::EXPONENT_CAPACITY,
        );
        let (en, gc): (Vec<Encoder>, Vec<GarbledCircuit>) = (0..*num)
            .into_par_iter()
            .map(move |_| {
                let mut c = c.clone();
                let (en, gc) = fancy_garbling::garble(&mut c).unwrap();
                (en, gc)
            })
            .unzip();
        encoders.extend(en);
        gc_s.extend(gc);
    }
    timer_end!(garble_time);

    let encode_time = timer_start!(|| "Encoding inputs");
    let (num_garbler_inputs, num_evaluator_inputs) = if activations > 0 {
        (
            encoders[0].num_garbler_inputs(),
            encoders[0].num_evaluator_inputs(),
        )
    } else {
        (0, 0)
    };

    let zero_inputs = vec![0u16; num_evaluator_inputs];
    let one_inputs = vec![1u16; num_evaluator_inputs];
    let mut labels = Vec::with_capacity(activations * num_evaluator_inputs);
    let mut randomizer_labels = Vec::with_capacity(activations);
    let mut output_randomizers = Vec::with_capacity(activations);
    for enc in encoders.iter() {
        // Output server randomization share
        let r = F::uniform(rng);
        output_randomizers.push(r);
        let r_bits: u64 = ((-r).into_repr()).into();
        let r_bits =
            fancy_garbling::util::u128_to_bits(r_bits.into(), crypto_primitives::gc::num_bits(p));
        for w in ((num_garbler_inputs / 2)..num_garbler_inputs)
            .zip(r_bits)
            .map(|(i, r_i)| enc.encode_garbler_input(r_i, i))
        {
            randomizer_labels.push(w);
        }

        let all_zeros = enc.encode_evaluator_inputs(&zero_inputs);
        let all_ones = enc.encode_evaluator_inputs(&one_inputs);
        all_zeros
            .into_iter()
            .zip(all_ones)
            .for_each(|(label_0, label_1)| labels.push((label_0.as_block(), label_1.as_block())));
    }

    // Extract out the zero labels for the carry bits since these aren't
    // used in CDS
    let (carry_labels, input_labels): (Vec<_>, Vec<_>) = labels
        .into_iter()
        .enumerate()
        .partition(|(i, _)| (i + 1) % (F::size_in_bits() + 1) == 0);

    let _carry_labels: Vec<Block> = carry_labels
        .into_iter()
        .map(|(_, (zero, _))| zero)
        .collect();
    let _input_labels: Vec<(Block, Block)> = input_labels.into_iter().map(|(_, l)| l).collect();
    timer_end!(encode_time);

    let send_gc_time = timer_start!(|| "Sending GCs");
    let randomizer_label_per_relu = if activations == 0 {
        8192
    } else {
        randomizer_labels.len() / activations
    };
    for msg_contents in gc_s
        .chunks(8192)
        .zip(randomizer_labels.chunks(randomizer_label_per_relu * 8192))
    {
        let sent_message = ServerGcMsgSend::new(&msg_contents);
        protocols::bytes::serialize(&mut writer, &sent_message).unwrap();
    }
    timer_end!(send_gc_time);
    timer_end!(garble_time);
    add_to_trace!(|| "Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        reader.count(),
        writer.count()
    ));
}

pub fn triples_gen<R: RngCore + CryptoRng>(server_addr: &str, num: usize, rng: &mut R) {
    let (mut reader, mut writer) = server_connect(server_addr);

    // Keygen
    let sfhe = server_keygen(&mut reader).unwrap();
    let mac_key = F::uniform(rng);
    reader.reset();

    // Generate triples
    let server_gen = ServerOfflineMPC::<F, _>::new(&sfhe, mac_key.into_repr().0);
    let triples = timer_start!(|| "Generating triples");
    server_gen.triples_gen(&mut reader, &mut writer, rng, num);
    timer_end!(triples);
    add_to_trace!(|| "Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        reader.count(),
        writer.count()
    ));
}

pub fn cds<R: RngCore + CryptoRng>(server_addr: &str, layers: &[usize], rng: &mut R) {
    let (mut reader, mut writer) = server_connect(server_addr);

    // Keygen
    let sfhe = server_keygen(&mut reader).unwrap();
    reader.reset();

    // Generate dummy labels/layer for CDS
    let activations: usize = layers.iter().map(|e| *e).sum();
    let modulus_bits = <F as PrimeField>::size_in_bits();

    let out_mac_keys = vec![F::zero(); layers.len()];
    let out_mac_shares = vec![F::zero(); activations];
    let inp_mac_keys = vec![F::zero(); layers.len()];
    let inp_mac_shares = vec![F::zero(); activations];
    let labels: Vec<(Block, Block)> = vec![(0.into(), 0.into()); 2 * activations * modulus_bits];

    // Generate triples
    protocols::cds::CDSProtocol::<TenBitExpParams>::server_cds(
        &mut reader,
        &mut writer,
        &sfhe,
        layers,
        &out_mac_keys,
        &out_mac_shares,
        &inp_mac_keys,
        &inp_mac_shares,
        labels.as_slice(),
        rng,
    )
    .unwrap();
    add_to_trace!(|| "Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        reader.count(),
        writer.count()
    ));
}

pub fn input_auth<R: RngCore + CryptoRng>(server_addr: &str, layers: &[usize], rng: &mut R) {
    let (mut reader, mut writer) = server_connect(server_addr);

    // Keygen
    let sfhe = server_keygen(&mut reader).unwrap();
    reader.reset();

    // Generate dummy labels/layer for CDS
    let activations: usize = layers.iter().map(|e| *e).sum();
    let modulus_bits = <F as PrimeField>::size_in_bits();
    let elems_per_label = (128.0 / (modulus_bits - 1) as f64).ceil() as usize;

    let out_mac_keys = vec![F::zero(); layers.len()];
    let out_mac_shares = vec![F::zero(); activations];
    let inp_mac_keys = vec![F::zero(); layers.len()];
    let inp_mac_shares = vec![F::zero(); activations];
    let zero_labels = vec![F::zero(); 2 * activations * modulus_bits * elems_per_label];
    let one_labels = vec![F::zero(); 2 * activations * modulus_bits * elems_per_label];

    let num_rands = 2 * (activations + activations * modulus_bits);

    // Generate rands
    let mac_key = F::uniform(rng);
    let gen = ServerOfflineMPC::new(&sfhe, mac_key.into_repr().0);

    let input_time = timer_start!(|| "Input Auth");
    let rands = gen.rands_gen(&mut reader, &mut writer, rng, num_rands);
    let mut mpc = ServerMPC::new(rands, Vec::new(), mac_key);

    // Share inputs
    let share_time = timer_start!(|| "Server sharing inputs");
    let _out_mac_keys = mpc
        .private_inputs(&mut reader, &mut writer, out_mac_keys.as_slice(), rng)
        .unwrap();
    let _inp_mac_keys = mpc
        .private_inputs(&mut reader, &mut writer, inp_mac_keys.as_slice(), rng)
        .unwrap();
    let _out_mac_shares = mpc
        .private_inputs(&mut reader, &mut writer, out_mac_shares.as_slice(), rng)
        .unwrap();
    let _inp_mac_shares = mpc
        .private_inputs(&mut reader, &mut writer, inp_mac_shares.as_slice(), rng)
        .unwrap();
    let _zero_labels = mpc
        .private_inputs(&mut reader, &mut writer, zero_labels.as_slice(), rng)
        .unwrap();
    let _one_labels = mpc
        .private_inputs(&mut reader, &mut writer, one_labels.as_slice(), rng)
        .unwrap();
    timer_end!(share_time);

    // Receive client shares
    let recv_time = timer_start!(|| "Server receiving inputs");
    let _out_bits = mpc
        .recv_private_inputs(&mut reader, &mut writer, activations * modulus_bits)
        .unwrap();
    let _inp_bits = mpc
        .recv_private_inputs(&mut reader, &mut writer, activations * modulus_bits)
        .unwrap();
    let _c_out_mac_shares = mpc
        .recv_private_inputs(&mut reader, &mut writer, activations)
        .unwrap();
    let _c_inp_mac_shares = mpc
        .recv_private_inputs(&mut reader, &mut writer, activations)
        .unwrap();
    timer_end!(recv_time);
    timer_end!(input_time);
    add_to_trace!(|| "Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        reader.count(),
        writer.count()
    ));
}

pub fn acg_gc<R: RngCore + CryptoRng>(
    server_addr: &str,
    nn: NeuralNetwork<TenBitAS, TenBitExpFP>,
    rng: &mut R,
) {
    //连接server和client
    //let (mut reader, mut writer) = acg_server_connect(server_addr);
    //转到ACG协议实现时连接！

    let mut linear_shares = BTreeMap::new();
    let mut mac_keys = BTreeMap::new();
    let linear_time = timer_start!(|| "预处理阶段线性层");
    for (i, layer) in nn.layers.iter().enumerate() {
        match layer {
            Layer::NLL(NonLinearLayer::ReLU { .. }) => {}
            Layer::LL(layer) => {
                let (shares, keys) = match &layer {//该层的acg输出，产生auth additive share。
                    LinearLayer::Conv2d { .. } | LinearLayer::FullyConnected { .. } |LinearLayer::AvgPool { .. } | LinearLayer::Identity { .. }=> {                         
                        //如果是卷积层或者全连接层，执行ACG协议,返回认证的秘密共享，以及mac密钥
                        //返回值要发给CDS用来对ri和Miri-si进行认证用的
                        //产生share
                        //返回值shares and keys
                        let number_of_ACGs=1;
                        ACGProtocol::<TenBitExpParams>::offline_client_acg_gc_protocol(
                            &server_addr,
                            //&mut reader,
                            //&mut writer,
                            number_of_ACGs,
                            rng,
                        )
                        .unwrap()
                    }
                    
                };
                linear_shares.insert(i, shares);
                mac_keys.insert(i, keys);
            }
        }
    }
    timer_end!(linear_time);
 
}