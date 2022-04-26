use crate::{InMessage, OutMessage};
use algebra::{
    fixed_point::{FixedPoint, FixedPointParameters},
    fp_64::Fp64Parameters,
    FpParameters, PrimeField, UniformRandom,
};
//use crypto_primitives::additive_share::{AuthShare, Share};
use crypto_primitives::{
    gc::{
        fancy_garbling,
        fancy_garbling::{
            circuit::{Circuit, CircuitBuilder},
            Encoder, GarbledCircuit, Wire,
        },
    },
};
use io_utils::imux::IMuxAsync;
use io_utils::imux::IMuxSync;
use neural_network::{
    layers::*,
    tensors::{Input, Output},
    Evaluate,
};

use protocols_sys::{SealClientACG, SealServerACG, *};
use rand::{CryptoRng, RngCore};
use std::{marker::PhantomData, os::raw::c_char};

use async_std::io::{Read, Write};

//new
use crate::{bytes, cds, error::MpcError, AdditiveShare,AuthAdditiveShare};
use algebra::{fields::near_mersenne_64::F,BigInteger64};
use crypto_primitives::{AuthShare, Share};
use scuttlebutt::Channel;
use ocelot::ot::{AlszReceiver as OTReceiver, AlszSender as OTSender, Receiver, Sender};
use rayon::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
/* 
pub struct ReluProtocol<P: FixedPointParameters> {
    _share: PhantomData<P>,
}

pub struct ReluProtocolType;
pub type ServerGcMsgSend<'a> = OutMessage<'a, (&'a [GarbledCircuit], &'a [Wire]), ReluProtocolType>;
pub type ClientGcMsgRcv = InMessage<(Vec<GarbledCircuit>, Vec<Wire>), ReluProtocolType>;
*/
pub struct ServerState<P: FixedPointParameters> {
    pub encoders: Vec<Encoder>,
    pub output_randomizers: Vec<P::Field>,
}

pub struct ClientState {
    pub gc_s: Vec<GarbledCircuit>,
    pub server_randomizer_labels: Vec<Wire>,
    pub client_input_labels: Vec<Wire>,
}

pub struct LinearProtocol<P: FixedPointParameters> {
    _share: PhantomData<P>,
}

pub struct LinearProtocolType;

pub type ServerGcMsgSend<'a> = OutMessage<'a, (&'a [GarbledCircuit], &'a [Wire]), LinearProtocolType>;
pub type ClientGcMsgRcv = InMessage<(Vec<GarbledCircuit>, Vec<Wire>), LinearProtocolType>;

// The message is a slice of (vectors of) input labels;
pub type ServerLabelMsgSend<'a> = OutMessage<'a, [Vec<Wire>], LinearProtocolType>;
pub type ClientLabelMsgRcv = InMessage<Vec<Vec<Wire>>, LinearProtocolType>;

pub type OfflineServerMsgSend<'a> = OutMessage<'a, Vec<c_char>, LinearProtocolType>;
pub type OfflineServerMsgRcv = InMessage<Vec<c_char>, LinearProtocolType>;

pub type OfflineClientMsgSend<'a> = OutMessage<'a, Vec<c_char>, LinearProtocolType>;
pub type OfflineClientMsgRcv = InMessage<Vec<c_char>, LinearProtocolType>;

pub type ClientShareMsgSend<'a, P> = OutMessage<'a, [AdditiveShare<P>], LinearProtocolType>;
pub type ServerShareMsgRcv<P> = InMessage<Vec<AdditiveShare<P>>, LinearProtocolType>;

pub type MsgSend<'a, P> = OutMessage<'a, Input<AdditiveShare<P>>, LinearProtocolType>;
pub type MsgRcv<P> = InMessage<Input<AdditiveShare<P>>, LinearProtocolType>;

pub type MacShareMsgSend<'a, P> = OutMessage<'a, AdditiveShare<P>, LinearProtocolType>;
pub type MacShareMsgRcv<P> = InMessage<AdditiveShare<P>, LinearProtocolType>;


struct TenBitExpParams {}
impl FixedPointParameters for TenBitExpParams {
    type Field = F;
    const MANTISSA_CAPACITY: u8 = 5;
    const EXPONENT_CAPACITY: u8 = 5;
}
type TenBitExpFP = FixedPoint<TenBitExpParams>;
const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

pub fn u128_from_share<P: FixedPointParameters>(s: AdditiveShare<P>) -> u128
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = BigInteger64>
{
    let s: u64 = s.inner.inner.into_repr().into();
    s.into()
}

fn generate_random_number<R: Rng>(rng: &mut R) -> (f64, TenBitExpFP) {
    let is_neg: bool = rng.gen();
    let mul = if is_neg { -10.0 } else { 10.0 };
    let float: f64 = rng.gen();
    let f = TenBitExpFP::truncate_float(float * mul);
    let n = TenBitExpFP::from(f);
    (f, n)
}


impl<P: FixedPointParameters> LinearProtocol<P>
where
    P: FixedPointParameters,
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
    P::Field: AuthShare,
{
    /// 运行 client ACG protocol by Garbled circuit. 产生随机遮蔽向量ri，生成混淆电路
    /// 发送给混淆电路及ri线标签，server返回计算结果，生成mac值并发送ss给server
    ///acg by garbled circuit
    /* 
    pub fn offline_client_acg_gc_protocol
    <
        R: Read + Send+ Unpin+ std::io::Read, 
        W: Write + Send+ Unpin+ std::io::Write, 
        RNG: CryptoRng + RngCore
        >(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        number_of_ACGs: usize,
        rng: &mut RNG,
        input_dims: (usize, usize, usize, usize),
        output_dims: (usize, usize, usize, usize),
    ) -> Result<ServerState<P>, bincode::Error> {
        let start_time = timer_start!(|| "预处理阶段客户端ACG协议(by GC)");

        // Client产生ri
        let mut ri= generate_random_number( rng).1;

        //生成mac
        let mac_key_r = P::Field::uniform(rng);//ri的俩mac
        let mac_key_y = P::Field::uniform(rng);//Miri-si的俩mac

        let mut gc_s = Vec::with_capacity(number_of_ACGs);
        let mut encoders = Vec::with_capacity(number_of_ACGs);
        let p = (<<P::Field as PrimeField>::Params>::MODULUS.0).into();
        let field_size = crypto_primitives::gc::num_bits(p);

        let mut b = CircuitBuilder::new();
        /*电路设计 */
        let c=b.finish();

        let garble_time = timer_start!(|| "电路混淆");
        //调用库混淆电路c，生活混淆后的电路gc，
        //以及编码参数en
        (0..number_of_ACGs)
            .into_par_iter()
            .map(|_| {
                let mut c = c.clone();
                let (en, gc) = fancy_garbling::garble(&mut c).unwrap();
                (en, gc)
            })
            .unzip_into_vecs(&mut encoders, &mut gc_s);
        timer_end!(garble_time);

        let encode_time = timer_start!(|| "对输入进行编码");
        //这是输入的个数
        let num_garbler_inputs = c.num_garbler_inputs();
        let num_evaluator_inputs = c.num_evaluator_inputs();

        //0,1标签个数
        let zero_inputs = vec![0u16; num_evaluator_inputs];
        let one_inputs = vec![1u16; num_evaluator_inputs];
        //线标签个数
        let mut labels = Vec::with_capacity(number_of_ACGs*num_evaluator_inputs);
        //server输入线标签
        let mut randomizer_labels = Vec::with_capacity(number_of_ACGs);
        let mut output_randomizers = Vec::with_capacity(number_of_ACGs);

        //编码输入标签
        for enc in encoders.iter() {
            let r = P::Field::uniform(rng);
            output_randomizers.push(r);
            let r_bits: u64 = ((-r).into_repr()).into();
            let r_bits = fancy_garbling::util::u128_to_bits(
                r_bits.into(),
                crypto_primitives::gc::num_bits(p),
            );
            for w in ((num_garbler_inputs / 2)..num_garbler_inputs)
                .zip(r_bits)
                .map(|(i, r_i)| enc.encode_garbler_input(r_i, i))
            {
                randomizer_labels.push(w);
            }

            //evaluator的0，1对应标签
            //labels为线标签
            let all_zeros = enc.encode_evaluator_inputs(&zero_inputs);
            let all_ones = enc.encode_evaluator_inputs(&one_inputs);
            all_zeros
                .into_iter()
                .zip(all_ones)
                .for_each(|(label_0, label_1)| {
                    labels.push((label_0.as_block(), label_1.as_block()))
                });
        }
        timer_end!(encode_time);

        let send_gc_time = timer_start!(|| "向服务器发送混淆电路GC");
        //如何？
        for msg_contents in gc_s
            .chunks(1)
            .zip(randomizer_labels.chunks(1))
        {
            let sent_message = ServerGcMsgSend::new(&msg_contents);
            crate::bytes::acg_serialize(writer, &sent_message)?;
        }
        timer_end!(send_gc_time);

        //OT协议传送标签
        if number_of_ACGs != 0 {
            let r = reader.get_mut_ref().remove(0);
            let w = writer.get_mut_ref().remove(0);

            let ot_time = timer_start!(|| "OT协议传送标签");
            let mut channel = Channel::new(r, w);
            let mut ot = OTSender::init(&mut channel, rng).unwrap();
            ot.send(&mut channel, labels.as_slice(), rng).unwrap();
            timer_end!(ot_time);
        }

        let encode_garbler_input_time = timer_start!(|| "对Garbler输入进行编码");
        //产生share
        let mut shares= Vec::with_capacity(1);
        //let mut rng1 = ChaChaRng::from_seed(RANDOMNESS);
        let (m, n1) = generate_random_number(rng);
        let x=m as u64;
        let share: AdditiveShare<P> =
            FixedPoint::new(P::Field::from_repr((x).into())).into();
        //let share:AdditiveShare<P>=n1.share(&mut rng1).1;
        shares.push(share);
        let mut en=&mut encoders;
        let wires = &shares
            .iter()
            .map(|share| {
                let share = u128_from_share(*share);
                fancy_garbling::util::u128_to_bits(share, field_size)
            })
            .zip(en)
            .map(|(share_bits, encoder)| encoder.encode_garbler_inputs(&share_bits))
            .collect::<Vec<Vec<_>>>();
        timer_end!(encode_garbler_input_time);

        let send_garbler_input_time = timer_start!(|| "发送Garbler输入线标签");
        let sent_message = ServerLabelMsgSend::new(wires.as_slice());
        crate::bytes::acg_serialize(writer, &sent_message);
        timer_end!(send_garbler_input_time);

        let recv_result_time = timer_start!(|| "接收混淆电路计算结果");
        let recv: ServerShareMsgRcv<P> = crate::bytes::acg_deserialize(reader)?;
        let results=recv.msg();
        timer_end!(recv_result_time);

        let comp_mac_time = timer_start!(|| "计算Mac");
        //获得mac密钥αi,βi
        let result= generate_random_number(rng).1;
        let alpha = generate_random_number(rng).1;//ri的mac:αi
        let beta = generate_random_number(rng).1;//Miri-si的mac:βi
        
        let mul1:FixedPoint<TenBitExpParams>=alpha*result;
        let mul2:FixedPoint<TenBitExpParams>=beta*ri;
        
        let (client_y_mac_share, server_y_mac_share) = (mul1).share( rng);//产生加性秘密共享！
        let (client_r_mac_share,server_r_mac_share)=(mul2).share(rng);
        timer_end!(comp_mac_time);

        let send_mac_time = timer_start!(|| "向server发送mac值");
        let sent_message_y = MacShareMsgSend::new(&server_y_mac_share);
        crate::bytes::acg_serialize(writer, &sent_message_y)?;
        let sent_message_r = MacShareMsgSend::new(&server_r_mac_share);
        crate::bytes::acg_serialize(writer, &sent_message_r)?;
        timer_end!(send_mac_time);
        
        timer_end!(start_time);
        
        Ok(ServerState {
            encoders,
            output_randomizers,
        })
    }

    /// 运行 server ACG protocol. 接收GC，ri标签，计算混淆电路，Miri-si
    ///返回计算结果给client
    pub fn offline_server_acg_gc_protocol
    <
        R: Read + Send+ Unpin+ std::io::Read, 
        W: Write + Send+ Unpin+ std::io::Write, 
        RNG: RngCore + CryptoRng
    >(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        number_of_ACGs: usize,
        shares: &[AdditiveShare<P>],
        rng: &mut RNG,
    ) -> Result<ClientState, bincode::Error> {
        use fancy_garbling::util::*;
        let start_time = timer_start!(|| "预处理阶段_服务器端_ACG协议(by GC)");
        let p = u128::from(<<P::Field as PrimeField>::Params>::MODULUS.0);
        let field_size = crypto_primitives::gc::num_bits(p);

        let rcv_gc_time = timer_start!(|| "接收混淆电路 GC");
        let mut gc_s = Vec::with_capacity(number_of_ACGs);
        let mut r_wires = Vec::with_capacity(number_of_ACGs);
        
        let in_msg: ClientGcMsgRcv = crate::bytes::acg_deserialize(reader)?;
        let (gc, r_wire) = in_msg.msg();
        gc_s.extend(gc);
        r_wires.extend(r_wire);//server输入线标签
        timer_end!(rcv_gc_time);

        //assert_eq!(gc_s.len(), number_of_ACGs);
        let bs = shares
            .iter()
            .flat_map(|s| u128_to_bits(u128_from_share(*s), field_size))
            .map(|b| b == 1)
            .collect::<Vec<_>>();
        //OT协议接收标签
        //client输入线标签
        let labels = if number_of_ACGs != 0 {
            let r = reader.get_mut_ref().remove(0);
            let w = writer.get_mut_ref().remove(0);

            let ot_time = timer_start!(|| "OT 协议接收标签");
            let mut channel = Channel::new(r, w);
            let mut ot = OTReceiver::init(&mut channel, rng).expect("should work");
            let labels = ot
                .receive(&mut channel, bs.as_slice(), rng)
                .expect("should work");
            let labels = labels
                .into_iter()
                .map(|l| Wire::from_block(l, 2))
                .collect::<Vec<_>>();
            timer_end!(ot_time);
            labels
        } else {
            Vec::new()
        };
        
        //需要gc_s,server_input_wires,client_input_wires
        let server_input_wires:&[Wire]= &r_wires;
        let client_input_wires:&[Wire]= &labels;
        let evaluators:&[GarbledCircuit]= &gc_s;

        let rcv_time = timer_start!(|| "接收server的输入线标签");
        let in_msg: ClientLabelMsgRcv = crate::bytes::acg_deserialize(reader)?;
        let mut garbler_wires = in_msg.msg();
        timer_end!(rcv_time);

        let eval_time = timer_start!(|| "计算混淆电路GC");
        let mut b = CircuitBuilder::new();
        /*电路设计 */
        let c=b.finish();
        let num_evaluator_inputs = c.num_evaluator_inputs();
        let num_garbler_inputs = c.num_garbler_inputs();
        garbler_wires
            .iter_mut()
            .zip(server_input_wires.chunks(num_garbler_inputs / 2))
            .for_each(|(w1, w2)| w1.extend_from_slice(w2));

        let mut results = client_input_wires
        .par_chunks(num_evaluator_inputs)
        .zip(garbler_wires)
        .zip(evaluators)
        .map(|((eval_inps, garbler_inps), gc)| {
            let mut c = c.clone();
            let result = gc
                .eval(&mut c, &garbler_inps, eval_inps)
                .expect("evaluation failed");
            use std::convert::TryFrom;
            let result = fancy_garbling::util::u128_from_bits(result.as_slice());
            FixedPoint::new(P::Field::from_repr(u64::try_from(result).unwrap().into())).into()
        })
        .collect::<Vec<AdditiveShare<P>>>();
        timer_end!(eval_time);

        let reslut_time = timer_start!(|| "将计算混淆电路GC结果发送给client");
        let sent_message = ClientShareMsgSend::new(&results);
        crate::bytes::acg_serialize(writer, &sent_message);
        timer_end!(reslut_time);

        let recv_mac_time = timer_start!(|| "接收mac值");
        let in_msg: MacShareMsgRcv<P> = crate::bytes::acg_deserialize(reader)?;
        let server_y_mac_share=in_msg.msg();
        let in_msg: MacShareMsgRcv<P> = crate::bytes::acg_deserialize(reader)?;
        let server_r_mac_share=in_msg.msg();
        timer_end!(recv_mac_time);

        timer_end!(start_time);

        //暂不改输出
        Ok(ClientState {
            gc_s,
            server_randomizer_labels: r_wires,//server input wires
            client_input_labels: labels,//client input wires!
        })
    }
 */
    pub fn offline_server_acg_protocol<
        R: Read + Send + Unpin,
        W: Write + Send + Unpin,
        RNG: RngCore + CryptoRng,
    >(
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        input_dims: (usize, usize, usize, usize),
        output_dims: (usize, usize, usize, usize),
        server_acg: &mut SealServerACG,
        rng: &mut RNG,
    ) -> Result<
        (
            (
                Input<AuthAdditiveShare<P::Field>>,
                Output<P::Field>,
                Output<AuthAdditiveShare<P::Field>>,
            ),
            (P::Field, P::Field),
        ),
        bincode::Error,
    > {
        // TODO: Add batch size
        let start_time = timer_start!(|| "Server linear offline protocol");
        let preprocess_time = timer_start!(|| "Preprocessing");

        // Sample MAC keys
        //server samples random MAC keys αi,βi！！！
        let mac_key_r = P::Field::uniform(rng);//ri的mac:αi？
        let mac_key_y = P::Field::uniform(rng);//Miri-si的mac:βi？

        // Sample server's randomness(si) for randomizing the i-th
        // layer MAC share and the i+1-th layer/MAC shares
        //3个随机值，分别代表什么？si
        let mut linear_share = Output::zeros(output_dims);//si
        let mut linear_mac_share = Output::zeros(output_dims);//αi<Miri-si>
        let mut r_mac_share = Input::zeros(input_dims);//βiri

        linear_share
            .iter_mut()
            .for_each(|e| *e = P::Field::uniform(rng));
        linear_mac_share
            .iter_mut()
            .for_each(|e| *e = P::Field::uniform(rng));
        r_mac_share
            .iter_mut()
            .for_each(|e| *e = P::Field::uniform(rng));

        // Create SEALServer object for C++ interopt
        // Preprocess filter rotations and noise masks
        server_acg.preprocess(
            &linear_share.to_u64(),
            &linear_mac_share.to_u64(),
            &r_mac_share.to_u64(),
            mac_key_y.into_repr().0,
            mac_key_r.into_repr().0,
        );
        timer_end!(preprocess_time);

        // Receive client Enc(r_i)
        let rcv_time = timer_start!(|| "Receiving Input");
        let client_share: OfflineServerMsgRcv = crate::bytes::deserialize(reader)?;
        let client_share_i = client_share.msg();//enc(ri)
        timer_end!(rcv_time);

        // Compute client's MAC share for layer `i`, and share + MAC share for layer `i
        // + 1`, That is, compute Lr - s, [a(Lr-s)]_1, [ar]_1
        let processing = timer_start!(|| "Processing Layer");
        let (linear_ct_vec, linear_mac_ct_vec, r_mac_ct_vec) = server_acg.process(client_share_i);
        timer_end!(processing);

        // Send shares to client
        let send_time = timer_start!(|| "Sending result");
        let sent_message = OfflineServerMsgSend::new(&linear_ct_vec);//发送enc(Miri-si)，下同
        crate::bytes::serialize(&mut *writer, &sent_message)?;
        let sent_message = OfflineServerMsgSend::new(&linear_mac_ct_vec);//enc αi（Miri-si）1
        crate::bytes::serialize(&mut *writer, &sent_message)?;
        let sent_message = OfflineServerMsgSend::new(&r_mac_ct_vec);//enc βi（ri）1
        crate::bytes::serialize(writer, &sent_message)?;
        timer_end!(send_time);

        // Collect shares and MACs into AuthAdditiveShares
        let linear_auth =
            Output::auth_share_from_parts(Output::zeros(output_dims), linear_mac_share);
        // Negate `r_mac_share` since client locally negates to get correct online share
        let r_auth = Input::auth_share_from_parts(Input::zeros(input_dims), -r_mac_share);

        timer_end!(start_time);
        //
        Ok(((r_auth, linear_share, linear_auth), (mac_key_r, mac_key_y)))
    }

    /// Runs client ACG protocol. Generates random input `r` and receives back
    /// authenticated shares of `r`, shares of `Lr`, and authenticated shares of
    /// shares of `Lr` --> [[r]]_1, <Lr>_1, [[<Lr>_1]]_1
    /// 
    /// ri,Miri-si,<biri>1,<ai(Miri-si)>1
    /// 
    /// 
    /// 
    pub fn offline_client_acg_protocol<
        R: Read + Send + Unpin,
        W: Write + Send + Unpin,
        RNG: RngCore + CryptoRng,
    >(
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        input_dims: (usize, usize, usize, usize),
        output_dims: (usize, usize, usize, usize),
        client_acg: &mut SealClientACG,
        rng: &mut RNG,
    ) -> Result<
        (
            Input<AuthAdditiveShare<P::Field>>,
            Output<AuthAdditiveShare<P::Field>>,
        ),
        bincode::Error,
    > {
        // TODO: Add batch size(单次传递给程序用以训练的参数个数)
        let start_time = timer_start!(|| "Linear offline protocol");
        let preprocess_time = timer_start!(|| "Client preprocessing");
        // Client产生ri
        let mut r: Input<FixedPoint<P>> = Input::zeros(input_dims);
        r.iter_mut()
            .for_each(|e| *e = FixedPoint::new(P::Field::uniform(&mut *rng)));

        // Create SEALClient object for C++ interopt
        // Preprocess and encrypt client secret share for sending
        //ct_vec:ciphertext
        let ct_vec = client_acg.preprocess(&r.to_repr());//预处理ri，即enc（pk，ri），或enc（ri）
        timer_end!(preprocess_time);

        // Send layer_i randomness(ri) for processing by server.
        let send_time = timer_start!(|| "Sending input");
        let sent_message = OfflineClientMsgSend::new(&ct_vec);//向server传送enc（ri）
        crate::bytes::serialize(writer, &sent_message)?;
        timer_end!(send_time);

        // Receive shares for layer `i + 1` output, MAC, and layer `i` MAC
        let rcv_time = timer_start!(|| "Receiving Result");
        let linear_ct: OfflineClientMsgRcv = crate::bytes::deserialize(&mut *reader)?;//接收enc(Miri-si)
        let linear_mac_ct: OfflineClientMsgRcv = crate::bytes::deserialize(&mut *reader)?;//enc(αi((Miri-si))1
        let r_mac_ct: OfflineClientMsgRcv = crate::bytes::deserialize(reader)?;//enc(βi(ri)1)
        timer_end!(rcv_time);

        let post_time = timer_start!(|| "Post-processing");
        let mut linear_auth = Output::zeros(output_dims);//要发给CDS用来认证(Miri-si)的
        let mut r_mac = Input::zeros(input_dims);
        // Decrypt + reshape resulting ciphertext and free C++ allocations
        //解密获得Miri-si、αi((Miri-si)1、αi(ri)1
        client_acg.decrypt(linear_ct.msg(), linear_mac_ct.msg(), r_mac_ct.msg());
        client_acg.postprocess::<P>(&mut linear_auth, &mut r_mac);//对αi((Miri-si)1和βi(ri)1进行处理

        // Negate both shares here so that we receive the correct
        // labels for the online phase
        //这是应该是要发给CDS对Miri-si和ri进行认证用的
        let r_auth = Input::auth_share_from_parts(-r.to_base(), -r_mac);//-ri和-βi(ri)1

        timer_end!(post_time);
        timer_end!(start_time);

        Ok((r_auth, linear_auth))
    }

    /// Client sends a value to the server and receives back a share of it's
    /// MAC's value
    pub fn offline_client_auth_share<R: Read + Send + Unpin, W: Write + Send + Unpin>(
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        input: Input<P::Field>,
        cfhe: &ClientFHE,
    ) -> Result<Input<AuthAdditiveShare<P::Field>>, bincode::Error> {
        let start_time = timer_start!(|| "Linear offline protocol");

        // Encrypt input and send to the server
        let mut share = SealCT::new();
        let ct = share.encrypt_vec(cfhe, input.to_u64().as_slice().unwrap().to_vec());

        let send_time = timer_start!(|| "Sending input");
        let sent_message = OfflineClientMsgSend::new(&ct);
        crate::bytes::serialize(writer, &sent_message)?;
        timer_end!(send_time);

        // Receive the result and decrypt
        let rcv_time = timer_start!(|| "Receiving Result");
        let auth_ct: OfflineClientMsgRcv = crate::bytes::deserialize(&mut *reader)?;
        timer_end!(rcv_time);

        let result = share
            .decrypt_vec(cfhe, auth_ct.msg(), input.len())
            .iter()
            .map(|e| P::Field::from_repr((*e).into()))
            .collect();
        let input_mac = Input::from_shape_vec(input.dim(), result).expect("Shapes should be same");
        timer_end!(start_time);
        Ok(Input::auth_share_from_parts(input, input_mac))
    }

    /// Server receives an encrypted vector from the client and shares its MAC
    pub fn offline_server_auth_share<
        R: Read + Send + Unpin,
        W: Write + Send + Unpin,
        RNG: RngCore + CryptoRng,
    >(
        reader: &mut IMuxAsync<R>,
        writer: &mut IMuxAsync<W>,
        input_dims: (usize, usize, usize, usize),
        sfhe: &ServerFHE,
        rng: &mut RNG,
    ) -> Result<(P::Field, Input<AuthAdditiveShare<P::Field>>), bincode::Error> {
        let start_time = timer_start!(|| "Linear offline protocol");

        // Sample MAC key and MAC share
        let mac_key = P::Field::uniform(rng);
        let mut mac_share = Input::zeros(input_dims);
        mac_share
            .iter_mut()
            .for_each(|e| *e = P::Field::uniform(rng));
        let mac_share_c: Vec<u64> = mac_share.to_u64().as_slice().unwrap().to_vec();

        // Receive client input and compute MAC share
        let mut share = SealCT::new();

        let rcv_time = timer_start!(|| "Receiving Input");
        let input: OfflineServerMsgRcv = crate::bytes::deserialize(&mut *reader)?;
        let mut input_ct = input.msg();
        timer_end!(rcv_time);

        share.inner.inner = input_ct.as_mut_ptr();
        share.inner.size = input_ct.len() as u64;
        let result_ct = share.gen_mac_share(sfhe, mac_share_c, mac_key.into_repr().0);

        // Send result back to client
        let send_time = timer_start!(|| "Sending Result");
        let sent_message = OfflineClientMsgSend::new(&result_ct);
        crate::bytes::serialize(writer, &sent_message)?;
        timer_end!(send_time);
        timer_end!(start_time);
        Ok((
            mac_key,
            Input::auth_share_from_parts(Input::zeros(input_dims), mac_share),
        ))
    }

    //不变
    pub fn online_client_protocol<W: Write + Send + Unpin>(
        writer: &mut IMuxAsync<W>,
        x_s: &Input<AdditiveShare<P>>,
        layer: &LinearLayerInfo<AdditiveShare<P>, FixedPoint<P>>,
    ) -> Result<(), bincode::Error> {
        let start = timer_start!(|| "Linear online protocol");
        match layer {
            LinearLayerInfo::Conv2d { .. } | LinearLayerInfo::FullyConnected => {
                let sent_message = MsgSend::new(x_s);
                crate::bytes::serialize(&mut *writer, &sent_message)?;
            }
            _ => {}
        }
        timer_end!(start);
        Ok(())
    }

    
    pub fn online_server_protocol<R: Read + Send + Unpin>(
        reader: &mut IMuxAsync<R>,
        layer: &LinearLayer<AdditiveShare<P>, FixedPoint<P>>,
        output_rerandomizer: &Output<P::Field>,
        input_derandomizer: &Input<P::Field>,
        output: &mut Output<AdditiveShare<P>>,
    ) -> Result<(), bincode::Error> {
        let start = timer_start!(|| "Linear online protocol");
        // Receive client share and compute layer if conv or fc
        let mut input: Input<AdditiveShare<P>> = match &layer {
            LinearLayer::Conv2d { .. } | LinearLayer::FullyConnected { .. } => {
                let recv: MsgRcv<P> = crate::bytes::deserialize(reader).unwrap();
                recv.msg()
            }
            _ => Input::zeros(input_derandomizer.dim()),
        };
        input.randomize_local_share(input_derandomizer);
        *output = layer.evaluate(&input);
        output.zip_mut_with(output_rerandomizer, |out, s| {
            *out = FixedPoint::randomize_local_share(out, s)
        });
        timer_end!(start);
        Ok(())
    }
}
