use crate::{InMessage, OutMessage};
use algebra::{
    fixed_point::{FixedPoint, FixedPointParameters},
    fp_64::Fp64Parameters,
    FpParameters, UniformRandom,
};
//use crypto_primitives::additive_share::{AuthShare, Share};
use crypto_primitives::{
    gc::{
        fancy_garbling,
        fancy_garbling::{
            //circuit::Circuit,
            circuit::CircuitBuilder,
            Encoder, GarbledCircuit, Wire,
        },
    },
};
use io_utils::imux::IMuxSync;
use neural_network::{
    tensors::{Input,Output},
};

use rand::{CryptoRng, RngCore};
use std::{marker::PhantomData, os::raw::c_char};

use std::io::{Read, Write};

//new
use crate::{AdditiveShare,AuthAdditiveShare};
//use crate::{bytes, cds, error::MpcError, AdditiveShare,AuthAdditiveShare};
use algebra::{fields::near_mersenne_64::F,BigInteger64,Fp64};
use algebra::{near_mersenne_64::FParameters,fields::PrimeField};
use crypto_primitives::{AuthShare, Share};
use scuttlebutt::Channel;
use ocelot::ot::{AlszReceiver as OTReceiver, AlszSender as OTSender, Receiver, Sender};
use rayon::prelude::*;
use rand::{Rng,SeedableRng};
use rand_chacha::ChaChaRng;

pub struct ACGProtocol<P: FixedPointParameters> {
    _share: PhantomData<P>,
}

pub struct ACGProtocolType;


pub type ServerGcMsgSend<'a> = OutMessage<'a, (&'a [GarbledCircuit], &'a [Wire]), ACGProtocolType>;
pub type ClientGcMsgRcv = InMessage<(Vec<GarbledCircuit>, Vec<Wire>), ACGProtocolType>;

// The message is a slice of (vectors of) input labels;
pub type ServerLabelMsgSend<'a> = OutMessage<'a, [Vec<Wire>], ACGProtocolType>;
pub type ClientLabelMsgRcv = InMessage<Vec<Vec<Wire>>, ACGProtocolType>;

pub type OfflineServerMsgSend<'a> = OutMessage<'a, Vec<c_char>, ACGProtocolType>;
pub type OfflineServerMsgRcv = InMessage<Vec<c_char>, ACGProtocolType>;

pub type OfflineClientMsgSend<'a> = OutMessage<'a, Vec<c_char>, ACGProtocolType>;
pub type OfflineClientMsgRcv = InMessage<Vec<c_char>, ACGProtocolType>;

pub type ClientShareMsgSend<'a, P> = OutMessage<'a, [AdditiveShare<P>], ACGProtocolType>;
pub type ServerShareMsgRcv<P> = InMessage<Vec<AdditiveShare<P>>, ACGProtocolType>;

pub type MsgSend<'a, P> = OutMessage<'a, Input<AdditiveShare<P>>, ACGProtocolType>;
pub type MsgRcv<P> = InMessage<Input<AdditiveShare<P>>, ACGProtocolType>;

pub type MacShareMsgSend<'a> = OutMessage<'a,AuthAdditiveShare<Fp64<FParameters>>, ACGProtocolType>;
pub type MacShareMsgRcv = InMessage<AuthAdditiveShare<Fp64<FParameters>>, ACGProtocolType>;


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


#[inline]
pub fn acg_serialize<W: std::io::Write + Send, T: ?Sized>(
    writer: &mut IMuxSync<W>,
    value: &T,
) -> Result<(), bincode::Error>
where
    T: serde::Serialize,
{
    let bytes: Vec<u8> = bincode::serialize(value)?;
    let _ = writer.write(&bytes)?;
    writer.flush()?;
    Ok(())
}

#[inline]
pub fn acg_deserialize<R, T>(reader: &mut IMuxSync<R>) -> bincode::Result<T>
where
    R: std::io::Read + Send,
    T: serde::de::DeserializeOwned,
{
    let bytes: Vec<u8> = reader.read()?;
    bincode::deserialize(&bytes[..])
}

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


impl<P: FixedPointParameters> ACGProtocol<P>
where
    P: FixedPointParameters,//+ crypto_primitives::AuthShare,
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
    P::Field: AuthShare,
{
    /// 运行 client ACG protocol by Garbled circuit. 产生随机遮蔽向量ri，生成混淆电路
    /// 发送给混淆电路及ri线标签，server返回计算结果，生成mac值并发送ss给server
    ///acg by garbled circuit
    
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
    ) ->Result<
        (
            (
                AuthAdditiveShare<Fp64<FParameters>>,
                Fp64<FParameters>,
                AuthAdditiveShare<Fp64<FParameters>>,
            ),
            (Fp64<FParameters>, Fp64<FParameters>),
        ),
        bincode::Error,
    > {
        let start_time = timer_start!(|| "预处理阶段客户端ACG协议(by GC)");

        // Client产生ri
        //let ri= generate_random_number( rng).1;

        //生成mac
        //let mac_key_r = P::Field::uniform(rng);//ri的俩mac
        //let mac_key_y = P::Field::uniform(rng);//Miri-si的俩mac

        let mut gc_s = Vec::with_capacity(number_of_ACGs);
        let mut encoders = Vec::with_capacity(number_of_ACGs);
        let p = u128::from(u64::from(P::Field::characteristic()));
        let field_size = (p.next_power_of_two() * 2).trailing_zeros() as usize;

        let mut b = CircuitBuilder::new();
        /*电路设计 */
        crypto_primitives::gc::relu::<P>(&mut b, 1).unwrap();
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
         //如果circuit为null的话，这里的label个数为0！运行出现逻辑错误！
        let randomizer_label_per_ACG = if number_of_ACGs == 0 {
            8192
        } else {
            randomizer_labels.len() / number_of_ACGs
        };
        for msg_contents in gc_s
            .chunks(8192)
            .zip(randomizer_labels.chunks(randomizer_label_per_ACG * 8192))
        {
            let sent_message = ServerGcMsgSend::new(&msg_contents);
            acg_serialize(writer, &sent_message)?;
            writer.flush().unwrap();
        }
        timer_end!(send_gc_time);
/* 
        add_to_trace!(|| "GC Communication", || format!(
            "Read {} bytes\nWrote {} bytes",
            reader.count(),
            writer.count()
        ));
        reader.reset();
        writer.reset();
*/

        //OT协议传送标签
        if number_of_ACGs != 0 {
            let r = reader.get_mut_ref().remove(0);
            let w = writer.get_mut_ref().remove(0);

            let ot_time = timer_start!(|| "OT协议传送标签");
            let mut channel = Channel::new(r, w);
            let mut ot = OTSender::init(&mut channel, rng).unwrap();
            println!("OT send 123\n");//运行到这了
            ot.send(&mut channel, labels.as_slice(), rng).unwrap();
            println!("OT send 2\n");
            timer_end!(ot_time);
        }

        let encode_garbler_input_time = timer_start!(|| "对Garbler输入进行编码");
        //产生share
        let mut shares= Vec::with_capacity(1);
        //let mut rng1 = ChaChaRng::from_seed(RANDOMNESS);
        let (m, _) = generate_random_number(rng);
        let x=m as u64;
        let share: AdditiveShare<P> =
            FixedPoint::new(P::Field::from_repr((x).into())).into();
        //let share:AdditiveShare<P>=n1.share(&mut rng1).1;
        shares.push(share);
        let en=&mut encoders;
        
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
        acg_serialize(writer, &sent_message)?;
        timer_end!(send_garbler_input_time);

        let recv_result_time = timer_start!(|| "接收混淆电路计算结果");
        let recv: ServerShareMsgRcv<P> = acg_deserialize(reader)?;
        let results=recv.msg();
        timer_end!(recv_result_time);

        let comp_mac_time = timer_start!(|| "计算Mac");
        //获得mac密钥αi,βi

        let mut rng = ChaChaRng::from_seed(RANDOMNESS);
        let ri=F::uniform(&mut rng);
        let result = F::uniform(&mut rng);
        let alpha = F::uniform(&mut rng);//ri的mac:αi
        let beta = F::uniform(&mut rng);//Miri-si的mac:βi
        
        let (client_y_mac_share, server_y_mac_share) = (result).auth_share(&alpha,&mut rng);//产生认证的加性秘密共享！
        let (client_r_mac_share,server_r_mac_share)=(ri).auth_share(&beta,&mut rng);
        timer_end!(comp_mac_time);

        let send_mac_time = timer_start!(|| "向server发送mac值");
        let sent_message_y = MacShareMsgSend::new(&server_y_mac_share);
        acg_serialize(writer, &sent_message_y)?;
        let sent_message_r = MacShareMsgSend::new(&server_r_mac_share);
        acg_serialize(writer, &sent_message_r)?;
        timer_end!(send_mac_time);
        
        timer_end!(start_time);
        
        Ok(
            (
                (server_r_mac_share, result, server_r_mac_share), 
                (alpha, beta)
            )
        )
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
    ) ->Result<
        (
            AuthAdditiveShare<Fp64<FParameters>>,
            AuthAdditiveShare<Fp64<FParameters>>,
        ),
        bincode::Error,
    >   {
        use fancy_garbling::util::*;
        let start_time = timer_start!(|| "预处理阶段_服务器端_ACG协议(by GC)");
        let p = u128::from(<<P::Field as PrimeField>::Params>::MODULUS.0);
        let field_size = crypto_primitives::gc::num_bits(p);

        let rcv_gc_time = timer_start!(|| "接收混淆电路 GC");
        let mut gc_s = Vec::with_capacity(number_of_ACGs);
        let mut r_wires = Vec::with_capacity(number_of_ACGs);
        
        let num_chunks = (number_of_ACGs as f64 / 8192.0).ceil() as usize;
        for i in 0..num_chunks {
            let in_msg: ClientGcMsgRcv = acg_deserialize(reader)?;
            let (gc_chunks, r_wire_chunks) = in_msg.msg();
            
            gc_s.extend(gc_chunks);
            r_wires.extend(r_wire_chunks);
        }
        timer_end!(rcv_gc_time);

        //assert_eq!(gc_s.len(), number_of_ACGs);
        use num_traits::identities::Zero;
        let shares = vec![AdditiveShare::<TenBitExpParams>::zero(); number_of_ACGs];
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
            println!("OT recv 1\n");//OT传不过去？
            let labels = ot
                .receive(&mut channel, bs.as_slice(), rng)
                .expect("should work");
            println!("OT recv 2\n");
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
        let in_msg: ClientLabelMsgRcv = acg_deserialize(reader)?;
        let mut garbler_wires = in_msg.msg();
        timer_end!(rcv_time);

        let eval_time = timer_start!(|| "计算混淆电路GC");
        let b = CircuitBuilder::new();
        /*电路设计 */
        let c=b.finish();
        let num_evaluator_inputs = c.num_evaluator_inputs();
        let num_garbler_inputs = c.num_garbler_inputs();
        garbler_wires
            .iter_mut()
            .zip(server_input_wires.chunks(num_garbler_inputs / 2))
            .for_each(|(w1, w2)| w1.extend_from_slice(w2));

        let results = client_input_wires
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
        acg_serialize(writer, &sent_message)?;
        timer_end!(reslut_time);

        let recv_mac_time = timer_start!(|| "接收mac值");
        let in_msg: MacShareMsgRcv = acg_deserialize(reader)?;
        let server_y_mac_share=in_msg.msg();
        let in_msg: MacShareMsgRcv = acg_deserialize(reader)?;
        let server_r_mac_share=in_msg.msg();
        timer_end!(recv_mac_time);

        timer_end!(start_time);

        //暂不改输出
        Ok(
            (   server_r_mac_share, 
                server_y_mac_share
            )
        )
        
    }
}
