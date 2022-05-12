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
