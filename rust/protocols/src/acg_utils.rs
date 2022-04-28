
use io_utils::imux::IMuxSync;
use io_utils::{counting::CountingIO};
use std::{
    io::{BufReader, BufWriter},
    net::{TcpListener, TcpStream},
};

pub fn acg_client_connect(
    addr: &str,
) -> (
    IMuxSync<CountingIO<BufReader<TcpStream>>>,
    IMuxSync<CountingIO<BufWriter<TcpStream>>>,
) {
    // TODO: Maybe change to rayon_num_threads
    let mut readers = Vec::with_capacity(16);
    let mut writers = Vec::with_capacity(16);
    for _ in 0..16 {
        println!("test client 1\n");
        let stream = TcpStream::connect(addr).unwrap();
        println!("test client 2\n");
        readers.push(CountingIO::new(BufReader::new(stream.try_clone().unwrap())));
        writers.push(CountingIO::new(BufWriter::new(stream)));
    }
    (IMuxSync::new(readers), IMuxSync::new(writers))
}

pub fn acg_server_connect(
    addr: &str,
) -> (
    IMuxSync<CountingIO<BufReader<TcpStream>>>,
    IMuxSync<CountingIO<BufWriter<TcpStream>>>,
) {
    println!("test server 1\n");
    let listener = TcpListener::bind(addr).unwrap();
    println!("test server 2\n");
    let mut incoming = listener.incoming();
    let mut readers = Vec::with_capacity(16);
    let mut writers = Vec::with_capacity(16);
    for _ in 0..16 {
        let stream = incoming.next().unwrap().unwrap();
        readers.push(CountingIO::new(BufReader::new(stream.try_clone().unwrap())));
        writers.push(CountingIO::new(BufWriter::new(stream)));
    }
    (IMuxSync::new(readers), IMuxSync::new(writers))
}


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

