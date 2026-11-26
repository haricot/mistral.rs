//! Resolve a metadata-only descriptor to an exact offline catalogue selection.
use anyhow::{ensure, Context, Result};
use mistralrs_kernel_runtime::model_scope::{resolve_model_scope, ModelCapabilityDescriptor, ProfileDependency, ProfileMetadata};
use serde::Deserialize;
use std::{fs::{File, OpenOptions}, io::{Read, Write}, path::Path};

#[derive(Deserialize)]
struct MasterInventory {
    schema: String,
    profiles: Vec<ProfileMetadata>,
}
fn read_json<T: serde::de::DeserializeOwned>(path:&Path, max:u64)->Result<T>{
    let mut bytes=Vec::new();
    File::open(path)?.take(max+1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64<=max,"input budget exceeded: {}",path.display());
    serde_json::from_slice(&bytes).with_context(||format!("invalid JSON: {}",path.display()))
}
fn main()->Result<()>{
    let args:Vec<_>=std::env::args_os().skip(1).collect();
    ensure!(args.len()==4,"usage: dkr_model_scope_resolve DESCRIPTOR INVENTORY DEPENDENCIES SELECTION_OUT");
    let descriptor:ModelCapabilityDescriptor=read_json(Path::new(&args[0]),8*1024*1024)?;
    let inventory:MasterInventory=read_json(Path::new(&args[1]),256*1024*1024)?;
    ensure!(inventory.schema=="DKR-MASTER-INVENTORY-V1","invalid inventory schema");
    let dependencies:Vec<ProfileDependency>=read_json(Path::new(&args[2]),32*1024*1024)?;
    let plan=resolve_model_scope(&descriptor,&inventory.profiles,&dependencies)?;
    let selection=plan.materialization_selection()?;
    let mut file=OpenOptions::new().write(true).create_new(true).open(Path::new(&args[3]))?;
    serde_json::to_writer_pretty(&mut file,selection)?;
    file.write_all(b"\n")?;
    println!("DKR_R126_CAPABILITY_RESOLUTION=COMPLETE profiles={} producer_coverage=DECLARED model_coverage=UNPROVEN",selection.profiles.len());
    Ok(())
}
