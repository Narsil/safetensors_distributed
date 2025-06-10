use super::{Result, RedistributorError, SafetensorsIndex};
use crate::topology::{Topology, Tensor, SharedInfo};
use reqwest::{Client, header::HeaderMap};
use safetensors::tensor::Metadata;
use std::collections::{HashMap, BTreeMap};
use url::Url;
use log::trace;

impl super::core::AsyncTensorRedistributor {
    pub async fn load_or_create_remote_topology_with_cache(
        client: &Client,
        base_url: &Url,
        auth_headers: &HeaderMap,
    ) -> Result<(Topology, HashMap<String, Metadata>)> {
        // Try each method in order of preference
        if let Ok(topology) = Self::try_fetch_topology_json(client, base_url, auth_headers).await {
            trace!("Found topology.json, using existing topology");
            let metadata_cache = HashMap::new(); // Empty cache for existing topology
            return Ok((topology, metadata_cache));
        }

        trace!("No topology.json found, trying model.safetensors.index.json");
        if let Ok(topology) = Self::try_fetch_index_json(client, base_url, auth_headers).await {
            trace!("Found model.safetensors.index.json, creating topology from index");
            let metadata_cache = HashMap::new(); // Empty cache, will be populated as needed
            return Ok((topology, metadata_cache));
        }

        trace!("No index found, trying single model.safetensors");
        if let Ok((topology, metadata)) =
            Self::try_fetch_model_safetensors_with_cache(client, base_url, auth_headers).await
        {
            trace!("Found model.safetensors, creating single-file topology");
            let mut metadata_cache = HashMap::new();
            metadata_cache.insert("model.safetensors".to_string(), metadata);
            return Ok((topology, metadata_cache));
        }

        Err(RedistributorError::NoValidInput {
            path: base_url.path().into(),
        })
    }

    pub async fn load_or_create_remote_topology(
        client: &Client,
        base_url: &Url,
        auth_headers: &HeaderMap,
    ) -> Result<Topology> {
        let (topology, _) = Self::load_or_create_remote_topology_with_cache(client, base_url, auth_headers).await?;
        Ok(topology)
    }

    async fn try_fetch_topology_json(
        client: &Client,
        base_url: &Url,
        auth_headers: &HeaderMap,
    ) -> Result<Topology> {
        let topology_url = base_url.join("topology.json").map_err(|e| {
            RedistributorError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Invalid URL: {}", e),
            ))
        })?;

        trace!("Fetching topology from: {}", topology_url);

        let response = client
            .get(topology_url)
            .headers(auth_headers.clone())
            .send()
            .await
            .map_err(|e| {
                RedistributorError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to fetch topology: {}", e),
                ))
            })?;

        if !response.status().is_success() {
            return Err(RedistributorError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("topology.json not found (status: {})", response.status()),
            )));
        }

        let content = response.text().await.map_err(|e| {
            RedistributorError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to read topology response: {}", e),
            ))
        })?;

        let topology: Topology = serde_json::from_str(&content)?;
        Ok(topology)
    }

    async fn try_fetch_index_json(
        client: &Client,
        base_url: &Url,
        auth_headers: &HeaderMap,
    ) -> Result<Topology> {
        let index_url = base_url.join("model.safetensors.index.json").map_err(|e| {
            RedistributorError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Invalid URL: {}", e),
            ))
        })?;

        trace!("Fetching index from: {}", index_url);

        let response = client
            .get(index_url)
            .headers(auth_headers.clone())
            .send()
            .await
            .map_err(|e| {
                RedistributorError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to fetch index: {}", e),
                ))
            })?;

        if !response.status().is_success() {
            return Err(RedistributorError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("model.safetensors.index.json not found (status: {})", response.status()),
            )));
        }

        let content = response.text().await.map_err(|e| {
            RedistributorError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to read index response: {}", e),
            ))
        })?;

        let index: SafetensorsIndex = serde_json::from_str(&content)?;

        // Get unique filenames
        let mut filenames: Vec<String> = index.weight_map.values().cloned().collect();
        filenames.sort();
        filenames.dedup();

        // Create shared tensors for all tensors (since we don't know the distribution)
        let mut tensors = BTreeMap::new();
        for (tensor_name, _) in index.weight_map {
            // We'll fetch metadata later to get the actual tensor info
            // For now, create a placeholder that will be replaced when we load metadata
            tensors.insert(
                tensor_name,
                Tensor::Shared(SharedInfo::new(vec![1], safetensors::Dtype::F32, vec![0])),
            );
        }

        Ok(Topology::new(tensors, filenames, 1)?)
    }

    async fn try_fetch_model_safetensors_with_cache(
        client: &Client,
        base_url: &Url,
        auth_headers: &HeaderMap,
    ) -> Result<(Topology, Metadata)> {
        let model_url = base_url.join("model.safetensors").map_err(|e| {
            RedistributorError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Invalid URL: {}", e),
            ))
        })?;

        trace!("Fetching model metadata from: {}", model_url);

        let metadata = Self::fetch_remote_safetensors_metadata(client, &model_url, auth_headers).await?;

        // Create shared tensors
        let mut tensors = BTreeMap::new();
        for (name, info) in metadata.tensors() {
            tensors.insert(
                name.clone(),
                Tensor::Shared(SharedInfo::new(info.shape.clone(), info.dtype, vec![0])),
            );
        }

        let topology = Topology::new(tensors, vec!["model.safetensors".to_string()], 1)?;
        Ok((topology, metadata))
    }

    pub async fn fetch_remote_safetensors_metadata(
        client: &Client,
        file_url: &Url,
        auth_headers: &HeaderMap,
    ) -> Result<Metadata> {
        // First, get the header size by reading the first 8 bytes
        let response = client
            .get(file_url.clone())
            .headers(auth_headers.clone())
            .header("Range", "bytes=0-7")
            .send()
            .await
            .map_err(|e| {
                RedistributorError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to fetch metadata header: {}", e),
                ))
            })?;

        if !response.status().is_success() {
            return Err(RedistributorError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to fetch metadata header (status: {})", response.status()),
            )));
        }

        let header_bytes = response.bytes().await.map_err(|e| {
            RedistributorError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to read metadata header: {}", e),
            ))
        })?;

        if header_bytes.len() != 8 {
            return Err(RedistributorError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid header size",
            )));
        }

        let header_size = u64::from_le_bytes([
            header_bytes[0],
            header_bytes[1], 
            header_bytes[2],
            header_bytes[3],
            header_bytes[4],
            header_bytes[5],
            header_bytes[6],
            header_bytes[7],
        ]) as usize;

        // Now fetch the metadata
        let metadata_end = 8 + header_size - 1;
        let response = client
            .get(file_url.clone())
            .headers(auth_headers.clone())
            .header("Range", format!("bytes=8-{}", metadata_end))
            .send()
            .await
            .map_err(|e| {
                RedistributorError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to fetch metadata: {}", e),
                ))
            })?;

        if !response.status().is_success() {
            return Err(RedistributorError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to fetch metadata (status: {})", response.status()),
            )));
        }

        let metadata_bytes = response.bytes().await.map_err(|e| {
            RedistributorError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to read metadata: {}", e),
            ))
        })?;

        let metadata: Metadata = serde_json::from_slice(&metadata_bytes)?;
        Ok(metadata)
    }
} 