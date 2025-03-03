use const_format::concatcp;

pub static COCOINDEX_PREFIX: &str = "cocoindex.io/";

/// Expected mime types for bytes and str.
pub static _MIME_TYPE: &str = concatcp!(COCOINDEX_PREFIX, "mime_type");

/// Base text for chunks.
pub static CHUNK_BASE_TEXT: &str = concatcp!(COCOINDEX_PREFIX, "chunk_base_text");

/// Base text for an embedding vector.
pub static _VECTOR_ORIGIN_TEXT: &str = concatcp!(COCOINDEX_PREFIX, "vector_origin_text");
