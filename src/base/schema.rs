use crate::builder::plan::AnalyzedValueMapping;

use super::spec::*;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    ops::Deref,
    sync::{Arc, LazyLock},
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VectorTypeSchema {
    pub element_type: Box<BasicValueType>,
    pub dimension: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind")]
pub enum BasicValueType {
    /// A sequence of bytes in binary.
    Bytes,

    /// String encoded in UTF-8.
    Str,

    /// A boolean value.
    Bool,

    /// 64-bit integer.
    Int64,

    /// 32-bit floating point number.
    Float32,

    /// 64-bit floating point number.
    Float64,

    /// A range, with a start offset and a length.
    Range,

    /// A JSON value.
    Json,

    /// A vector of values (usually numbers, for embeddings).
    Vector(VectorTypeSchema),
}

impl std::fmt::Display for BasicValueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BasicValueType::Bytes => write!(f, "bytes"),
            BasicValueType::Str => write!(f, "str"),
            BasicValueType::Bool => write!(f, "bool"),
            BasicValueType::Int64 => write!(f, "int64"),
            BasicValueType::Float32 => write!(f, "float32"),
            BasicValueType::Float64 => write!(f, "float64"),
            BasicValueType::Range => write!(f, "range"),
            BasicValueType::Json => write!(f, "json"),
            BasicValueType::Vector(s) => write!(
                f,
                "vector({}, {})",
                s.dimension
                    .map(|d| d.to_string())
                    .unwrap_or_else(|| "*".to_string()),
                s.element_type
            ),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StructSchema {
    pub fields: Arc<Vec<FieldSchema>>,
}

impl StructSchema {
    pub fn without_attrs(&self) -> Self {
        Self {
            fields: Arc::new(self.fields.iter().map(|f| f.without_attrs()).collect()),
        }
    }
}

impl std::fmt::Display for StructSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Struct(")?;
        for (i, field) in self.fields.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", field)?;
        }
        write!(f, ")")
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CollectionKind {
    /// A generic collection can have any row type.
    Collection,
    /// A table's first field is the key.
    Table,
    /// A list is a table whose key type is int64 starting from 0 continuously..
    List,
}

impl std::fmt::Display for CollectionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CollectionKind::Collection => write!(f, "Collection"),
            CollectionKind::Table => write!(f, "Table"),
            CollectionKind::List => write!(f, "List"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CollectionSchema {
    pub kind: CollectionKind,
    pub row: StructSchema,

    #[serde(default = "Vec::new", skip_serializing_if = "Vec::is_empty")]
    pub collectors: Vec<NamedSpec<StructSchema>>,
}

impl CollectionSchema {
    pub fn has_key(&self) -> bool {
        match self.kind {
            CollectionKind::Collection => false,
            CollectionKind::Table | CollectionKind::List => true,
        }
    }

    pub fn key_type(&self) -> Option<&EnrichedValueType> {
        match self.kind {
            CollectionKind::Collection => None,
            CollectionKind::Table => self
                .row
                .fields
                .first()
                .as_ref()
                .map(|field| &field.value_type),
            CollectionKind::List => Some(&LIST_INDEX_FIELD.value_type),
        }
    }

    pub fn without_attrs(&self) -> Self {
        Self {
            kind: self.kind,
            row: self.row.without_attrs(),
            collectors: self
                .collectors
                .iter()
                .map(|c| NamedSpec {
                    name: c.name.clone(),
                    spec: c.spec.without_attrs(),
                })
                .collect(),
        }
    }
}

impl std::fmt::Display for CollectionSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({}", self.kind, self.row)?;
        for collector in self.collectors.iter() {
            write!(f, "; COLLECTOR {} ({})", collector.name, collector.spec)?;
        }
        write!(f, ")")?;
        Ok(())
    }
}

pub const KEY_FIELD_NAME: &'static str = "__key";
pub const VALUE_FIELD_NAME: &'static str = "__value";
pub const LIST_INDEX_FIELD_NAME: &'static str = "__index";

pub static LIST_INDEX_FIELD: LazyLock<FieldSchema> = LazyLock::new(|| FieldSchema {
    name: LIST_INDEX_FIELD_NAME.to_string(),
    value_type: EnrichedValueType {
        typ: ValueType::Basic(BasicValueType::Int64),
        nullable: false,
        attrs: Default::default(),
    },
});

impl CollectionSchema {
    pub fn new_collection(value_name: Option<String>, value: EnrichedValueType) -> Self {
        Self {
            kind: CollectionKind::Collection,
            row: StructSchema {
                fields: Arc::new(vec![FieldSchema {
                    name: value_name.unwrap_or_else(|| VALUE_FIELD_NAME.to_string()),
                    value_type: value,
                }]),
            },
            collectors: Default::default(),
        }
    }

    pub fn new_table(
        key_name: Option<String>,
        key: EnrichedValueType,
        value_name: Option<String>,
        value: EnrichedValueType,
    ) -> Self {
        Self {
            kind: CollectionKind::Table,
            row: StructSchema {
                fields: Arc::new(vec![
                    FieldSchema {
                        name: key_name.unwrap_or_else(|| KEY_FIELD_NAME.to_string()),
                        value_type: key,
                    },
                    FieldSchema {
                        name: value_name.unwrap_or_else(|| VALUE_FIELD_NAME.to_string()),
                        value_type: value,
                    },
                ]),
            },
            collectors: Default::default(),
        }
    }

    pub fn new_list(value_name: Option<String>, value: EnrichedValueType) -> Self {
        Self {
            kind: CollectionKind::List,
            row: StructSchema {
                fields: Arc::new(vec![
                    LIST_INDEX_FIELD.clone(),
                    FieldSchema {
                        name: value_name.unwrap_or_else(|| VALUE_FIELD_NAME.to_string()),
                        value_type: value,
                    },
                ]),
            },
            collectors: Default::default(),
        }
    }

    pub fn is_table(&self) -> bool {
        match self.kind {
            CollectionKind::Collection => false,
            CollectionKind::Table | CollectionKind::List => true,
        }
    }

    pub fn is_list(&self) -> bool {
        self.kind == CollectionKind::List
    }

    pub fn key_field<'a>(&'a self) -> Option<&'a FieldSchema> {
        if self.is_table() {
            Some(self.row.fields.first().unwrap())
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind")]
pub enum ValueType {
    Struct(StructSchema),

    #[serde(untagged)]
    Basic(BasicValueType),

    #[serde(untagged)]
    Collection(CollectionSchema),
}

impl ValueType {
    pub fn key_type(&self) -> Option<&EnrichedValueType> {
        match self {
            ValueType::Basic(_) => None,
            ValueType::Struct(_) => None,
            ValueType::Collection(c) => c.key_type(),
        }
    }

    // Type equality, ignoring attributes.
    pub fn without_attrs(&self) -> Self {
        match self {
            ValueType::Basic(a) => ValueType::Basic(a.clone()),
            ValueType::Struct(a) => ValueType::Struct(a.without_attrs()),
            ValueType::Collection(a) => ValueType::Collection(a.without_attrs()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnrichedValueType<DataType = ValueType> {
    #[serde(rename = "type")]
    pub typ: DataType,

    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub nullable: bool,

    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attrs: Arc<BTreeMap<String, serde_json::Value>>,
}

impl EnrichedValueType {
    pub fn without_attrs(&self) -> Self {
        Self {
            typ: self.typ.without_attrs(),
            nullable: self.nullable,
            attrs: Default::default(),
        }
    }
}

impl<DataType> EnrichedValueType<DataType> {
    pub fn from_alternative<AltDataType>(
        value_type: &EnrichedValueType<AltDataType>,
    ) -> Result<Self>
    where
        for<'a> &'a AltDataType: TryInto<DataType, Error = anyhow::Error>,
    {
        Ok(Self {
            typ: (&value_type.typ).try_into()?,
            nullable: value_type.nullable,
            attrs: value_type.attrs.clone(),
        })
    }

    pub fn with_attr(mut self, key: &str, value: serde_json::Value) -> Self {
        Arc::make_mut(&mut self.attrs).insert(key.to_string(), value);
        self
    }
}

impl std::fmt::Display for EnrichedValueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.typ)?;
        if self.nullable {
            write!(f, "?")?;
        }
        if !self.attrs.is_empty() {
            write!(
                f,
                " [{}]",
                self.attrs
                    .iter()
                    .map(|(k, v)| format!("{k}: {v}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )?;
        }
        Ok(())
    }
}

impl std::fmt::Display for ValueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValueType::Basic(b) => write!(f, "{}", b),
            ValueType::Struct(s) => write!(f, "{}", s),
            ValueType::Collection(c) => write!(f, "{}", c),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FieldSchema<DataType = ValueType> {
    /// ID is used to identify the field in the schema.
    pub name: FieldName,

    #[serde(flatten)]
    pub value_type: EnrichedValueType<DataType>,
}

impl FieldSchema {
    pub fn without_attrs(&self) -> Self {
        Self {
            name: self.name.clone(),
            value_type: self.value_type.without_attrs(),
        }
    }
}

impl<DataType> FieldSchema<DataType> {
    pub fn from_alternative<AltDataType>(field: &FieldSchema<AltDataType>) -> Result<Self>
    where
        for<'a> &'a AltDataType: TryInto<DataType, Error = anyhow::Error>,
    {
        Ok(Self {
            name: field.name.clone(),
            value_type: EnrichedValueType::from_alternative(&field.value_type)?,
        })
    }
}

impl std::fmt::Display for FieldSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.name, self.value_type)
    }
}

/// Top-level schema for a flow instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSchema {
    pub schema: StructSchema,

    #[serde(default = "Vec::new", skip_serializing_if = "Vec::is_empty")]
    pub collectors: Vec<NamedSpec<StructSchema>>,
}

impl Deref for DataSchema {
    type Target = StructSchema;

    fn deref(&self) -> &Self::Target {
        &self.schema
    }
}

pub struct OpArgSchema {
    pub name: OpArgName,
    pub value_type: EnrichedValueType,
    pub analyzed_value: AnalyzedValueMapping,
}
