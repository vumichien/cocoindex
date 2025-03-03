use crate::{api_bail, api_error};

use super::schema::*;
use anyhow::Result;
use base64::prelude::*;
use serde::{
    de::{SeqAccess, Visitor},
    ser::{SerializeSeq, SerializeTuple},
    Deserialize, Serialize,
};
use std::{collections::BTreeMap, ops::Deref, sync::Arc};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RangeValue {
    pub start: usize,
    pub end: usize,
}

impl RangeValue {
    pub fn new(start: usize, end: usize) -> Self {
        RangeValue { start, end }
    }
}

impl Serialize for RangeValue {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut tuple = serializer.serialize_tuple(2)?;
        tuple.serialize_element(&self.start)?;
        tuple.serialize_element(&self.end)?;
        tuple.end()
    }
}

impl<'de> Deserialize<'de> for RangeValue {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct RangeVisitor;

        impl<'de> Visitor<'de> for RangeVisitor {
            type Value = RangeValue;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a tuple of two u64")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Self::Value, V::Error>
            where
                V: SeqAccess<'de>,
            {
                let start = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::missing_field("missing begin"))?;
                let end = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::missing_field("missing end"))?;
                Ok(RangeValue { start, end })
            }
        }
        deserializer.deserialize_tuple(2, RangeVisitor)
    }
}

/// Value of key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum KeyValue {
    Bytes(Arc<[u8]>),
    Str(Arc<str>),
    Bool(bool),
    Int64(i64),
    Range(RangeValue),
    Struct(Vec<KeyValue>),
}

impl From<Arc<[u8]>> for KeyValue {
    fn from(value: Arc<[u8]>) -> Self {
        KeyValue::Bytes(value)
    }
}

impl From<Vec<u8>> for KeyValue {
    fn from(value: Vec<u8>) -> Self {
        KeyValue::Bytes(Arc::from(value))
    }
}

impl From<Arc<str>> for KeyValue {
    fn from(value: Arc<str>) -> Self {
        KeyValue::Str(value)
    }
}

impl From<String> for KeyValue {
    fn from(value: String) -> Self {
        KeyValue::Str(Arc::from(value))
    }
}

impl From<bool> for KeyValue {
    fn from(value: bool) -> Self {
        KeyValue::Bool(value)
    }
}

impl From<i64> for KeyValue {
    fn from(value: i64) -> Self {
        KeyValue::Int64(value)
    }
}

impl From<RangeValue> for KeyValue {
    fn from(value: RangeValue) -> Self {
        KeyValue::Range(value)
    }
}

impl From<Vec<KeyValue>> for KeyValue {
    fn from(value: Vec<KeyValue>) -> Self {
        KeyValue::Struct(value)
    }
}

impl serde::Serialize for KeyValue {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Value::from(self.clone()).serialize(serializer)
    }
}

impl KeyValue {
    fn parts_from_str(
        values_iter: &mut impl Iterator<Item = String>,
        schema: &ValueType,
    ) -> Result<Self> {
        let result = match schema {
            ValueType::Basic(basic_type) => {
                let v = values_iter
                    .next()
                    .ok_or_else(|| api_error!("Key parts less than expected"))?;
                match basic_type {
                    BasicValueType::Bytes { .. } => {
                        KeyValue::Bytes(Arc::from(BASE64_STANDARD.decode(v)?))
                    }
                    BasicValueType::Str { .. } => KeyValue::Str(Arc::from(v)),
                    BasicValueType::Bool => KeyValue::Bool(v.parse::<bool>()?),
                    BasicValueType::Int64 => KeyValue::Int64(v.parse::<i64>()?),
                    BasicValueType::Range => {
                        let v2 = values_iter
                            .next()
                            .ok_or_else(|| api_error!("Key parts less than expected"))?;
                        KeyValue::Range(RangeValue {
                            start: v.parse::<usize>()?,
                            end: v2.parse::<usize>()?,
                        })
                    }
                    schema => api_bail!("Invalid key type {schema}"),
                }
            }
            ValueType::Struct(s) => KeyValue::Struct(
                s.fields
                    .iter()
                    .map(|f| KeyValue::parts_from_str(values_iter, &f.value_type.typ))
                    .collect::<Result<Vec<_>>>()?,
            ),
            _ => api_bail!("Invalid key type {schema}"),
        };
        Ok(result)
    }

    pub fn from_strs(value: impl IntoIterator<Item = String>, schema: &ValueType) -> Result<Self> {
        let mut values_iter = value.into_iter();
        let result = Self::parts_from_str(&mut values_iter, schema)?;
        if values_iter.next().is_some() {
            api_bail!("Key parts more than expected");
        }
        Ok(result)
    }

    pub fn kind_str(&self) -> &'static str {
        match self {
            KeyValue::Bytes(_) => "bytes",
            KeyValue::Str(_) => "str",
            KeyValue::Bool(_) => "bool",
            KeyValue::Int64(_) => "int64",
            KeyValue::Range { .. } => "range",
            KeyValue::Struct(_) => "struct",
        }
    }

    pub fn bytes_value(&self) -> Result<&Arc<[u8]>> {
        match self {
            KeyValue::Bytes(v) => Ok(v),
            _ => anyhow::bail!("expected bytes value, but got {}", self.kind_str()),
        }
    }

    pub fn str_value(&self) -> Result<&Arc<str>> {
        match self {
            KeyValue::Str(v) => Ok(v),
            _ => anyhow::bail!("expected str value, but got {}", self.kind_str()),
        }
    }

    pub fn bool_value(&self) -> Result<bool> {
        match self {
            KeyValue::Bool(v) => Ok(*v),
            _ => anyhow::bail!("expected bool value, but got {}", self.kind_str()),
        }
    }

    pub fn int64_value(&self) -> Result<i64> {
        match self {
            KeyValue::Int64(v) => Ok(*v),
            _ => anyhow::bail!("expected int64 value, but got {}", self.kind_str()),
        }
    }

    pub fn range_value(&self) -> Result<RangeValue> {
        match self {
            KeyValue::Range(v) => Ok(*v),
            _ => anyhow::bail!("expected range value, but got {}", self.kind_str()),
        }
    }

    pub fn struct_value(&self) -> Result<&Vec<KeyValue>> {
        match self {
            KeyValue::Struct(v) => Ok(v),
            _ => anyhow::bail!("expected struct value, but got {}", self.kind_str()),
        }
    }
}

#[derive(Debug, Clone)]
pub enum BasicValue {
    Bytes(Arc<[u8]>),
    Str(Arc<str>),
    Bool(bool),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Range(RangeValue),
    Json(Arc<serde_json::Value>),
    Vector(Arc<[BasicValue]>),
}

impl From<Arc<[u8]>> for BasicValue {
    fn from(value: Arc<[u8]>) -> Self {
        BasicValue::Bytes(value)
    }
}

impl From<Vec<u8>> for BasicValue {
    fn from(value: Vec<u8>) -> Self {
        BasicValue::Bytes(Arc::from(value))
    }
}

impl From<Arc<str>> for BasicValue {
    fn from(value: Arc<str>) -> Self {
        BasicValue::Str(value)
    }
}

impl From<String> for BasicValue {
    fn from(value: String) -> Self {
        BasicValue::Str(Arc::from(value))
    }
}

impl From<bool> for BasicValue {
    fn from(value: bool) -> Self {
        BasicValue::Bool(value)
    }
}

impl From<i64> for BasicValue {
    fn from(value: i64) -> Self {
        BasicValue::Int64(value)
    }
}

impl From<f32> for BasicValue {
    fn from(value: f32) -> Self {
        BasicValue::Float32(value)
    }
}

impl From<f64> for BasicValue {
    fn from(value: f64) -> Self {
        BasicValue::Float64(value)
    }
}

impl From<serde_json::Value> for BasicValue {
    fn from(value: serde_json::Value) -> Self {
        BasicValue::Json(Arc::from(value))
    }
}

impl<T: Into<BasicValue>> From<Vec<T>> for BasicValue {
    fn from(value: Vec<T>) -> Self {
        BasicValue::Vector(Arc::from(
            value.into_iter().map(|v| v.into()).collect::<Vec<_>>(),
        ))
    }
}

impl BasicValue {
    pub fn to_key(self) -> Result<KeyValue> {
        let result = match self {
            BasicValue::Bytes(v) => KeyValue::Bytes(v),
            BasicValue::Str(v) => KeyValue::Str(v),
            BasicValue::Bool(v) => KeyValue::Bool(v),
            BasicValue::Int64(v) => KeyValue::Int64(v),
            BasicValue::Range(v) => KeyValue::Range(v),
            BasicValue::Float32(_)
            | BasicValue::Float64(_)
            | BasicValue::Json(_)
            | BasicValue::Vector(_) => api_bail!("invalid key value type"),
        };
        Ok(result)
    }

    pub fn as_key(&self) -> Result<KeyValue> {
        let result = match self {
            BasicValue::Bytes(v) => KeyValue::Bytes(v.clone()),
            BasicValue::Str(v) => KeyValue::Str(v.clone()),
            BasicValue::Bool(v) => KeyValue::Bool(*v),
            BasicValue::Int64(v) => KeyValue::Int64(*v),
            BasicValue::Range(v) => KeyValue::Range(*v),
            BasicValue::Float32(_)
            | BasicValue::Float64(_)
            | BasicValue::Json(_)
            | BasicValue::Vector(_) => api_bail!("invalid key value type"),
        };
        Ok(result)
    }

    pub fn kind(&self) -> &'static str {
        match &self {
            BasicValue::Bytes(_) => "bytes",
            BasicValue::Str(_) => "str",
            BasicValue::Bool(_) => "bool",
            BasicValue::Int64(_) => "int64",
            BasicValue::Float32(_) => "float32",
            BasicValue::Float64(_) => "float64",
            BasicValue::Range(_) => "range",
            BasicValue::Json(_) => "json",
            BasicValue::Vector(_) => "vector",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub enum Value<VS = ScopeValue> {
    #[default]
    Null,
    Basic(BasicValue),
    Struct(FieldValues<VS>),
    Collection(Vec<VS>),
    Table(BTreeMap<KeyValue, VS>),
    List(Vec<VS>),
}

impl<T: Into<BasicValue>> From<T> for Value {
    fn from(value: T) -> Self {
        Value::Basic(value.into())
    }
}

impl From<KeyValue> for Value {
    fn from(value: KeyValue) -> Self {
        match value {
            KeyValue::Bytes(v) => Value::Basic(BasicValue::Bytes(v)),
            KeyValue::Str(v) => Value::Basic(BasicValue::Str(v)),
            KeyValue::Bool(v) => Value::Basic(BasicValue::Bool(v)),
            KeyValue::Int64(v) => Value::Basic(BasicValue::Int64(v)),
            KeyValue::Range(v) => Value::Basic(BasicValue::Range(v)),
            KeyValue::Struct(v) => Value::Struct(FieldValues {
                fields: v.into_iter().map(|k| Value::from(k)).collect(),
            }),
        }
    }
}

impl From<FieldValues> for Value {
    fn from(value: FieldValues) -> Self {
        Value::Struct(value)
    }
}

impl<VS> Value<VS> {
    pub fn from_alternative<AltVS>(value: Value<AltVS>) -> Self
    where
        AltVS: Into<VS>,
    {
        match value {
            Value::Null => Value::Null,
            Value::Basic(v) => Value::Basic(v),
            Value::Struct(v) => Value::Struct(FieldValues::<VS> {
                fields: v
                    .fields
                    .into_iter()
                    .map(|v| Value::<VS>::from_alternative(v))
                    .collect(),
            }),
            Value::Collection(v) => Value::Collection(v.into_iter().map(|v| v.into()).collect()),
            Value::Table(v) => {
                Value::Table(v.into_iter().map(|(k, v)| (k.clone(), v.into())).collect())
            }
            Value::List(v) => Value::List(v.into_iter().map(|v| v.into()).collect()),
        }
    }

    pub fn from_alternative_ref<AltVS>(value: &Value<AltVS>) -> Self
    where
        for<'a> &'a AltVS: Into<VS>,
    {
        match value {
            Value::Null => Value::Null,
            Value::Basic(v) => Value::Basic(v.clone()),
            Value::Struct(v) => Value::Struct(FieldValues::<VS> {
                fields: v
                    .fields
                    .iter()
                    .map(|v| Value::<VS>::from_alternative_ref(v))
                    .collect(),
            }),
            Value::Collection(v) => Value::Collection(v.into_iter().map(|v| v.into()).collect()),
            Value::Table(v) => {
                Value::Table(v.into_iter().map(|(k, v)| (k.clone(), v.into())).collect())
            }
            Value::List(v) => Value::List(v.into_iter().map(|v| v.into()).collect()),
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    pub fn to_key(self) -> Result<KeyValue> {
        let result = match self {
            Value::Basic(v) => v.to_key()?,
            Value::Struct(v) => KeyValue::Struct(
                v.fields
                    .into_iter()
                    .map(|v| v.to_key())
                    .collect::<Result<Vec<_>>>()?,
            ),
            Value::Null | Value::Collection(_) | Value::Table(_) | Value::List(_) => {
                anyhow::bail!("invalid key value type")
            }
        };
        Ok(result)
    }

    pub fn as_key(&self) -> Result<KeyValue> {
        let result = match self {
            Value::Basic(v) => v.as_key()?,
            Value::Struct(v) => KeyValue::Struct(
                v.fields
                    .iter()
                    .map(|v| v.as_key())
                    .collect::<Result<Vec<_>>>()?,
            ),
            Value::Null | Value::Collection(_) | Value::Table(_) | Value::List(_) => {
                anyhow::bail!("invalid key value type")
            }
        };
        Ok(result)
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Basic(v) => v.kind(),
            Value::Struct(_) => "struct",
            Value::Collection(_) => "collection",
            Value::Table(_) => "table",
            Value::List(_) => "list",
        }
    }

    pub fn as_bytes(&self) -> Result<&Arc<[u8]>> {
        match self {
            Value::Basic(BasicValue::Bytes(v)) => Ok(v),
            _ => anyhow::bail!("expected bytes value, but got {}", self.kind()),
        }
    }

    pub fn as_str(&self) -> Result<&Arc<str>> {
        match self {
            Value::Basic(BasicValue::Str(v)) => Ok(v),
            _ => anyhow::bail!("expected str value, but got {}", self.kind()),
        }
    }

    pub fn as_bool(&self) -> Result<bool> {
        match self {
            Value::Basic(BasicValue::Bool(v)) => Ok(*v),
            _ => anyhow::bail!("expected bool value, but got {}", self.kind()),
        }
    }

    pub fn as_int64(&self) -> Result<i64> {
        match self {
            Value::Basic(BasicValue::Int64(v)) => Ok(*v),
            _ => anyhow::bail!("expected int64 value, but got {}", self.kind()),
        }
    }

    pub fn as_float32(&self) -> Result<f32> {
        match self {
            Value::Basic(BasicValue::Float32(v)) => Ok(*v),
            _ => anyhow::bail!("expected float32 value, but got {}", self.kind()),
        }
    }

    pub fn as_float64(&self) -> Result<f64> {
        match self {
            Value::Basic(BasicValue::Float64(v)) => Ok(*v),
            _ => anyhow::bail!("expected float64 value, but got {}", self.kind()),
        }
    }

    pub fn as_range(&self) -> Result<RangeValue> {
        match self {
            Value::Basic(BasicValue::Range(v)) => Ok(*v),
            _ => anyhow::bail!("expected range value, but got {}", self.kind()),
        }
    }

    pub fn as_json(&self) -> Result<&Arc<serde_json::Value>> {
        match self {
            Value::Basic(BasicValue::Json(v)) => Ok(v),
            _ => anyhow::bail!("expected json value, but got {}", self.kind()),
        }
    }

    pub fn as_vector(&self) -> Result<&Arc<[BasicValue]>> {
        match self {
            Value::Basic(BasicValue::Vector(v)) => Ok(v),
            _ => anyhow::bail!("expected vector value, but got {}", self.kind()),
        }
    }

    pub fn as_struct(&self) -> Result<&FieldValues<VS>> {
        match self {
            Value::Struct(v) => Ok(v),
            _ => anyhow::bail!("expected struct value, but got {}", self.kind()),
        }
    }

    pub fn as_collection(&self) -> Result<&Vec<VS>> {
        match self {
            Value::Collection(v) => Ok(v),
            _ => anyhow::bail!("expected collection value, but got {}", self.kind()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FieldValues<VS = ScopeValue> {
    pub fields: Vec<Value<VS>>,
}

impl serde::Serialize for FieldValues {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.fields.serialize(serializer)
    }
}

impl<VS: Clone> FieldValues<VS>
where
    FieldValues<VS>: Into<VS>,
{
    pub fn new(num_fields: usize) -> Self {
        let mut fields = Vec::with_capacity(num_fields);
        fields.resize(num_fields, Value::<VS>::Null);
        Self { fields }
    }

    pub fn from_json_values(
        values: impl Iterator<Item = serde_json::Value>,
        schema: &[FieldSchema],
    ) -> Result<Self> {
        let fields = values
            .zip(schema)
            .map(|(v, s)| {
                let value = Value::<VS>::from_json(v, &s.value_type.typ)?;
                if value.is_null() && !s.value_type.nullable {
                    api_bail!("expected non-null value for `{}`", s.name);
                }
                Ok(value)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { fields })
    }

    pub fn from_json(value: serde_json::Value, schema: &[FieldSchema]) -> Result<Self> {
        match value {
            serde_json::Value::Array(v) => {
                if v.len() != schema.len() {
                    api_bail!("unmatched value length");
                }
                Self::from_json_values(v.into_iter(), &schema)
            }
            _ => api_bail!("invalid value type"),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ScopeValue(pub FieldValues);

impl Deref for ScopeValue {
    type Target = FieldValues;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<FieldValues> for ScopeValue {
    fn from(value: FieldValues) -> Self {
        Self(value)
    }
}

impl serde::Serialize for BasicValue {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            BasicValue::Bytes(v) => serializer.serialize_str(&BASE64_STANDARD.encode(v)),
            BasicValue::Str(v) => serializer.serialize_str(v),
            BasicValue::Bool(v) => serializer.serialize_bool(*v),
            BasicValue::Int64(v) => serializer.serialize_i64(*v),
            BasicValue::Float32(v) => serializer.serialize_f32(*v),
            BasicValue::Float64(v) => serializer.serialize_f64(*v),
            BasicValue::Range(v) => v.serialize(serializer),
            BasicValue::Json(v) => v.serialize(serializer),
            BasicValue::Vector(v) => v.serialize(serializer),
        }
    }
}

impl BasicValue {
    pub fn from_json(value: serde_json::Value, schema: &BasicValueType) -> Result<Self> {
        let result = match (value, schema) {
            (serde_json::Value::String(v), BasicValueType::Bytes { .. }) => {
                BasicValue::Bytes(Arc::from(BASE64_STANDARD.decode(v)?))
            }
            (serde_json::Value::String(v), BasicValueType::Str { .. }) => {
                BasicValue::Str(Arc::from(v))
            }
            (serde_json::Value::Bool(v), BasicValueType::Bool) => BasicValue::Bool(v),
            (serde_json::Value::Number(v), BasicValueType::Int64) => BasicValue::Int64(
                v.as_i64()
                    .ok_or_else(|| anyhow::anyhow!("invalid int64 value {v}"))?,
            ),
            (serde_json::Value::Number(v), BasicValueType::Float32) => BasicValue::Float32(
                v.as_f64()
                    .ok_or_else(|| anyhow::anyhow!("invalid fp32 value {v}"))?
                    as f32,
            ),
            (serde_json::Value::Number(v), BasicValueType::Float64) => BasicValue::Float64(
                v.as_f64()
                    .ok_or_else(|| anyhow::anyhow!("invalid fp64 value {v}"))?,
            ),
            (v, BasicValueType::Range) => BasicValue::Range(serde_json::from_value(v)?),
            (v, BasicValueType::Json) => BasicValue::Json(Arc::from(v)),
            (
                serde_json::Value::Array(v),
                BasicValueType::Vector(VectorTypeSchema { element_type, .. }),
            ) => {
                let vec = v
                    .into_iter()
                    .map(|v| BasicValue::from_json(v, &element_type))
                    .collect::<Result<Vec<_>>>()?;
                BasicValue::Vector(Arc::from(vec))
            }
            (v, t) => {
                anyhow::bail!("Value and type not matched.\nTarget type {t:?}\nJSON value: {v}\n")
            }
        };
        Ok(result)
    }
}

struct TableEntry<'a>(&'a KeyValue, &'a ScopeValue);

impl serde::Serialize for Value<ScopeValue> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Value::Null => serializer.serialize_none(),
            Value::Basic(v) => v.serialize(serializer),
            Value::Struct(v) => v.serialize(serializer),
            Value::Collection(v) => v.serialize(serializer),
            Value::Table(m) => {
                let mut seq = serializer.serialize_seq(Some(m.len()))?;
                for (k, v) in m.iter() {
                    seq.serialize_element(&TableEntry(k, v))?;
                }
                seq.end()
            }
            Value::List(v) => v.serialize(serializer),
        }
    }
}

impl<'a> serde::Serialize for TableEntry<'a> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let &TableEntry(key, value) = self;
        let mut seq = serializer.serialize_seq(Some(value.0.fields.len() + 1))?;
        seq.serialize_element(key)?;
        for item in value.0.fields.iter() {
            seq.serialize_element(item)?;
        }
        seq.end()
    }
}

impl<VS: Clone> Value<VS>
where
    FieldValues<VS>: Into<VS>,
{
    pub fn from_json(value: serde_json::Value, schema: &ValueType) -> Result<Self> {
        let result = match (value, schema) {
            (serde_json::Value::Null, _) => Value::<VS>::Null,
            (v, ValueType::Basic(t)) => Value::<VS>::Basic(BasicValue::from_json(v, t)?),
            (v, ValueType::Struct(s)) => {
                Value::<VS>::Struct(FieldValues::<VS>::from_json(v, &s.fields)?)
            }
            (serde_json::Value::Array(v), ValueType::Collection(s)) => match s.kind {
                CollectionKind::Collection => {
                    let rows = v
                        .into_iter()
                        .map(|v| Ok(FieldValues::from_json(v, &s.row.fields)?.into()))
                        .collect::<Result<Vec<_>>>()?;
                    Value::List(rows)
                }
                CollectionKind::Table => {
                    let rows = v
                        .into_iter()
                        .map(|v| match v {
                            serde_json::Value::Array(v) => {
                                let mut fields_iter = v.into_iter();
                                let key = Self::from_json(
                                    fields_iter
                                        .next()
                                        .ok_or_else(|| api_error!("Empty struct field values"))?,
                                    &s.row.fields[0].value_type.typ,
                                )?
                                .to_key()?;
                                let values =
                                    FieldValues::from_json_values(fields_iter, &s.row.fields[1..])?
                                        .into();
                                Ok((key, values))
                            }
                            _ => api_bail!("Table value must be a JSON array"),
                        })
                        .collect::<Result<BTreeMap<_, _>>>()?;
                    Value::Table(rows)
                }
                CollectionKind::List => {
                    let rows = v
                        .into_iter()
                        .map(|v| Ok(FieldValues::from_json(v, &s.row.fields[1..])?.into()))
                        .collect::<Result<Vec<_>>>()?;
                    Value::List(rows)
                }
            },
            (v, t) => {
                anyhow::bail!("Value and type not matched.\nTarget type {t:?}\nJSON value: {v}\n")
            }
        };
        Ok(result)
    }
}
