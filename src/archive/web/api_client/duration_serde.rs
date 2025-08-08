use std::time::Duration;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    duration.as_millis().serialize(serializer)
}

pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
where
    D: Deserializer<'de>,
{
    let millis = u128::deserialize(deserializer)?;
    Ok(Duration::from_millis(millis as u64))
}
