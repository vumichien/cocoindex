use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct Settings {
    pub database_url: String,
}
