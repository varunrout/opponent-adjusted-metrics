terraform {
  backend "gcs" {
    bucket = "oam-varun-260819-tfstate"
    prefix = "production"
  }
}
